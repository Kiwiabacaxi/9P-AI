package tsp

import (
	"math"
	"math/rand"
	"sort"
	"strconv"
)

// =============================================================================
// Algoritmo Genético — Caixeiro Viajante (TSP) — Aula 12
//
// Aplicação real do AG: encontrar o tour mais curto que visita N cidades e
// volta à origem. Problema NP-difícil (N! tours possíveis), onde busca exaustiva
// é inviável já com ~15-20 cidades, mas o AG acha tours bons em segundos.
//
// O encoding é uma PERMUTAÇÃO de índices: tour = [c0, c1, ..., cN-1], que
// significa visitar c0 → c1 → ... → cN-1 → c0. Crossover de bit-string como
// nas aulas 10-11 quebra aqui (gera permutações inválidas com cidades
// repetidas e faltando), por isso usamos operadores específicos pra
// permutação: OX (Order Crossover) e PMX (Partially Mapped Crossover).
//
// É o exemplo do "outros tipos de codificação de cromossomo" do slide aula 12.
// =============================================================================

// Constantes — métodos selecionáveis.
const (
	DistEuclidiana = "euclidiana" // lat/lng tratados como plano (didático)
	DistHaversine  = "haversine"  // great-circle (km reais)
	DistOSRM       = "osrm"       // matriz vinda do OSRM (rota por estradas)

	SelRoleta  = "roleta"
	SelTorneio = "torneio"

	CrossOX  = "ox"  // Order Crossover (Davis 1985)
	CrossPMX = "pmx" // Partially Mapped Crossover (Goldberg & Lingle 1985)

	MutSwap     = "swap"     // troca duas cidades
	MutInversao = "inversao" // reverte um segmento (2-opt-like)
)

// Cidade — um ponto a ser visitado.
type Cidade struct {
	ID   int     `json:"id"`
	Nome string  `json:"nome"`
	UF   string  `json:"uf,omitempty"`
	Lat  float64 `json:"lat"`
	Lng  float64 `json:"lng"`
}

// Config — todos os hiperparâmetros do AG.
type Config struct {
	PopSize        int     `json:"popSize"`
	MaxGeracoes    int     `json:"maxGeracoes"`
	ProbCruzamento float64 `json:"probCruzamento"`
	ProbMutacao    float64 `json:"probMutacao"`
	Selecao        string  `json:"selecao"`        // "roleta" | "torneio"
	TamanhoTorneio int     `json:"tamanhoTorneio"`
	Cruzamento     string  `json:"cruzamento"`     // "ox" | "pmx"
	Mutacao        string  `json:"mutacao"`        // "swap" | "inversao"
	Elitismo       int     `json:"elitismo"`

	// LambdaMaxLeg — "tempero" opcional na função de fitness:
	// custo = distancia_total + λ · max_leg
	// λ = 0 → TSP puro (só minimiza distância total).
	// λ > 0 → penaliza tours com algum trecho muito longo (caso real:
	//         autonomia do veículo, descanso obrigatório do motorista, etc.).
	LambdaMaxLeg float64 `json:"lambdaMaxLeg"`

	// LastVisit — restrição lógica do cenário: id da cidade que DEVE ser
	// visitada por último, imediatamente antes do retorno ao depot.
	// -1 = sem restrição.
	//
	// Caso típico: rota Cargill da soja deve descarregar no Porto de Santos
	// no FIM (depois de coletar nos silos), não no meio. O TSP puro não
	// modela ordem; sem essa penalidade, o GA acha rotas tipo
	// "depot → porto vazio → silos → depot cheio" que são absurdas.
	//
	// Implementação: penalidade proporcional ao desvio cíclico entre a
	// posição atual de LastVisit e a posição "logo antes do depot".
	LastVisit int `json:"lastVisit"`

	// Gamma — peso do tempo total na fitness, em "km equivalentes por hora".
	// γ = 0 → ignora tempo (TSP só de distância);
	// γ > 0 → cada hora de tour custa γ km na fitness — útil pra cold-chain
	// (leite/carne) onde o tempo importa mais que a quilometragem.
	Gamma float64 `json:"gamma"`

	// JornadaMaxSec — jornada máxima do motorista em segundos.
	// Default 36000 = 10h (limite ANTT). Usado junto com MuOvertime.
	JornadaMaxSec float64 `json:"jornadaMaxSec"`

	// MuOvertime — coef. da penalidade quadrática por exceder a jornada.
	// μ = 0 → desliga a penalidade;
	// μ > 0 → cada hora além da jornada custa μ · h² na fitness — força o
	// GA a achar tours que cabem num "shift" único, ou a aceitar
	// pernoites/troca de motorista (refletido no overtime explosivo).
	MuOvertime float64 `json:"muOvertime"`

	Seed int64 `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		PopSize:        80,
		MaxGeracoes:    300,
		ProbCruzamento: 0.85,
		ProbMutacao:    0.15,
		Selecao:        SelTorneio,
		TamanhoTorneio: 4,
		Cruzamento:     CrossOX,
		Mutacao:        MutInversao,
		Elitismo:       2,
		LambdaMaxLeg:   0,
		LastVisit:      -1,
		Gamma:          0,
		JornadaMaxSec:  36000, // 10h
		MuOvertime:     0,
	}
}

// Individuo — um tour (permutação) com sua distância, tempo e custo já calculados.
//
// Distancia = soma "crua" das distâncias percorridas (km reais).
// MaxLeg    = maior trecho único do tour.
// TempoSec  = tempo total dirigindo (segundos) — vem da matriz de duração
//             (real do OSRM ou sintetizada de distância via 70 km/h).
// Custo     = soma de termos da fitness — é o valor que a seleção/elitismo MINIMIZAM:
//
//             Custo = Distancia
//                   + λ · MaxLeg                   (penaliza trecho gigante)
//                   + ω · desvio(lastVisit)        (restrição de ordem)
//                   + γ · (TempoSec / 3600)        (peso do tempo em km equivalentes)
//                   + μ · max(0, T - Tmax)²        (overtime quadrático)
type Individuo struct {
	Tour      []int   `json:"tour"`
	Distancia float64 `json:"distancia"`
	MaxLeg    float64 `json:"maxLeg"`
	TempoSec  float64 `json:"tempoSec"`
	Custo     float64 `json:"custo"`
}

// VelMediaKmH — velocidade média assumida pra sintetizar duração quando
// a matriz de origem é Haversine (não temos dado de OSRM real). 70 km/h é
// uma estimativa razoável pra rotas mistas BR (rodovia + acesso urbano).
const VelMediaKmH = 70.0

// SintetizarMatrizDuracao — converte matriz de distâncias (km) em duração
// (segundos) usando velocidade média constante. Usar quando matDuracao real
// não está disponível (modos Haversine, fallback OSRM).
func SintetizarMatrizDuracao(matDist [][]float64) [][]float64 {
	n := len(matDist)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			out[i][j] = matDist[i][j] / VelMediaKmH * 3600.0
		}
	}
	return out
}

// Step — payload por geração via SSE.
type Step struct {
	Geracao          int     `json:"geracao"`
	MelhorTour       []int   `json:"melhorTour"`
	MelhorDist       float64 `json:"melhorDist"`
	MelhorMaxLeg     float64 `json:"melhorMaxLeg"`
	MelhorTempoSec   float64 `json:"melhorTempoSec"`
	MelhorCusto      float64 `json:"melhorCusto"`
	MediaDist        float64 `json:"mediaDist"`
	PiorDist         float64 `json:"piorDist"`
	Diversidade      int     `json:"diversidade"`
	MelhorGlobal     []int   `json:"melhorGlobal"`
	MelhorGlobalDist float64 `json:"melhorGlobalDist"`
}

type Result struct {
	Geracoes        int       `json:"geracoes"`
	MelhorTour      []int     `json:"melhorTour"`
	MelhorDist      float64   `json:"melhorDist"`
	MelhorMaxLeg    float64   `json:"melhorMaxLeg"`
	MelhorTempoSec  float64   `json:"melhorTempoSec"`
	MelhorCusto     float64   `json:"melhorCusto"`
	HistMelhor      []float64 `json:"histMelhor"`
	HistMedia       []float64 `json:"histMedia"`
	HistDiversidade []int     `json:"histDiversidade"`
	Cfg             Config    `json:"cfg"`
}

// =============================================================================
// Distance helpers
// =============================================================================

// haversineKm — distância de great-circle em km entre dois pontos lat/lng.
func haversineKm(latA, lngA, latB, lngB float64) float64 {
	const R = 6371.0 // raio médio da Terra em km
	rad := func(d float64) float64 { return d * math.Pi / 180 }
	dLat := rad(latB - latA)
	dLng := rad(lngB - lngA)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(rad(latA))*math.Cos(rad(latB))*math.Sin(dLng/2)*math.Sin(dLng/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

// euclidiana — distância "plana" tratando lat/lng como x/y. Apenas didática
// (distorce em escalas grandes). A distância dada está em "graus".
func euclidiana(latA, lngA, latB, lngB float64) float64 {
	dLat := latB - latA
	dLng := lngB - lngA
	return math.Sqrt(dLat*dLat + dLng*dLng)
}

// CalcularMatrizDistancias — gera a matriz N×N entre cidades.
// (modo OSRM é tratado fora deste pacote — o backend faz a chamada e injeta
// a matriz pronta no Treinar.)
func CalcularMatrizDistancias(cidades []Cidade, modo string) [][]float64 {
	n := len(cidades)
	matriz := make([][]float64, n)
	for i := range matriz {
		matriz[i] = make([]float64, n)
	}
	fn := haversineKm
	if modo == DistEuclidiana {
		fn = euclidiana
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := fn(cidades[i].Lat, cidades[i].Lng, cidades[j].Lat, cidades[j].Lng)
			matriz[i][j] = d
			matriz[j][i] = d
		}
	}
	return matriz
}

// =============================================================================
// Fitness
// =============================================================================

// CalcularDistanciaTour — soma das distâncias percorridas no tour fechado.
func CalcularDistanciaTour(tour []int, matriz [][]float64) float64 {
	if len(tour) < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < len(tour); i++ {
		from := tour[i]
		to := tour[(i+1)%len(tour)]
		total += matriz[from][to]
	}
	return total
}

// avaliar — calcula distancia, maxLeg, tempo e custo. Custo é o que a seleção
// do AG minimiza:
//
//	custo = distancia
//	      + λ · maxLeg                  (penaliza trecho gigante)
//	      + ω · desvio(lastVisit)       (restrição de ordem)
//	      + γ · (T / 3600)              (peso do tempo em km equivalentes)
//	      + μ · max(0, T - Tmax)²       (overtime quadrático em h²)
//
// Termos com matDur == nil (modo Euclidiano) são ignorados — distância em
// graus não traduz pra unidades de tempo de jeito coerente.
func avaliar(
	tour []int,
	matDist, matDur [][]float64,
	lambda, gamma, mu, jornadaMaxSec float64,
	lastVisit int,
) (distancia, maxLeg, tempoSec, custo float64) {
	n := len(tour)
	if n < 2 {
		return
	}
	for i := 0; i < n; i++ {
		from, to := tour[i], tour[(i+1)%n]
		d := matDist[from][to]
		distancia += d
		if d > maxLeg {
			maxLeg = d
		}
		if matDur != nil {
			tempoSec += matDur[from][to]
		}
	}
	custo = distancia + lambda*maxLeg

	if lastVisit >= 0 {
		// Posição do depot (cidade 0) e da cidade-última.
		depotPos, lvPos := -1, -1
		for i, c := range tour {
			if c == 0 {
				depotPos = i
			}
			if c == lastVisit {
				lvPos = i
			}
		}
		if depotPos >= 0 && lvPos >= 0 {
			// "Posição ideal" da cidade-última: imediatamente antes do depot
			// no ciclo. Em coordenadas: (depotPos - 1 + n) mod n.
			wanted := (depotPos - 1 + n) % n
			// Distância cíclica mínima (forward ou backward).
			fwd := (wanted - lvPos + n) % n
			bwd := (lvPos - wanted + n) % n
			deviation := fwd
			if bwd < fwd {
				deviation = bwd
			}
			avgLeg := distancia / float64(n)
			custo += float64(deviation) * 2.0 * avgLeg
		}
	}

	// γ · T (tempo em horas)
	if gamma > 0 && matDur != nil {
		custo += gamma * (tempoSec / 3600.0)
	}

	// μ · max(0, T - Tmax)² (overtime em h²)
	if mu > 0 && matDur != nil && jornadaMaxSec > 0 {
		excedeSec := tempoSec - jornadaMaxSec
		if excedeSec > 0 {
			excedeH := excedeSec / 3600.0
			custo += mu * excedeH * excedeH
		}
	}

	return
}

// =============================================================================
// População inicial — N permutações aleatórias.
// =============================================================================

func gerarPopulacaoInicial(
	rng *rand.Rand, popSize, n int,
	matDist, matDur [][]float64,
	lambda, gamma, mu, jornadaMaxSec float64,
	lastVisit int,
) []Individuo {
	pop := make([]Individuo, popSize)
	for i := range pop {
		tour := rng.Perm(n)
		pop[i] = Individuo{Tour: tour}
		pop[i].Distancia, pop[i].MaxLeg, pop[i].TempoSec, pop[i].Custo =
			avaliar(tour, matDist, matDur, lambda, gamma, mu, jornadaMaxSec, lastVisit)
	}
	return pop
}

// =============================================================================
// Seleção
// =============================================================================

// fitnessRoleta — converte custo em "fitness positivo" pra roleta.
// Como queremos MINIMIZAR custo, usamos (maxCusto - custo) + ε.
func fitnessRoleta(pop []Individuo) []float64 {
	n := len(pop)
	out := make([]float64, n)
	maxCusto := pop[0].Custo
	for _, ind := range pop {
		if ind.Custo > maxCusto {
			maxCusto = ind.Custo
		}
	}
	for i, ind := range pop {
		out[i] = (maxCusto - ind.Custo) + 1e-6
	}
	return out
}

func selecionarRoleta(pop []Individuo, cumul []float64, rng *rand.Rand) Individuo {
	r := rng.Float64() * cumul[len(cumul)-1]
	idx := 0
	for idx < len(cumul)-1 && cumul[idx] < r {
		idx++
	}
	return clonarIndividuo(pop[idx])
}

func selecionarTorneio(pop []Individuo, k int, rng *rand.Rand) (Individuo, Individuo) {
	if k > len(pop) {
		k = len(pop)
	}
	if k < 2 {
		k = 2
	}
	perm := rng.Perm(len(pop))[:k]
	sort.Slice(perm, func(i, j int) bool {
		return pop[perm[i]].Custo < pop[perm[j]].Custo
	})
	return clonarIndividuo(pop[perm[0]]), clonarIndividuo(pop[perm[1]])
}

// =============================================================================
// Cruzamento — versões específicas pra permutação.
//
// Por que não dá pra usar 1pt/2pt como nas aulas 10-11? Permutação requer que
// cada cidade apareça EXATAMENTE uma vez. Cortar e trocar bits gera filhos
// inválidos com cidades repetidas e faltando. Os operadores abaixo preservam
// a propriedade "permutação válida" by design.
// =============================================================================

// OX (Order Crossover):
//   1) copia o segmento p1[cut1:cut2] no filho.
//   2) preenche os slots vazios com as cidades restantes em ordem em que
//      aparecem em p2, começando da posição cut2 (com wrap).
func cruzamentoOX(p1, p2 []int, cut1, cut2 int) []int {
	n := len(p1)
	if cut1 > cut2 {
		cut1, cut2 = cut2, cut1
	}
	child := make([]int, n)
	inChild := make(map[int]bool, n)

	for i := cut1; i < cut2; i++ {
		child[i] = p1[i]
		inChild[p1[i]] = true
	}

	// cidades restantes na ordem de p2 a partir de cut2 (com wrap)
	remaining := make([]int, 0, n-(cut2-cut1))
	for i := 0; i < n; i++ {
		c := p2[(cut2+i)%n]
		if !inChild[c] {
			remaining = append(remaining, c)
		}
	}

	// posições vazias na ordem a partir de cut2 (com wrap, pulando o segmento)
	j := 0
	for i := 0; i < n; i++ {
		pos := (cut2 + i) % n
		if pos >= cut1 && pos < cut2 {
			continue
		}
		child[pos] = remaining[j]
		j++
	}
	return child
}

// PMX (Partially Mapped Crossover):
//   1) copia o segmento p1[cut1:cut2] no filho.
//   2) pra cada cidade c de p2 no segmento que ainda não está no filho:
//      seguimos a "cadeia de mapeamento" p1[i] → posição em p2 até cair fora
//      do segmento — coloca c naquela posição.
//   3) preenche as posições restantes com p2 direto.
func cruzamentoPMX(p1, p2 []int, cut1, cut2 int) []int {
	n := len(p1)
	if cut1 > cut2 {
		cut1, cut2 = cut2, cut1
	}
	child := make([]int, n)
	for i := range child {
		child[i] = -1
	}

	inSegment := make(map[int]bool)
	for i := cut1; i < cut2; i++ {
		child[i] = p1[i]
		inSegment[p1[i]] = true
	}

	posInP2 := make(map[int]int, n)
	for i, c := range p2 {
		posInP2[c] = i
	}
	for i := cut1; i < cut2; i++ {
		c := p2[i]
		if inSegment[c] {
			continue
		}
		// segue a cadeia
		target := p1[i]
		targetPos := posInP2[target]
		for targetPos >= cut1 && targetPos < cut2 {
			target = p1[targetPos]
			targetPos = posInP2[target]
		}
		child[targetPos] = c
	}

	for i := 0; i < n; i++ {
		if child[i] == -1 {
			child[i] = p2[i]
		}
	}
	return child
}

// =============================================================================
// Mutação — sempre mantendo a validade da permutação.
// =============================================================================

func mutacaoSwap(tour []int, prob float64, rng *rand.Rand) {
	if len(tour) < 2 {
		return
	}
	if rng.Float64() < prob {
		i := rng.Intn(len(tour))
		j := rng.Intn(len(tour))
		tour[i], tour[j] = tour[j], tour[i]
	}
}

// mutacaoInversao — escolhe um intervalo [i, j] e reverte. Equivalente a um
// movimento 2-opt — muito mais efetivo que swap pra TSP, porque conserta
// "cruzamentos" no tour de uma vez.
func mutacaoInversao(tour []int, prob float64, rng *rand.Rand) {
	n := len(tour)
	if n < 2 {
		return
	}
	if rng.Float64() < prob {
		i := rng.Intn(n)
		j := rng.Intn(n)
		if i > j {
			i, j = j, i
		}
		for x, y := i, j; x < y; x, y = x+1, y-1 {
			tour[x], tour[y] = tour[y], tour[x]
		}
	}
}

// =============================================================================
// Helpers
// =============================================================================

func clonarIndividuo(src Individuo) Individuo {
	tour := make([]int, len(src.Tour))
	copy(tour, src.Tour)
	return Individuo{
		Tour:      tour,
		Distancia: src.Distancia,
		MaxLeg:    src.MaxLeg,
		TempoSec:  src.TempoSec,
		Custo:     src.Custo,
	}
}

func cloneTour(t []int) []int {
	out := make([]int, len(t))
	copy(out, t)
	return out
}

// RotateToStart — rotaciona o tour cíclico pra que `startCity` (id) fique
// na posição 0. O tour fechado é matematicamente o mesmo (rotação não muda
// distância), só muda onde começamos a leitura.
//
// Importante porque o GA encontra um ciclo, mas a narrativa de logística
// precisa do depot fixo no início (caminhão sai de X, faz a rota, volta a X).
func RotateToStart(tour []int, startCity int) []int {
	if len(tour) == 0 {
		return tour
	}
	startIdx := -1
	for i, c := range tour {
		if c == startCity {
			startIdx = i
			break
		}
	}
	if startIdx <= 0 {
		return tour
	}
	n := len(tour)
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = tour[(startIdx+i)%n]
	}
	return out
}

func extrairElites(pop []Individuo, p int) []Individuo {
	if p <= 0 {
		return nil
	}
	if p > len(pop) {
		p = len(pop)
	}
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(i, j int) bool {
		return pop[idxs[i]].Custo < pop[idxs[j]].Custo
	})
	elites := make([]Individuo, p)
	for i := 0; i < p; i++ {
		elites[i] = clonarIndividuo(pop[idxs[i]])
	}
	return elites
}

// diversidade — quantos tours únicos (por hash da sequência) na população.
func diversidade(pop []Individuo) int {
	seen := make(map[string]struct{}, len(pop))
	for _, ind := range pop {
		seen[tourKey(ind.Tour)] = struct{}{}
	}
	return len(seen)
}

func tourKey(tour []int) string {
	b := make([]byte, 0, len(tour)*4)
	for _, c := range tour {
		b = strconv.AppendInt(b, int64(c), 10)
		b = append(b, ',')
	}
	return string(b)
}

// =============================================================================
// Treinar — orquestra o AG, emite Step por geração via canal.
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config, matDist, matDur [][]float64) Result {
	n := len(matDist)
	cfg = sanitizar(cfg, n)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	pop := gerarPopulacaoInicial(
		rng, cfg.PopSize, n,
		matDist, matDur,
		cfg.LambdaMaxLeg, cfg.Gamma, cfg.MuOvertime, cfg.JornadaMaxSec,
		cfg.LastVisit,
	)

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)
	histDiv := make([]int, 0, cfg.MaxGeracoes)

	melhorGlobal := Individuo{Distancia: math.Inf(1), Custo: math.Inf(1)}

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// estatísticas (todas baseadas em Custo, que é o que a seleção minimiza)
		melhorIdx := 0
		soma := 0.0
		piorDist := 0.0
		for i, ind := range pop {
			soma += ind.Distancia
			if ind.Custo < pop[melhorIdx].Custo {
				melhorIdx = i
			}
			if ind.Distancia > piorDist {
				piorDist = ind.Distancia
			}
		}
		melhorDist := pop[melhorIdx].Distancia
		mediaDist := soma / float64(len(pop))
		div := diversidade(pop)
		histMelhor = append(histMelhor, melhorDist)
		histMedia = append(histMedia, mediaDist)
		histDiv = append(histDiv, div)

		if pop[melhorIdx].Custo < melhorGlobal.Custo {
			melhorGlobal = clonarIndividuo(pop[melhorIdx])
		}

		if progressCh != nil {
			// Ancorar o tour no depot (cidade 0) — não muda a soma cíclica,
			// mas garante consistência narrativa (o depot é o início da leitura).
			melhorTourAnc := RotateToStart(cloneTour(pop[melhorIdx].Tour), 0)
			melhorGlobalAnc := RotateToStart(cloneTour(melhorGlobal.Tour), 0)
			progressCh <- Step{
				Geracao:          g + 1,
				MelhorTour:       melhorTourAnc,
				MelhorDist:       melhorDist,
				MelhorMaxLeg:     pop[melhorIdx].MaxLeg,
				MelhorTempoSec:   pop[melhorIdx].TempoSec,
				MelhorCusto:      pop[melhorIdx].Custo,
				MediaDist:        mediaDist,
				PiorDist:         piorDist,
				Diversidade:      div,
				MelhorGlobal:     melhorGlobalAnc,
				MelhorGlobalDist: melhorGlobal.Distancia,
			}
		}

		// === próxima geração ===
		elites := extrairElites(pop, cfg.Elitismo)
		precisamos := cfg.PopSize - len(elites)
		numCasais := (precisamos + 1) / 2

		var cumul []float64
		if cfg.Selecao == SelRoleta {
			fits := fitnessRoleta(pop)
			cumul = make([]float64, len(fits))
			soma := 0.0
			for i, f := range fits {
				soma += f
				cumul[i] = soma
			}
		}

		filhos := make([]Individuo, 0, 2*numCasais)
		for c := 0; c < numCasais; c++ {
			var paiA, paiB Individuo
			if cfg.Selecao == SelTorneio {
				paiA, paiB = selecionarTorneio(pop, cfg.TamanhoTorneio, rng)
			} else {
				paiA = selecionarRoleta(pop, cumul, rng)
				paiB = selecionarRoleta(pop, cumul, rng)
			}

			var t1, t2 []int
			if rng.Float64() < cfg.ProbCruzamento {
				cut1 := rng.Intn(n)
				cut2 := rng.Intn(n)
				if cut1 > cut2 {
					cut1, cut2 = cut2, cut1
				}
				if cut1 == cut2 {
					cut2 = (cut2 + 1) % n
					if cut1 > cut2 {
						cut1, cut2 = cut2, cut1
					}
				}
				if cfg.Cruzamento == CrossPMX {
					t1 = cruzamentoPMX(paiA.Tour, paiB.Tour, cut1, cut2)
					t2 = cruzamentoPMX(paiB.Tour, paiA.Tour, cut1, cut2)
				} else {
					t1 = cruzamentoOX(paiA.Tour, paiB.Tour, cut1, cut2)
					t2 = cruzamentoOX(paiB.Tour, paiA.Tour, cut1, cut2)
				}
			} else {
				t1 = cloneTour(paiA.Tour)
				t2 = cloneTour(paiB.Tour)
			}

			// mutação só nos filhos (Obs 03 da aula 10)
			if cfg.Mutacao == MutSwap {
				mutacaoSwap(t1, cfg.ProbMutacao, rng)
				mutacaoSwap(t2, cfg.ProbMutacao, rng)
			} else {
				mutacaoInversao(t1, cfg.ProbMutacao, rng)
				mutacaoInversao(t2, cfg.ProbMutacao, rng)
			}

			f1 := Individuo{Tour: t1}
			f1.Distancia, f1.MaxLeg, f1.TempoSec, f1.Custo = avaliar(
				t1, matDist, matDur,
				cfg.LambdaMaxLeg, cfg.Gamma, cfg.MuOvertime, cfg.JornadaMaxSec,
				cfg.LastVisit,
			)
			f2 := Individuo{Tour: t2}
			f2.Distancia, f2.MaxLeg, f2.TempoSec, f2.Custo = avaliar(
				t2, matDist, matDur,
				cfg.LambdaMaxLeg, cfg.Gamma, cfg.MuOvertime, cfg.JornadaMaxSec,
				cfg.LastVisit,
			)
			filhos = append(filhos, f1, f2)
		}
		if len(filhos) > precisamos {
			filhos = filhos[:precisamos]
		}
		pop = append(elites, filhos...)
	}

	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorTour:      RotateToStart(cloneTour(melhorGlobal.Tour), 0),
		MelhorDist:      melhorGlobal.Distancia,
		MelhorMaxLeg:    melhorGlobal.MaxLeg,
		MelhorTempoSec:  melhorGlobal.TempoSec,
		MelhorCusto:     melhorGlobal.Custo,
		HistMelhor:      histMelhor,
		HistMedia:       histMedia,
		HistDiversidade: histDiv,
		Cfg:             cfg,
	}
}

func sanitizar(cfg Config, n int) Config {
	if cfg.PopSize < 4 {
		cfg.PopSize = 4
	}
	if cfg.PopSize%2 != 0 {
		cfg.PopSize++
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 100
	}
	if cfg.ProbCruzamento < 0 {
		cfg.ProbCruzamento = 0
	}
	if cfg.ProbCruzamento > 1 {
		cfg.ProbCruzamento = 1
	}
	if cfg.ProbMutacao < 0 {
		cfg.ProbMutacao = 0
	}
	if cfg.ProbMutacao > 1 {
		cfg.ProbMutacao = 1
	}
	if cfg.Selecao != SelRoleta && cfg.Selecao != SelTorneio {
		cfg.Selecao = SelTorneio
	}
	if cfg.Cruzamento != CrossOX && cfg.Cruzamento != CrossPMX {
		cfg.Cruzamento = CrossOX
	}
	if cfg.Mutacao != MutSwap && cfg.Mutacao != MutInversao {
		cfg.Mutacao = MutInversao
	}
	if cfg.TamanhoTorneio < 2 {
		cfg.TamanhoTorneio = 2
	}
	if cfg.TamanhoTorneio > cfg.PopSize {
		cfg.TamanhoTorneio = cfg.PopSize
	}
	if cfg.Elitismo < 0 {
		cfg.Elitismo = 0
	}
	if cfg.Elitismo >= cfg.PopSize {
		cfg.Elitismo = cfg.PopSize - 2
	}
	if cfg.LambdaMaxLeg < 0 {
		cfg.LambdaMaxLeg = 0
	}
	if cfg.LastVisit < -1 || cfg.LastVisit >= n {
		cfg.LastVisit = -1
	}
	if cfg.LastVisit == 0 {
		// 0 é o depot — não pode ser também a "última visita"
		cfg.LastVisit = -1
	}
	if cfg.Gamma < 0 {
		cfg.Gamma = 0
	}
	if cfg.MuOvertime < 0 {
		cfg.MuOvertime = 0
	}
	if cfg.JornadaMaxSec <= 0 {
		cfg.JornadaMaxSec = 36000 // 10h default
	}
	return cfg
}

// =============================================================================
// Presets — cenários temáticos com narrativa de logística real.
//
// Cada preset é um pacote: cidades + origem (depot) + narrativa explicando
// a operação real que serve de inspiração + parâmetros de fitness sugeridos
// pra aquele cenário. A ideia é que o usuário escolha um cenário e veja o
// AG resolvendo um problema com cara de problema real, não um TSP abstrato.
//
// Empresas e operações citadas são reais — coordenadas são aproximações do
// centro municipal (suficiente pra calcular distâncias e desenhar mapa).
// =============================================================================

// Preset — um cenário temático completo.
type Preset struct {
	ID             string   `json:"id"`
	Nome           string   `json:"nome"`
	Descricao      string   `json:"descricao"`      // 1 linha pro select
	Narrativa      string   `json:"narrativa"`      // texto multi-parágrafo
	Origem         string   `json:"origem"`         // nome do depot (cidade ID 0)
	Cidades        []Cidade `json:"cidades"`
	LambdaSugerido float64  `json:"lambdaSugerido"` // tempero recomendado pra esse cenário
	ModoSugerido   string   `json:"modoSugerido"`   // "haversine" | "osrm" | "euclidiana"
	FitnessNota    string   `json:"fitnessNota"`    // por que esse λ + esse modo aqui

	// LastVisit — restrição lógica do cenário: id da cidade que deve ser
	// visitada por último, imediatamente antes do retorno ao depot.
	// -1 = sem restrição. Ex: pra Cargill, é o Porto de Santos
	// (caminhão coleta soja nos silos primeiro, descarrega no porto por último).
	LastVisit     int    `json:"lastVisit"`
	LastVisitNome string `json:"lastVisitNome,omitempty"`

	// GammaSugerido — peso do tempo recomendado pra esse cenário (km/h equiv).
	// Cold-chain (leite, carne) → alto. Peso de carga (fertilizante, milho) → 0.
	GammaSugerido float64 `json:"gammaSugerido"`

	// MuOvertimeSugerido — coef. da penalidade quadrática de jornada > Tmax
	// (km/h² equivalente). Long-haul (Cargill, ~24h) → alto.
	MuOvertimeSugerido float64 `json:"muOvertimeSugerido"`
}

// Presets — lista todos os cenários disponíveis.
func Presets() []Preset {
	return []Preset{
		presetItambe(),
		presetMosaic(),
		presetJBS(),
		presetCargillSoja(),
		presetADMMilho(),
	}
}

// GetPreset — busca por ID. Retorna nil se não existe.
func GetPreset(id string) *Preset {
	for _, p := range Presets() {
		if p.ID == id {
			return &p
		}
	}
	return nil
}

// =============================================================================
// 1) Itambé · Coleta de Leite (Triângulo Mineiro)
// =============================================================================

func presetItambe() Preset {
	return Preset{
		ID:        "itambe-leite",
		Nome:      "Itambé · Coleta de Leite",
		Descricao: "Caminhão refrigerado coleta leite cru em fazendas do Triângulo",
		Origem:    "Laticínio Itambé Uberaba (CCPR)",
		Narrativa: "Cooperativa Central dos Produtores Rurais (CCPR), dona da marca Itambé, " +
			"opera unidade real em Uberaba que recebe leite cru de centenas de fazendas " +
			"associadas no Triângulo Mineiro. Caminhões refrigerados saem do laticínio, " +
			"passam por uma rota de fazendas coletando o leite do dia e voltam pra " +
			"unidade de pasteurização. " +
			"\n\n" +
			"Aqui a rota inclui o laticínio + 12 cidades-município reais ao redor " +
			"(Sacramento, Conceição das Alagoas, Frutal, Veríssimo, etc.) representando " +
			"pontos de coleta. A escala (50–250 km) é a típica do dia-a-dia.",
		LambdaSugerido: 1.5,
		ModoSugerido:   "osrm",
		FitnessNota: "Leite tem cadeia fria — γ alto (60 km/h equiv) prioriza tempo " +
			"sobre quilometragem (ar-condicionado consome combustível por hora, e leite " +
			"estraga). λ > 0 penaliza tour com algum trecho muito longo. Modo OSRM " +
			"porque caminhão segue rodovias — nada de cortar reto pelo pasto.",
		LastVisit:          -1,
		GammaSugerido:      60,
		MuOvertimeSugerido: 30,
		Cidades: []Cidade{
			{ID: 0, Nome: "Itambé Uberaba", UF: "MG", Lat: -19.7479, Lng: -47.9381},
			{ID: 1, Nome: "Conceição das Alagoas", UF: "MG", Lat: -19.9119, Lng: -48.3858},
			{ID: 2, Nome: "Veríssimo", UF: "MG", Lat: -19.6803, Lng: -48.3083},
			{ID: 3, Nome: "Sacramento", UF: "MG", Lat: -19.8650, Lng: -47.4378},
			{ID: 4, Nome: "Frutal", UF: "MG", Lat: -20.0247, Lng: -48.9408},
			{ID: 5, Nome: "Planura", UF: "MG", Lat: -20.1389, Lng: -48.6772},
			{ID: 6, Nome: "Comendador Gomes", UF: "MG", Lat: -19.6919, Lng: -49.1408},
			{ID: 7, Nome: "Pirajuba", UF: "MG", Lat: -19.9025, Lng: -48.6936},
			{ID: 8, Nome: "Água Comprida", UF: "MG", Lat: -20.0344, Lng: -48.0850},
			{ID: 9, Nome: "Conquista", UF: "MG", Lat: -19.9281, Lng: -47.5575},
			{ID: 10, Nome: "Delta", UF: "MG", Lat: -19.9747, Lng: -47.7864},
			{ID: 11, Nome: "Campo Florido", UF: "MG", Lat: -19.7711, Lng: -48.5667},
			{ID: 12, Nome: "Igarapava (SP)", UF: "SP", Lat: -20.0397, Lng: -47.7461},
		},
	}
}

// =============================================================================
// 2) Mosaic · Distribuição de Fertilizantes (Cerrado Mineiro)
// =============================================================================

func presetMosaic() Preset {
	return Preset{
		ID:        "mosaic-fertilizante",
		Nome:      "Mosaic · Fertilizantes pelo Cerrado",
		Descricao: "Caminhão sai da mineração de Araxá pra fazendas do Alto Paranaíba",
		Origem:    "Mosaic Fertilizantes Araxá",
		Narrativa: "Mosaic Fertilizantes opera complexo industrial real em Araxá-MG, com " +
			"mineração de rocha fosfática e produção de fertilizantes (MAP, TSP, " +
			"superfosfato). O produto sai dali em caminhões pesados pra fazendas de " +
			"soja, milho e café espalhadas pelo Cerrado Mineiro / Alto Paranaíba. " +
			"\n\n" +
			"Esta rota cobre 11 municípios produtores reais da região (Patos de Minas, " +
			"Patrocínio, São Gotardo, Coromandel, Tapira, Perdizes, etc.). Tapira-MG " +
			"inclusive abriga outra mineração de fosfato da Vale.",
		LambdaSugerido: 0,
		ModoSugerido:   "osrm",
		FitnessNota: "Fertilizante é volumoso e pesado — o que importa é minimizar quilometragem total " +
			"(diesel é o maior custo operacional). λ = 0 e γ = 0 — TSP puro de distância " +
			"funciona perfeito. OSRM essencial: Vanderléia de 30+ ton só anda em BR.",
		LastVisit:          -1,
		GammaSugerido:      0,
		MuOvertimeSugerido: 0,
		Cidades: []Cidade{
			{ID: 0, Nome: "Mosaic Araxá", UF: "MG", Lat: -19.5933, Lng: -46.9406},
			{ID: 1, Nome: "Patos de Minas", UF: "MG", Lat: -18.5789, Lng: -46.5181},
			{ID: 2, Nome: "Patrocínio", UF: "MG", Lat: -18.9442, Lng: -46.9931},
			{ID: 3, Nome: "São Gotardo", UF: "MG", Lat: -19.3119, Lng: -46.0497},
			{ID: 4, Nome: "Coromandel", UF: "MG", Lat: -18.4731, Lng: -47.1944},
			{ID: 5, Nome: "Ibiá", UF: "MG", Lat: -19.4736, Lng: -46.5400},
			{ID: 6, Nome: "Sacramento", UF: "MG", Lat: -19.8650, Lng: -47.4378},
			{ID: 7, Nome: "Monte Carmelo", UF: "MG", Lat: -18.7250, Lng: -47.4983},
			{ID: 8, Nome: "Iraí de Minas", UF: "MG", Lat: -18.9756, Lng: -47.4639},
			{ID: 9, Nome: "Tapira", UF: "MG", Lat: -19.9114, Lng: -46.8253},
			{ID: 10, Nome: "Pratinha", UF: "MG", Lat: -19.7314, Lng: -46.9608},
			{ID: 11, Nome: "Perdizes", UF: "MG", Lat: -19.3458, Lng: -47.2961},
		},
	}
}

// =============================================================================
// 3) JBS · Carne pra Capitais (longa distância)
// =============================================================================

func presetJBS() Preset {
	return Preset{
		ID:        "jbs-carne",
		Nome:      "JBS · Carne pra Capitais",
		Descricao: "Frigorífico em Uberlândia distribui carne pras capitais regionais",
		Origem:    "JBS Friboi Uberlândia",
		Narrativa: "JBS opera unidade real em Uberlândia (Friboi) que abate gado e processa " +
			"carne resfriada/congelada pra distribuição. Caminhões refrigerados saem " +
			"diariamente pra centros consumidores nas capitais regionais (varejo + atacarejo). " +
			"\n\n" +
			"Esta rota inclui 10 capitais ao alcance de 1-2 dias de viagem terrestre saindo " +
			"de Uberlândia: BH, Brasília, Goiânia, SP, RJ, Curitiba, Cuiabá, Campo Grande, " +
			"Salvador (mais distante mas tem baldeação), Vitória.",
		LambdaSugerido: 2,
		ModoSugerido:   "osrm",
		FitnessNota: "Cadeia fria + longa distância. γ alto (50 km/h equiv) porque carne " +
			"refrigerada estraga em horas — tempo manda. λ alto (2) penaliza tours com " +
			"algum trecho monstruoso. μ alto (50) porque tour > 10h obriga troca de " +
			"motorista (descanso ANTT). OSRM porque caminhão refrigerado de 20 ton tem " +
			"rotas obrigatórias por BR.",
		LastVisit:          -1,
		GammaSugerido:      50,
		MuOvertimeSugerido: 50,
		Cidades: []Cidade{
			{ID: 0, Nome: "JBS Uberlândia", UF: "MG", Lat: -18.9128, Lng: -48.2755},
			{ID: 1, Nome: "Belo Horizonte", UF: "MG", Lat: -19.9167, Lng: -43.9345},
			{ID: 2, Nome: "Brasília", UF: "DF", Lat: -15.7942, Lng: -47.8825},
			{ID: 3, Nome: "Goiânia", UF: "GO", Lat: -16.6869, Lng: -49.2648},
			{ID: 4, Nome: "Cuiabá", UF: "MT", Lat: -15.6014, Lng: -56.0979},
			{ID: 5, Nome: "Campo Grande", UF: "MS", Lat: -20.4697, Lng: -54.6201},
			{ID: 6, Nome: "São Paulo", UF: "SP", Lat: -23.5505, Lng: -46.6333},
			{ID: 7, Nome: "Rio de Janeiro", UF: "RJ", Lat: -22.9068, Lng: -43.1729},
			{ID: 8, Nome: "Curitiba", UF: "PR", Lat: -25.4284, Lng: -49.2733},
			{ID: 9, Nome: "Vitória", UF: "ES", Lat: -20.3155, Lng: -40.3128},
			{ID: 10, Nome: "Salvador", UF: "BA", Lat: -12.9714, Lng: -38.5014},
		},
	}
}

// =============================================================================
// 4) Cargill · Soja → Porto de Santos (corredor MT → SP)
// =============================================================================

func presetCargillSoja() Preset {
	return Preset{
		ID:        "cargill-soja",
		Nome:      "Cargill · Soja → Porto de Santos",
		Descricao: "Caminhão coleta soja em silos do MT e descarrega no Porto de Santos",
		Origem:    "Cargill Rondonópolis (terminal de soja)",
		Narrativa: "Cargill opera complexo real em Rondonópolis-MT — um dos maiores terminais de " +
			"recepção e armazenamento de soja do país. Caminhões saem de Rondonópolis pra " +
			"silos parceiros em outras cidades produtoras do MT (Sorriso, Sinop, Lucas do " +
			"Rio Verde, Sapezal — o famoso \"corredor da soja\"), passam por hubs de " +
			"transbordo (Cuiabá), seguem pelo interior de SP (Ribeirão Preto, Campinas) " +
			"até descarregar no <b>Porto de Santos</b>, principal porta de saída do agro " +
			"brasileiro pro mercado externo. " +
			"\n\n" +
			"Esta rota cobre 11 pontos reais. Distâncias na ordem de 1500-2500 km no total — " +
			"frete \"long-haul\" típico do agronegócio. A rota física é fechada (caminhão " +
			"volta a Rondonópolis depois de descarregar) porque é o que o TSP modela.",
		LambdaSugerido: 1,
		ModoSugerido:   "osrm",
		FitnessNota: "Soja é peso → distância manda; γ baixo (20 km/h equiv) só pra " +
			"considerar tempo no equilíbrio. μ ALTO (80 km/h² equiv) porque o tour bate " +
			"~24h dirigindo — explode a fitness, forçando o GA a aceitar que essa rota " +
			"NÃO cabe em uma jornada única (tem que trocar motorista, pernoitar, ou usar " +
			"comboio). É o famoso problema real do agro brasileiro. " +
			"Restrição extra: Porto de Santos é o último ponto antes de voltar a Rondonópolis.",
		LastVisit:          10,
		LastVisitNome:      "Porto de Santos",
		GammaSugerido:      20,
		MuOvertimeSugerido: 80,
		Cidades: []Cidade{
			{ID: 0, Nome: "Cargill Rondonópolis", UF: "MT", Lat: -16.4706, Lng: -54.6353},
			{ID: 1, Nome: "Sorriso", UF: "MT", Lat: -12.5450, Lng: -55.7211},
			{ID: 2, Nome: "Sinop", UF: "MT", Lat: -11.8642, Lng: -55.5028},
			{ID: 3, Nome: "Lucas do Rio Verde", UF: "MT", Lat: -13.0506, Lng: -55.9114},
			{ID: 4, Nome: "Sapezal", UF: "MT", Lat: -12.9892, Lng: -58.7642},
			{ID: 5, Nome: "Cuiabá", UF: "MT", Lat: -15.6014, Lng: -56.0979},
			{ID: 6, Nome: "Goiânia", UF: "GO", Lat: -16.6869, Lng: -49.2648},
			{ID: 7, Nome: "Uberlândia", UF: "MG", Lat: -18.9128, Lng: -48.2755},
			{ID: 8, Nome: "Ribeirão Preto", UF: "SP", Lat: -21.1775, Lng: -47.8103},
			{ID: 9, Nome: "Campinas", UF: "SP", Lat: -22.9099, Lng: -47.0626},
			{ID: 10, Nome: "Porto de Santos", UF: "SP", Lat: -23.9353, Lng: -46.3258},
		},
	}
}

// =============================================================================
// 5) ADM · Coleta de Milho no Triângulo (cenário regional)
// =============================================================================

func presetADMMilho() Preset {
	return Preset{
		ID:        "adm-milho",
		Nome:      "ADM · Coleta de Milho",
		Descricao: "Caminhão da ADM coleta milho em fazendas do Triângulo Mineiro",
		Origem:    "ADM Uberlândia (silo + esmagadora)",
		Narrativa: "Archer Daniels Midland (ADM) opera unidade real em Uberlândia-MG — silo + " +
			"esmagadora de soja/milho que abastece o complexo agroindustrial brasileiro. " +
			"Caminhões saem da unidade e fazem rota por fazendas e silos parceiros em " +
			"cidades vizinhas, recolhendo milho na safra. " +
			"\n\n" +
			"Esta rota cobre 11 cidades-município reais do Triângulo Mineiro / Alto " +
			"Paranaíba (Patrocínio, Patos de Minas, Coromandel, Monte Carmelo, Araguari, " +
			"Tupaciguara, Ituiutaba, Capinópolis, Frutal, Iturama). Escala 50-300 km, " +
			"caminhão volta no mesmo dia.",
		LambdaSugerido: 1,
		ModoSugerido:   "osrm",
		FitnessNota: "Milho a granel é peso, minimizar km importa. λ = 1 modesto pra " +
			"penalizar trecho fora da curva — o motorista volta no mesmo dia, então cada " +
			"km a mais empurra a janela de descarga e diminui ciclos por dia. " +
			"γ baixo (15 km/h equiv) porque o tempo importa mas não tanto quanto pra " +
			"cold-chain.",
		LastVisit:          -1,
		GammaSugerido:      15,
		MuOvertimeSugerido: 0,
		Cidades: []Cidade{
			{ID: 0, Nome: "ADM Uberlândia", UF: "MG", Lat: -18.9128, Lng: -48.2755},
			{ID: 1, Nome: "Patrocínio", UF: "MG", Lat: -18.9442, Lng: -46.9931},
			{ID: 2, Nome: "Patos de Minas", UF: "MG", Lat: -18.5789, Lng: -46.5181},
			{ID: 3, Nome: "Coromandel", UF: "MG", Lat: -18.4731, Lng: -47.1944},
			{ID: 4, Nome: "Monte Carmelo", UF: "MG", Lat: -18.7250, Lng: -47.4983},
			{ID: 5, Nome: "Araguari", UF: "MG", Lat: -18.6443, Lng: -48.1864},
			{ID: 6, Nome: "Tupaciguara", UF: "MG", Lat: -18.5944, Lng: -48.7050},
			{ID: 7, Nome: "Ituiutaba", UF: "MG", Lat: -18.9742, Lng: -49.4634},
			{ID: 8, Nome: "Capinópolis", UF: "MG", Lat: -18.6822, Lng: -49.5697},
			{ID: 9, Nome: "Frutal", UF: "MG", Lat: -20.0247, Lng: -48.9408},
			{ID: 10, Nome: "Iturama", UF: "MG", Lat: -19.7283, Lng: -50.1969},
		},
	}
}
