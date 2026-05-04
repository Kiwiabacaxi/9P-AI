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
	}
}

// Individuo — um tour (permutação) com sua distância e custo já calculados.
//
// Distancia = soma "crua" das distâncias percorridas (km reais).
// MaxLeg    = maior trecho único do tour.
// Custo     = Distancia + λ · MaxLeg — é o valor que a seleção/elitismo MINIMIZAM.
//             Quando λ = 0, Custo == Distancia (idêntico ao TSP clássico).
type Individuo struct {
	Tour      []int   `json:"tour"`
	Distancia float64 `json:"distancia"`
	MaxLeg    float64 `json:"maxLeg"`
	Custo     float64 `json:"custo"`
}

// Step — payload por geração via SSE.
type Step struct {
	Geracao          int     `json:"geracao"`
	MelhorTour       []int   `json:"melhorTour"`
	MelhorDist       float64 `json:"melhorDist"`
	MelhorMaxLeg     float64 `json:"melhorMaxLeg"`
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

// avaliar — calcula distancia, maxLeg e custo (= distancia + lambda*maxLeg).
// É o que a seleção do AG efetivamente compara (minimizar Custo).
func avaliar(tour []int, matriz [][]float64, lambda float64) (distancia, maxLeg, custo float64) {
	if len(tour) < 2 {
		return
	}
	for i := 0; i < len(tour); i++ {
		d := matriz[tour[i]][tour[(i+1)%len(tour)]]
		distancia += d
		if d > maxLeg {
			maxLeg = d
		}
	}
	custo = distancia + lambda*maxLeg
	return
}

// =============================================================================
// População inicial — N permutações aleatórias.
// =============================================================================

func gerarPopulacaoInicial(rng *rand.Rand, popSize, n int, matriz [][]float64, lambda float64) []Individuo {
	pop := make([]Individuo, popSize)
	for i := range pop {
		tour := rng.Perm(n)
		pop[i] = Individuo{Tour: tour}
		pop[i].Distancia, pop[i].MaxLeg, pop[i].Custo = avaliar(tour, matriz, lambda)
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
	return Individuo{Tour: tour, Distancia: src.Distancia, MaxLeg: src.MaxLeg, Custo: src.Custo}
}

func cloneTour(t []int) []int {
	out := make([]int, len(t))
	copy(out, t)
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

func Treinar(progressCh chan<- Step, cfg Config, matriz [][]float64) Result {
	n := len(matriz)
	cfg = sanitizar(cfg, n)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	pop := gerarPopulacaoInicial(rng, cfg.PopSize, n, matriz, cfg.LambdaMaxLeg)

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
			progressCh <- Step{
				Geracao:          g + 1,
				MelhorTour:       cloneTour(pop[melhorIdx].Tour),
				MelhorDist:       melhorDist,
				MelhorMaxLeg:     pop[melhorIdx].MaxLeg,
				MelhorCusto:      pop[melhorIdx].Custo,
				MediaDist:        mediaDist,
				PiorDist:         piorDist,
				Diversidade:      div,
				MelhorGlobal:     cloneTour(melhorGlobal.Tour),
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
			f1.Distancia, f1.MaxLeg, f1.Custo = avaliar(t1, matriz, cfg.LambdaMaxLeg)
			f2 := Individuo{Tour: t2}
			f2.Distancia, f2.MaxLeg, f2.Custo = avaliar(t2, matriz, cfg.LambdaMaxLeg)
			filhos = append(filhos, f1, f2)
		}
		if len(filhos) > precisamos {
			filhos = filhos[:precisamos]
		}
		pop = append(elites, filhos...)
	}

	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorTour:      cloneTour(melhorGlobal.Tour),
		MelhorDist:      melhorGlobal.Distancia,
		MelhorMaxLeg:    melhorGlobal.MaxLeg,
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
	return cfg
}

// =============================================================================
// Presets — datasets prontos pra demonstração.
// =============================================================================

// TrianguloMineiro — 20 cidades do Triângulo Mineiro / Alto Paranaíba (MG).
//
// Cenário de logística real: Centro de Distribuição em Uberlândia (id 0) que
// precisa atender lojas/clientes em 19 cidades vizinhas. Distâncias na ordem
// de 50-300 km — escala real de roteirização de frota terrestre.
//
// É uma região estratégica do agronegócio brasileiro: Mosaic Fertilizantes em
// Araxá, frigoríficos JBS/Marfrig em Uberlândia/Uberaba, redes regionais de
// supermercado (Bretas/Mais Mart), distribuição de combustíveis e bebidas.
// Coleta de leite cooperada (CCPR/Itambé) na região tem o mesmo padrão.
//
// Coordenadas: centro aproximado de cada cidade (lat/lng).
func TrianguloMineiro() []Cidade {
	return []Cidade{
		{ID: 0, Nome: "Uberlândia", UF: "MG", Lat: -18.9128, Lng: -48.2755},
		{ID: 1, Nome: "Uberaba", UF: "MG", Lat: -19.7479, Lng: -47.9381},
		{ID: 2, Nome: "Araxá", UF: "MG", Lat: -19.5933, Lng: -46.9406},
		{ID: 3, Nome: "Araguari", UF: "MG", Lat: -18.6443, Lng: -48.1864},
		{ID: 4, Nome: "Patos de Minas", UF: "MG", Lat: -18.5789, Lng: -46.5181},
		{ID: 5, Nome: "Patrocínio", UF: "MG", Lat: -18.9442, Lng: -46.9931},
		{ID: 6, Nome: "Frutal", UF: "MG", Lat: -20.0247, Lng: -48.9408},
		{ID: 7, Nome: "Ituiutaba", UF: "MG", Lat: -18.9742, Lng: -49.4634},
		{ID: 8, Nome: "Monte Carmelo", UF: "MG", Lat: -18.7250, Lng: -47.4983},
		{ID: 9, Nome: "Tupaciguara", UF: "MG", Lat: -18.5944, Lng: -48.7050},
		{ID: 10, Nome: "Coromandel", UF: "MG", Lat: -18.4731, Lng: -47.1944},
		{ID: 11, Nome: "São Gotardo", UF: "MG", Lat: -19.3119, Lng: -46.0497},
		{ID: 12, Nome: "Iturama", UF: "MG", Lat: -19.7283, Lng: -50.1969},
		{ID: 13, Nome: "Sacramento", UF: "MG", Lat: -19.8650, Lng: -47.4378},
		{ID: 14, Nome: "Conceição das Alagoas", UF: "MG", Lat: -19.9119, Lng: -48.3858},
		{ID: 15, Nome: "Monte Alegre de Minas", UF: "MG", Lat: -18.8689, Lng: -48.8769},
		{ID: 16, Nome: "Capinópolis", UF: "MG", Lat: -18.6822, Lng: -49.5697},
		{ID: 17, Nome: "Prata", UF: "MG", Lat: -19.3072, Lng: -48.9264},
		{ID: 18, Nome: "Ibiá", UF: "MG", Lat: -19.4736, Lng: -46.5400},
		{ID: 19, Nome: "Campina Verde", UF: "MG", Lat: -19.5394, Lng: -49.4858},
	}
}

func CapitaisBR() []Cidade {
	return []Cidade{
		{ID: 0, Nome: "Aracaju", UF: "SE", Lat: -10.9472, Lng: -37.0731},
		{ID: 1, Nome: "Belém", UF: "PA", Lat: -1.4558, Lng: -48.5039},
		{ID: 2, Nome: "Belo Horizonte", UF: "MG", Lat: -19.9167, Lng: -43.9345},
		{ID: 3, Nome: "Boa Vista", UF: "RR", Lat: 2.8235, Lng: -60.6758},
		{ID: 4, Nome: "Brasília", UF: "DF", Lat: -15.7942, Lng: -47.8825},
		{ID: 5, Nome: "Campo Grande", UF: "MS", Lat: -20.4697, Lng: -54.6201},
		{ID: 6, Nome: "Cuiabá", UF: "MT", Lat: -15.6014, Lng: -56.0979},
		{ID: 7, Nome: "Curitiba", UF: "PR", Lat: -25.4284, Lng: -49.2733},
		{ID: 8, Nome: "Florianópolis", UF: "SC", Lat: -27.5949, Lng: -48.5482},
		{ID: 9, Nome: "Fortaleza", UF: "CE", Lat: -3.7172, Lng: -38.5434},
		{ID: 10, Nome: "Goiânia", UF: "GO", Lat: -16.6869, Lng: -49.2648},
		{ID: 11, Nome: "João Pessoa", UF: "PB", Lat: -7.1153, Lng: -34.8610},
		{ID: 12, Nome: "Macapá", UF: "AP", Lat: 0.0349, Lng: -51.0694},
		{ID: 13, Nome: "Maceió", UF: "AL", Lat: -9.6498, Lng: -35.7089},
		{ID: 14, Nome: "Manaus", UF: "AM", Lat: -3.1190, Lng: -60.0217},
		{ID: 15, Nome: "Natal", UF: "RN", Lat: -5.7945, Lng: -35.2110},
		{ID: 16, Nome: "Palmas", UF: "TO", Lat: -10.1689, Lng: -48.3317},
		{ID: 17, Nome: "Porto Alegre", UF: "RS", Lat: -30.0346, Lng: -51.2177},
		{ID: 18, Nome: "Porto Velho", UF: "RO", Lat: -8.7619, Lng: -63.9039},
		{ID: 19, Nome: "Recife", UF: "PE", Lat: -8.0476, Lng: -34.8770},
		{ID: 20, Nome: "Rio Branco", UF: "AC", Lat: -9.9747, Lng: -67.8243},
		{ID: 21, Nome: "Rio de Janeiro", UF: "RJ", Lat: -22.9068, Lng: -43.1729},
		{ID: 22, Nome: "Salvador", UF: "BA", Lat: -12.9714, Lng: -38.5014},
		{ID: 23, Nome: "São Luís", UF: "MA", Lat: -2.5391, Lng: -44.2829},
		{ID: 24, Nome: "São Paulo", UF: "SP", Lat: -23.5505, Lng: -46.6333},
		{ID: 25, Nome: "Teresina", UF: "PI", Lat: -5.0892, Lng: -42.8019},
		{ID: 26, Nome: "Vitória", UF: "ES", Lat: -20.3155, Lng: -40.3128},
	}
}
