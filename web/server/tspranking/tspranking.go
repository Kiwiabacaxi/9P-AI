package tspranking

import (
	"math"
	"math/rand"
	"sort"
)

// =============================================================================
// Algoritmo Genético — Caixeiro Viajante (TSP) com seleção por RANKING
// Trabalho 14 — Aulas 13 (Caixeiro Viajante) + 16 (AG com Ranking)
//
// O TSP busca o tour mais curto que parte de Uberaba, visita as outras 9 cidades
// do Triângulo Mineiro uma única vez e volta. O cromossomo é uma PERMUTAÇÃO de
// índices de cidade; cruzamento de bit-string quebra (gera cidade repetida/faltando),
// então usamos operadores de permutação: OX e PMX. Mutação troca/inverte cidades.
//
// O FOCO do Trabalho 14 é a SELEÇÃO POR RANKING (Aula 16): a probabilidade de um
// indivíduo ser pai NÃO depende do valor absoluto do fitness, mas apenas da sua
// POSIÇÃO no ranking (1 = melhor). Isso combate convergência prematura, domínio
// dos extremamente aptos e perda de diversidade — problemas da seleção
// proporcional clássica (roleta). Implementamos:
//
//   • Ranking LINEAR (Baker, 1985):
//       P_i = (1/N)·[η_max − (η_max − η_min)·(i−1)/(N−1)],  η_min = 2 − η_max
//   • Ranking EXPONENCIAL:
//       P_i = c^(N−i) / Σ_j c^(N−j),  c > 1
//
// e mantemos Torneio e Roleta clássicos selecionáveis, só para comparação.
// =============================================================================

// Métodos selecionáveis.
const (
	SelRankingLinear = "rankingLinear"
	SelRankingExp    = "rankingExp"
	SelTorneio       = "torneio"
	SelRoleta        = "roleta"

	CrossOX  = "ox"  // Order Crossover (Davis, 1985)
	CrossPMX = "pmx" // Partially Mapped Crossover (Goldberg & Lingle, 1985)

	MutSwap     = "swap"     // troca duas cidades (Aula 13)
	MutInversao = "inversao" // inverte um segmento (2-opt-like)
)

// Cidade — um ponto a ser visitado.
type Cidade struct {
	ID   int     `json:"id"`
	Nome string  `json:"nome"`
	UF   string  `json:"uf"`
	Lat  float64 `json:"lat"`
	Lng  float64 `json:"lng"`
}

// Config — hiperparâmetros do AG.
type Config struct {
	PopSize        int     `json:"popSize"`
	MaxGeracoes    int     `json:"maxGeracoes"`
	ProbCruzamento float64 `json:"probCruzamento"`
	ProbMutacao    float64 `json:"probMutacao"`
	Selecao        string  `json:"selecao"` // ranking{Linear,Exp} | torneio | roleta
	TamanhoTorneio int     `json:"tamanhoTorneio"`
	EtaMax         float64 `json:"etaMax"` // ranking linear: pressão máxima (η_min = 2 − η_max)
	CExp           float64 `json:"cExp"`   // ranking exponencial: base c > 1
	Cruzamento     string  `json:"cruzamento"` // ox | pmx
	Mutacao        string  `json:"mutacao"`    // swap | inversao
	Elitismo       int     `json:"elitismo"`
	Seed           int64   `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		PopSize:        80,
		MaxGeracoes:    250,
		ProbCruzamento: 0.9,
		ProbMutacao:    0.2,
		Selecao:        SelRankingLinear,
		TamanhoTorneio: 4,
		EtaMax:         1.5,
		CExp:           1.07,
		Cruzamento:     CrossOX,
		Mutacao:        MutSwap,
		Elitismo:       2,
	}
}

// Individuo — um tour (permutação fechada) com sua distância total.
type Individuo struct {
	Tour      []int   `json:"tour"`
	Distancia float64 `json:"distancia"`
}

// Step — payload por geração via SSE.
type Step struct {
	Geracao          int       `json:"geracao"`
	MelhorTour       []int     `json:"melhorTour"`
	MelhorDist       float64   `json:"melhorDist"`
	MediaDist        float64   `json:"mediaDist"`
	PiorDist         float64   `json:"piorDist"`
	Diversidade      int       `json:"diversidade"`
	MelhorGlobalTour []int     `json:"melhorGlobalTour"`
	MelhorGlobalDist float64   `json:"melhorGlobalDist"`
	// PopDist — distâncias da população ORDENADAS asc (rank 1..N). Alimenta o
	// "Laboratório do Ranking" no frontend (mostra o fitness real por posição).
	PopDist []float64 `json:"popDist"`
}

// Result — devolvido ao final.
type Result struct {
	Geracoes        int       `json:"geracoes"`
	MelhorTour      []int     `json:"melhorTour"`
	MelhorDist      float64   `json:"melhorDist"`
	HistMelhor      []float64 `json:"histMelhor"`
	HistMedia       []float64 `json:"histMedia"`
	HistDiversidade []int     `json:"histDiversidade"`
	Cfg             Config    `json:"cfg"`
}

// =============================================================================
// Probabilidades de seleção por RANKING (Aula 16)
// Devolvem P_1..P_N indexadas por (rank−1); rank 1 = melhor indivíduo.
// =============================================================================

// ProbsRankingLinear — Baker (1985):
//
//	P_i = (1/N)·[η_max − (η_max − η_min)·(i−1)/(N−1)],  com η_min = 2 − η_max.
//
// η_max ∈ [1,2] controla a pressão seletiva. Conferido com o slide (N=5,
// η_max=1.5 → [0.30, 0.25, 0.20, 0.15, 0.10]).
func ProbsRankingLinear(n int, etaMax float64) []float64 {
	probs := make([]float64, n)
	if n <= 0 {
		return probs
	}
	if n == 1 {
		probs[0] = 1
		return probs
	}
	etaMin := 2 - etaMax
	for i := 1; i <= n; i++ {
		probs[i-1] = (1.0 / float64(n)) * (etaMax - (etaMax-etaMin)*float64(i-1)/float64(n-1))
	}
	return probs
}

// ProbsRankingExp — ranking exponencial:
//
//	P_i = c^(N−i) / Σ_j c^(N−j),  c > 1.
//
// c maior = mais pressão. Conferido com o slide (N=5, c=2 → pesos [16,8,4,2,1]/31).
func ProbsRankingExp(n int, c float64) []float64 {
	probs := make([]float64, n)
	if n <= 0 {
		return probs
	}
	if n == 1 {
		probs[0] = 1
		return probs
	}
	if c <= 1 {
		c = 1.0 + 1e-6
	}
	pesos := make([]float64, n)
	soma := 0.0
	for i := 1; i <= n; i++ {
		w := math.Pow(c, float64(n-i))
		pesos[i-1] = w
		soma += w
	}
	for i := range pesos {
		probs[i] = pesos[i] / soma
	}
	return probs
}

// =============================================================================
// Mapa: 10 cidades reais do Triângulo Mineiro + matriz de distâncias.
// A matriz usa a TABELA da Aula 13 (slide 9) onde existe; os pares "—" são
// preenchidos por Haversine·fator (fator = mediana de tabela/Haversine sobre os
// pares conhecidos), pra completar o grafo na MESMA escala de km da tabela.
// =============================================================================

// Índices: 0 = Uberaba (depósito/partida).
var cidadesFixas = []Cidade{
	{0, "Uberaba", "MG", -19.7472, -47.9381},
	{1, "Uberlândia", "MG", -18.9186, -48.2772},
	{2, "Araguari", "MG", -18.6473, -48.1873},
	{3, "Ituiutaba", "MG", -18.9741, -49.4647},
	{4, "Patos de Minas", "MG", -18.5789, -46.5181},
	{5, "Frutal", "MG", -20.0245, -48.9406},
	{6, "Araxá", "MG", -19.5933, -46.9403},
	{7, "Monte Carmelo", "MG", -18.7256, -47.4992},
	{8, "Tupaciguara", "MG", -18.5922, -48.7049},
	{9, "Campina Verde", "MG", -19.5353, -49.4869},
}

// tabelaAula13 — pares (i<j) com distância conhecida na tabela do slide 9 (km).
var tabelaAula13 = []struct {
	i, j int
	km   float64
}{
	{0, 1, 106}, {0, 4, 265}, {0, 5, 105}, {0, 6, 110}, {0, 9, 160},
	{1, 2, 30}, {1, 3, 138}, {1, 4, 190}, {1, 6, 175}, {1, 7, 89}, {1, 8, 56},
	{2, 3, 117}, {2, 4, 221}, {2, 6, 205}, {2, 7, 66}, {2, 8, 47},
	{3, 5, 265}, {3, 7, 163}, {3, 9, 186},
	{4, 6, 137}, {4, 7, 114},
	{5, 6, 185}, {5, 9, 75},
	{6, 7, 145},
	{7, 8, 111},
}

// Mapa — cidades + matriz simétrica + origem de cada distância.
type Mapa struct {
	Cidades []Cidade    `json:"cidades"`
	Matriz  [][]float64 `json:"matriz"`
	Fonte   [][]bool    `json:"fonte"` // true = tabela Aula 13; false = preenchido (Haversine·fator)
	Fator   float64     `json:"fator"` // fator de calibração estrada/reta usado nos preenchidos
}

// ConstruirMapa monta o mapa completo do cenário do Triângulo Mineiro.
func ConstruirMapa() Mapa {
	n := len(cidadesFixas)
	mat := make([][]float64, n)
	fonte := make([][]bool, n)
	for i := range mat {
		mat[i] = make([]float64, n)
		fonte[i] = make([]bool, n)
	}

	// 1) valores conhecidos da tabela.
	for _, e := range tabelaAula13 {
		mat[e.i][e.j] = e.km
		mat[e.j][e.i] = e.km
		fonte[e.i][e.j] = true
		fonte[e.j][e.i] = true
	}

	// 2) fator de calibração = mediana de (tabela / Haversine) nos pares conhecidos.
	ratios := make([]float64, 0, len(tabelaAula13))
	for _, e := range tabelaAula13 {
		h := haversineKm(cidadesFixas[e.i].Lat, cidadesFixas[e.i].Lng, cidadesFixas[e.j].Lat, cidadesFixas[e.j].Lng)
		if h > 0 {
			ratios = append(ratios, e.km/h)
		}
	}
	sort.Float64s(ratios)
	fator := 1.2
	if len(ratios) > 0 {
		fator = ratios[len(ratios)/2]
	}

	// 3) preenche os pares faltantes (os "—" da tabela) com Haversine·fator.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if fonte[i][j] {
				continue
			}
			h := haversineKm(cidadesFixas[i].Lat, cidadesFixas[i].Lng, cidadesFixas[j].Lat, cidadesFixas[j].Lng)
			d := math.Round(h * fator)
			mat[i][j] = d
			mat[j][i] = d
		}
	}

	cs := make([]Cidade, n)
	copy(cs, cidadesFixas)
	return Mapa{Cidades: cs, Matriz: mat, Fonte: fonte, Fator: fator}
}

// haversineKm — distância de great-circle (km) entre dois pontos lat/lng.
func haversineKm(latA, lngA, latB, lngB float64) float64 {
	const R = 6371.0
	rad := func(d float64) float64 { return d * math.Pi / 180 }
	dLat := rad(latB - latA)
	dLng := rad(lngB - lngA)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(rad(latA))*math.Cos(rad(latB))*math.Sin(dLng/2)*math.Sin(dLng/2)
	return R * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

// CalcularDistanciaTour — soma das distâncias do ciclo fechado (volta à origem).
func CalcularDistanciaTour(tour []int, mat [][]float64) float64 {
	if len(tour) < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < len(tour); i++ {
		from := tour[i]
		to := tour[(i+1)%len(tour)]
		total += mat[from][to]
	}
	return total
}

// =============================================================================
// Cruzamentos de PERMUTAÇÃO — garantem filhos sem cidade repetida/faltando.
// =============================================================================

// CruzamentoOX — Order Crossover. Copia um segmento [a,b] do 1º pai e completa
// o resto na ORDEM em que as cidades aparecem no 2º pai (a partir de b+1, circular).
func CruzamentoOX(pa, pb []int, rng *rand.Rand) ([]int, []int) {
	n := len(pa)
	a, b := doisCortes(n, rng)
	return ox(pa, pb, a, b), ox(pb, pa, a, b)
}

func ox(p1, p2 []int, a, b int) []int {
	n := len(p1)
	filho := make([]int, n)
	usado := make([]bool, n)
	for i := range filho {
		filho[i] = -1
	}
	for i := a; i <= b; i++ {
		filho[i] = p1[i]
		usado[p1[i]] = true
	}
	pos := (b + 1) % n
	for k := 0; k < n; k++ {
		cidade := p2[(b+1+k)%n]
		if usado[cidade] {
			continue
		}
		filho[pos] = cidade
		usado[cidade] = true
		pos = (pos + 1) % n
	}
	return filho
}

// CruzamentoPMX — Partially Mapped Crossover. Copia um segmento [a,b] do 1º pai
// e resolve os conflitos do 2º pai seguindo a cadeia de mapeamento posicional.
func CruzamentoPMX(pa, pb []int, rng *rand.Rand) ([]int, []int) {
	n := len(pa)
	a, b := doisCortes(n, rng)
	return pmx(pa, pb, a, b), pmx(pb, pa, a, b)
}

func pmx(p1, p2 []int, a, b int) []int {
	n := len(p1)
	filho := make([]int, n)
	noSegmento := make([]bool, n)
	for i := range filho {
		filho[i] = -1
	}
	for i := a; i <= b; i++ {
		filho[i] = p1[i]
		noSegmento[p1[i]] = true
	}
	// posição de cada cidade em p2 (pra seguir a cadeia de deslocamento).
	posP2 := make([]int, n)
	for i, c := range p2 {
		posP2[c] = i
	}
	for i := a; i <= b; i++ {
		cidade := p2[i]
		if noSegmento[cidade] {
			continue // já entrou pelo segmento de p1
		}
		pos := i
		for {
			deslocada := p1[pos]  // cidade de p1 que ocupa 'pos' no filho
			j := posP2[deslocada] // onde 'deslocada' aparece em p2
			if filho[j] == -1 {
				filho[j] = cidade
				break
			}
			pos = j
		}
	}
	for i := 0; i < n; i++ {
		if filho[i] == -1 {
			filho[i] = p2[i]
		}
	}
	return filho
}

func doisCortes(n int, rng *rand.Rand) (int, int) {
	a := rng.Intn(n)
	b := rng.Intn(n)
	if a > b {
		a, b = b, a
	}
	return a, b
}

// =============================================================================
// Mutações de PERMUTAÇÃO
// =============================================================================

// MutacaoSwap — sorteia duas cidades distintas e troca de posição (Aula 13).
func MutacaoSwap(t []int, rng *rand.Rand) {
	n := len(t)
	if n < 2 {
		return
	}
	i := rng.Intn(n)
	j := rng.Intn(n)
	for j == i {
		j = rng.Intn(n)
	}
	t[i], t[j] = t[j], t[i]
}

// MutacaoInversao — reverte um segmento aleatório (corta dois "fios" do tour).
func MutacaoInversao(t []int, rng *rand.Rand) {
	n := len(t)
	if n < 2 {
		return
	}
	i, j := doisCortes(n, rng)
	for i < j {
		t[i], t[j] = t[j], t[i]
		i++
		j--
	}
}

// =============================================================================
// Treinar — orquestra o AG, emite Step por geração via canal.
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config) Result {
	cfg = sanitizar(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	mapa := ConstruirMapa()
	mat := mapa.Matriz
	nCidades := len(mapa.Cidades)

	pop := make([]Individuo, cfg.PopSize)
	for i := range pop {
		tour := rng.Perm(nCidades)
		pop[i] = Individuo{Tour: tour, Distancia: CalcularDistanciaTour(tour, mat)}
	}

	// As probabilidades de ranking dependem só de N e da pressão — constantes ao
	// longo das gerações; o que muda é QUEM ocupa cada rank (a pop é reordenada).
	var cumulRank []float64
	switch cfg.Selecao {
	case SelRankingLinear:
		cumulRank = cumulativo(ProbsRankingLinear(cfg.PopSize, cfg.EtaMax))
	case SelRankingExp:
		cumulRank = cumulativo(ProbsRankingExp(cfg.PopSize, cfg.CExp))
	}

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)
	histDiv := make([]int, 0, cfg.MaxGeracoes)
	melhorGlobal := Individuo{Distancia: math.Inf(1)}

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// ordena por distância asc → rank 1 (melhor) = pop[0].
		sort.Slice(pop, func(i, j int) bool { return pop[i].Distancia < pop[j].Distancia })

		melhorDaGen := pop[0]
		soma := 0.0
		popDist := make([]float64, len(pop))
		for i, ind := range pop {
			soma += ind.Distancia
			popDist[i] = ind.Distancia
		}
		media := soma / float64(len(pop))
		pior := pop[len(pop)-1].Distancia
		div := diversidade(pop)
		histMelhor = append(histMelhor, melhorDaGen.Distancia)
		histMedia = append(histMedia, media)
		histDiv = append(histDiv, div)

		if melhorDaGen.Distancia < melhorGlobal.Distancia {
			melhorGlobal = clonar(melhorDaGen)
		}

		if progressCh != nil {
			progressCh <- Step{
				Geracao:          g + 1,
				MelhorTour:       rotacionar(melhorDaGen.Tour, 0),
				MelhorDist:       melhorDaGen.Distancia,
				MediaDist:        media,
				PiorDist:         pior,
				Diversidade:      div,
				MelhorGlobalTour: rotacionar(melhorGlobal.Tour, 0),
				MelhorGlobalDist: melhorGlobal.Distancia,
				PopDist:          popDist,
			}
		}

		// ----- próxima geração -----
		elites := make([]Individuo, 0, cfg.Elitismo)
		for i := 0; i < cfg.Elitismo && i < len(pop); i++ {
			elites = append(elites, clonar(pop[i]))
		}

		// roleta proporcional precisa de cumulativo recalculado por geração.
		var cumulRoleta []float64
		if cfg.Selecao == SelRoleta {
			cumulRoleta = cumulativo(fitnessRoleta(pop))
		}

		precisamos := cfg.PopSize - len(elites)
		filhos := make([]Individuo, 0, precisamos)
		for len(filhos) < precisamos {
			paiA := selecionar(pop, cfg, rng, cumulRank, cumulRoleta)
			paiB := selecionar(pop, cfg, rng, cumulRank, cumulRoleta)

			var t1, t2 []int
			if rng.Float64() < cfg.ProbCruzamento {
				if cfg.Cruzamento == CrossPMX {
					t1, t2 = CruzamentoPMX(paiA.Tour, paiB.Tour, rng)
				} else {
					t1, t2 = CruzamentoOX(paiA.Tour, paiB.Tour, rng)
				}
			} else {
				t1 = append([]int(nil), paiA.Tour...)
				t2 = append([]int(nil), paiB.Tour...)
			}
			aplicarMutacao(t1, cfg, rng)
			aplicarMutacao(t2, cfg, rng)

			filhos = append(filhos, Individuo{Tour: t1, Distancia: CalcularDistanciaTour(t1, mat)})
			if len(filhos) < precisamos {
				filhos = append(filhos, Individuo{Tour: t2, Distancia: CalcularDistanciaTour(t2, mat)})
			}
		}

		pop = append(elites, filhos...)
	}

	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorTour:      rotacionar(melhorGlobal.Tour, 0),
		MelhorDist:      melhorGlobal.Distancia,
		HistMelhor:      histMelhor,
		HistMedia:       histMedia,
		HistDiversidade: histDiv,
		Cfg:             cfg,
	}
}

// =============================================================================
// Seleção e helpers
// =============================================================================

func selecionar(pop []Individuo, cfg Config, rng *rand.Rand, cumulRank, cumulRoleta []float64) Individuo {
	switch cfg.Selecao {
	case SelTorneio:
		return torneio(pop, cfg.TamanhoTorneio, rng)
	case SelRoleta:
		return clonar(pop[amostrar(cumulRoleta, rng)])
	default:
		// ranking linear/exp — pop ORDENADA asc, então rank k = pop[k].
		return clonar(pop[amostrar(cumulRank, rng)])
	}
}

func torneio(pop []Individuo, k int, rng *rand.Rand) Individuo {
	if k < 2 {
		k = 2
	}
	if k > len(pop) {
		k = len(pop)
	}
	melhor := pop[rng.Intn(len(pop))]
	for i := 1; i < k; i++ {
		c := pop[rng.Intn(len(pop))]
		if c.Distancia < melhor.Distancia {
			melhor = c
		}
	}
	return clonar(melhor)
}

// fitnessRoleta — converte distância (minimizar) em aptidão positiva
// (maxDist − dist + ε) pra seleção proporcional clássica.
func fitnessRoleta(pop []Individuo) []float64 {
	maxD := pop[0].Distancia
	for _, ind := range pop {
		if ind.Distancia > maxD {
			maxD = ind.Distancia
		}
	}
	out := make([]float64, len(pop))
	for i, ind := range pop {
		out[i] = (maxD - ind.Distancia) + 1e-9
	}
	return out
}

func cumulativo(w []float64) []float64 {
	c := make([]float64, len(w))
	s := 0.0
	for i, x := range w {
		s += x
		c[i] = s
	}
	return c
}

// amostrar — roleta sobre o vetor cumulativo; devolve o índice sorteado.
func amostrar(cumul []float64, rng *rand.Rand) int {
	if len(cumul) == 0 {
		return 0
	}
	r := rng.Float64() * cumul[len(cumul)-1]
	idx := 0
	for idx < len(cumul)-1 && cumul[idx] < r {
		idx++
	}
	return idx
}

func aplicarMutacao(t []int, cfg Config, rng *rand.Rand) {
	if rng.Float64() >= cfg.ProbMutacao {
		return
	}
	if cfg.Mutacao == MutInversao {
		MutacaoInversao(t, rng)
	} else {
		MutacaoSwap(t, rng)
	}
}

func clonar(src Individuo) Individuo {
	t := make([]int, len(src.Tour))
	copy(t, src.Tour)
	return Individuo{Tour: t, Distancia: src.Distancia}
}

// rotacionar — devolve o tour reescrito pra começar na cidade `inicio` (custo do
// ciclo é o mesmo; só muda a apresentação, partindo sempre de Uberaba).
func rotacionar(tour []int, inicio int) []int {
	n := len(tour)
	out := make([]int, 0, n)
	start := 0
	for i, c := range tour {
		if c == inicio {
			start = i
			break
		}
	}
	for i := 0; i < n; i++ {
		out = append(out, tour[(start+i)%n])
	}
	return out
}

// diversidade — nº de tours distintos (canonizando pela rotação que começa em 0).
func diversidade(pop []Individuo) int {
	seen := make(map[string]struct{}, len(pop))
	for _, ind := range pop {
		rot := rotacionar(ind.Tour, 0)
		b := make([]byte, 0, len(rot))
		for _, c := range rot {
			b = append(b, byte(c+1)) // +1 evita o byte 0 (corta string)
		}
		seen[string(b)] = struct{}{}
	}
	return len(seen)
}

func sanitizar(cfg Config) Config {
	if cfg.PopSize < 4 {
		cfg.PopSize = 4
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 250
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
	switch cfg.Selecao {
	case SelRankingLinear, SelRankingExp, SelTorneio, SelRoleta:
	default:
		cfg.Selecao = SelRankingLinear
	}
	if cfg.Cruzamento != CrossOX && cfg.Cruzamento != CrossPMX {
		cfg.Cruzamento = CrossOX
	}
	if cfg.Mutacao != MutSwap && cfg.Mutacao != MutInversao {
		cfg.Mutacao = MutSwap
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
	if cfg.EtaMax < 1.0 {
		cfg.EtaMax = 1.0
	}
	if cfg.EtaMax > 2.0 {
		cfg.EtaMax = 2.0
	}
	if cfg.CExp <= 1.0 {
		cfg.CExp = 1.0 + 1e-6
	}
	return cfg
}
