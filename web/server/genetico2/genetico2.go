package genetico2

import (
	"math"
	"math/rand"
	"sort"
)

// =============================================================================
// Algoritmo Genético v2 — Aula 11
//
// Mantém a mesma função objetivo do aula 10 (f(x) = -|x · sin(√|x|)| em [0, 512])
// mas é totalmente PARAMETRIZÁVEL conforme pedido no slide:
//
//   • Seleção: roleta (clássica) OU torneio (k indivíduos, top 2)
//   • Cruzamento: 1 ponto OU 2 pontos
//   • Elitismo: p melhores são preservados de uma geração pra outra
//   • Tamanho da população, do cromossomo, Pc, Pm e gerações: livres
//
// Adições didáticas (não estão no slide, mas ajudam a entender):
//   • Métrica de diversidade (# de x's únicos na população) — útil pra
//     visualizar o "problema do super-indivíduo" que a aula 11 discute.
//
// O loop principal segue o fluxograma do slide (página 12):
//
//     Inicializa → Seleção → Cruzamento → Mutação → repete
//
// Cada geração emite um Step pelo canal pra animar no frontend.
// =============================================================================

const (
	// Métodos de seleção
	SelRoleta  = "roleta"
	SelTorneio = "torneio"

	dominioMin = 0.0
	dominioMax = 512.0
)

// Config — todos os hiperparâmetros do AG.
type Config struct {
	Bits           int     `json:"bits"`           // bits por cromossomo
	PopSize        int     `json:"popSize"`        // tamanho da população
	MaxGeracoes    int     `json:"maxGeracoes"`
	ProbCruzamento float64 `json:"probCruzamento"` // 0..1 — por casal
	ProbMutacao    float64 `json:"probMutacao"`    // 0..1 — por bit dos filhos
	Selecao        string  `json:"selecao"`        // "roleta" | "torneio"
	TamanhoTorneio int     `json:"tamanhoTorneio"` // k — só se Selecao = "torneio"
	PontosCorte    int     `json:"pontosCorte"`    // 1 ou 2
	Elitismo       int     `json:"elitismo"`       // p melhores preservados (0 = sem elitismo)
	Seed           int64   `json:"seed,omitempty"` // 0 = aleatório
}

func DefaultConfig() Config {
	return Config{
		Bits:           10,
		PopSize:        20,
		MaxGeracoes:    100,
		ProbCruzamento: 0.8,
		ProbMutacao:    0.05,
		Selecao:        SelTorneio,
		TamanhoTorneio: 3,
		PontosCorte:    2,
		Elitismo:       2,
	}
}

type Individuo struct {
	Bits    []int   `json:"bits"`
	Dec     int     `json:"dec"`
	X       float64 `json:"x"`
	Fitness float64 `json:"fitness"` // -f(x), positivo
	Fx      float64 `json:"fx"`      // f(x)
}

type Step struct {
	Geracao     int         `json:"geracao"`
	MelhorX     float64     `json:"melhorX"`
	MelhorFx    float64     `json:"melhorFx"`
	MediaFx     float64     `json:"mediaFx"`
	PiorFx      float64     `json:"piorFx"`
	Diversidade int         `json:"diversidade"` // # de x's únicos
	Populacao   []Individuo `json:"populacao"`
	MelhorIndiv Individuo   `json:"melhorIndiv"`
}

type Result struct {
	Geracoes        int       `json:"geracoes"`
	MelhorX         float64   `json:"melhorX"`
	MelhorFx        float64   `json:"melhorFx"`
	MelhorIndiv     Individuo `json:"melhorIndiv"`
	HistMelhorFx    []float64 `json:"histMelhorFx"`
	HistMediaFx     []float64 `json:"histMediaFx"`
	HistDiversidade []int     `json:"histDiversidade"`
	Bits            int       `json:"bits"`
	PopSize         int       `json:"popSize"`
	ProbCruzamento  float64   `json:"probCruzamento"`
	ProbMutacao     float64   `json:"probMutacao"`
	Selecao         string    `json:"selecao"`
	TamanhoTorneio  int       `json:"tamanhoTorneio"`
	PontosCorte     int       `json:"pontosCorte"`
	Elitismo        int       `json:"elitismo"`
}

// =============================================================================
// f(x) e conversões binário ↔ x
// =============================================================================

func F(x float64) float64 {
	return -math.Abs(x * math.Sin(math.Sqrt(math.Abs(x))))
}

func bitsParaDecimal(bits []int) int {
	dec := 0
	for _, b := range bits {
		dec = (dec << 1) | (b & 1)
	}
	return dec
}

func decimalParaX(dec, bits int) float64 {
	maxDec := (1 << bits) - 1
	if maxDec == 0 {
		return dominioMin
	}
	return float64(dec)/float64(maxDec)*(dominioMax-dominioMin) + dominioMin
}

func sincronizarBitsDecX(pop []Individuo, bits int) {
	for i := range pop {
		pop[i].Dec = bitsParaDecimal(pop[i].Bits)
		pop[i].X = decimalParaX(pop[i].Dec, bits)
	}
}

func gerarPopulacaoInicial(rng *rand.Rand, popSize, bits int) []Individuo {
	pop := make([]Individuo, popSize)
	for i := range pop {
		pop[i].Bits = make([]int, bits)
		for j := 0; j < bits; j++ {
			pop[i].Bits[j] = rng.Intn(2)
		}
	}
	sincronizarBitsDecX(pop, bits)
	return pop
}

// gerarImagem — calcula f(x) e fitness pra cada indivíduo.
func gerarImagem(pop []Individuo) {
	for i := range pop {
		pop[i].Fx = F(pop[i].X)
		pop[i].Fitness = -pop[i].Fx
	}
}

func gerarProbabilidades(pop []Individuo) []float64 {
	probs := make([]float64, len(pop))
	soma := 0.0
	for _, ind := range pop {
		soma += ind.Fitness
	}
	if soma == 0 {
		for i := range probs {
			probs[i] = 1.0 / float64(len(pop))
		}
		return probs
	}
	for i, ind := range pop {
		probs[i] = ind.Fitness / soma
	}
	return probs
}

func clonarIndividuo(src Individuo) Individuo {
	bits := make([]int, len(src.Bits))
	copy(bits, src.Bits)
	return Individuo{Bits: bits, Dec: src.Dec, X: src.X, Fitness: src.Fitness, Fx: src.Fx}
}

// =============================================================================
// Seleção: roleta (uma escolha) e torneio (duas escolhas — top 2 de k)
// =============================================================================

func selecionarPorRoleta(pop []Individuo, cumul []float64, rng *rand.Rand) Individuo {
	r := rng.Float64() * cumul[len(cumul)-1]
	idx := 0
	for idx < len(cumul)-1 && cumul[idx] < r {
		idx++
	}
	return clonarIndividuo(pop[idx])
}

// selecionarPorTorneio — sorteia k indivíduos (sem reposição) e devolve os 2 mais aptos.
//
// É o método recomendado pelo slide pra evitar o "problema do super-indivíduo":
// como cada torneio é um sorteio aleatório de k da população, indivíduos médios
// e fracos têm chance de serem pais (mantém diversidade), mas dentro de cada
// torneio os melhores ainda vencem (mantém pressão seletiva).
func selecionarPorTorneio(pop []Individuo, k int, rng *rand.Rand) (Individuo, Individuo) {
	if k > len(pop) {
		k = len(pop)
	}
	if k < 2 {
		k = 2
	}
	// k índices distintos via permutação parcial
	perm := rng.Perm(len(pop))[:k]
	// ordena por fitness desc
	sort.Slice(perm, func(i, j int) bool {
		return pop[perm[i]].Fitness > pop[perm[j]].Fitness
	})
	return clonarIndividuo(pop[perm[0]]), clonarIndividuo(pop[perm[1]])
}

// =============================================================================
// Cruzamento (1 ou 2 pontos — implementação genérica n-point)
//
// 1 ponto p:        bits[p..N-1]      são trocados (cauda).
// 2 pontos p1<p2:   bits[p1..p2-1]    são trocados (segmento central).
//
// Implementação: dado uma lista ordenada de pontos de corte, alternamos
// "trocar/não-trocar" a cada segmento — o primeiro segmento (head) NÃO troca.
// =============================================================================

func gerarPontosDeCorte(bits, n int, rng *rand.Rand) []int {
	if bits < 2 {
		return []int{1}
	}
	if n < 1 {
		n = 1
	}
	if n > bits-1 {
		n = bits - 1
	}
	// pontos válidos: 1, 2, ..., bits-1 (cortar antes da posição p)
	perm := rng.Perm(bits - 1)[:n]
	cuts := make([]int, n)
	for i, p := range perm {
		cuts[i] = p + 1
	}
	sort.Ints(cuts)
	return cuts
}

func cruzamentoComPontos(paiA, paiB Individuo, cuts []int) (Individuo, Individuo) {
	filhoA := clonarIndividuo(paiA)
	filhoB := clonarIndividuo(paiB)
	pontos := append([]int{}, cuts...)
	sort.Ints(pontos)
	pontos = append(pontos, len(paiA.Bits)) // sentinela final

	swap := false
	start := 0
	for _, end := range pontos {
		if swap {
			for k := start; k < end; k++ {
				filhoA.Bits[k] = paiB.Bits[k]
				filhoB.Bits[k] = paiA.Bits[k]
			}
		}
		swap = !swap
		start = end
	}
	return filhoA, filhoB
}

// =============================================================================
// Mutação — só nos filhos.
// =============================================================================

func mutarIndividuo(ind *Individuo, probMutacao float64, rng *rand.Rand) {
	for j := range ind.Bits {
		if rng.Float64() < probMutacao {
			ind.Bits[j] = 1 - ind.Bits[j]
		}
	}
}

// =============================================================================
// Elitismo: separa os p melhores indivíduos (cópia profunda).
// =============================================================================

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
		return pop[idxs[i]].Fitness > pop[idxs[j]].Fitness
	})
	elites := make([]Individuo, p)
	for i := 0; i < p; i++ {
		elites[i] = clonarIndividuo(pop[idxs[i]])
	}
	return elites
}

// diversidade — número de valores decimais únicos na população.
// Útil pra visualizar a "perda de diversidade genética" mencionada na aula 11.
func diversidade(pop []Individuo) int {
	seen := make(map[int]struct{}, len(pop))
	for _, ind := range pop {
		seen[ind.Dec] = struct{}{}
	}
	return len(seen)
}

// =============================================================================
// Treinar — orquestra o algoritmo e emite um Step por geração via canal.
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config) Result {
	cfg = sanitizarConfig(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	pop := gerarPopulacaoInicial(rng, cfg.PopSize, cfg.Bits)

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)
	histDiv := make([]int, 0, cfg.MaxGeracoes)

	var melhorGlobal Individuo
	melhorGlobal.Fitness = math.Inf(-1)

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// fitness
		gerarImagem(pop)

		// estatísticas + melhor da geração
		melhorIdx := 0
		soma := 0.0
		piorFx := math.Inf(-1)
		for i, ind := range pop {
			soma += ind.Fx
			if ind.Fitness > pop[melhorIdx].Fitness {
				melhorIdx = i
			}
			if ind.Fx > piorFx {
				piorFx = ind.Fx
			}
		}
		melhorFx := pop[melhorIdx].Fx
		mediaFx := soma / float64(len(pop))
		div := diversidade(pop)

		histMelhor = append(histMelhor, melhorFx)
		histMedia = append(histMedia, mediaFx)
		histDiv = append(histDiv, div)

		if pop[melhorIdx].Fitness > melhorGlobal.Fitness {
			melhorGlobal = clonarIndividuo(pop[melhorIdx])
		}

		// emit step
		if progressCh != nil {
			progressCh <- Step{
				Geracao:     g + 1,
				MelhorX:     pop[melhorIdx].X,
				MelhorFx:    melhorFx,
				MediaFx:     mediaFx,
				PiorFx:      piorFx,
				Diversidade: div,
				Populacao:   clonarPop(pop),
				MelhorIndiv: clonarIndividuo(pop[melhorIdx]),
			}
		}

		// === próxima geração ===

		// 1) elites preservadas (cópia profunda — não são mutadas)
		elites := extrairElites(pop, cfg.Elitismo)

		// 2) preparar cumul pra roleta (só se for o caso)
		var cumul []float64
		if cfg.Selecao == SelRoleta {
			probs := gerarProbabilidades(pop)
			cumul = make([]float64, len(probs))
			soma := 0.0
			for i, p := range probs {
				soma += p
				cumul[i] = soma
			}
		}

		// 3) gerar filhos suficientes pra completar PopSize
		precisamos := cfg.PopSize - len(elites)
		numCasais := (precisamos + 1) / 2
		filhos := make([]Individuo, 0, 2*numCasais)
		for c := 0; c < numCasais; c++ {
			var paiA, paiB Individuo
			if cfg.Selecao == SelTorneio {
				paiA, paiB = selecionarPorTorneio(pop, cfg.TamanhoTorneio, rng)
			} else {
				paiA = selecionarPorRoleta(pop, cumul, rng)
				paiB = selecionarPorRoleta(pop, cumul, rng)
			}

			var filhoA, filhoB Individuo
			if rng.Float64() < cfg.ProbCruzamento {
				cuts := gerarPontosDeCorte(cfg.Bits, cfg.PontosCorte, rng)
				filhoA, filhoB = cruzamentoComPontos(paiA, paiB, cuts)
			} else {
				filhoA = clonarIndividuo(paiA)
				filhoB = clonarIndividuo(paiB)
			}

			// mutação só nos filhos
			mutarIndividuo(&filhoA, cfg.ProbMutacao, rng)
			mutarIndividuo(&filhoB, cfg.ProbMutacao, rng)
			filhos = append(filhos, filhoA, filhoB)
		}
		// trunca pro tamanho exato
		if len(filhos) > precisamos {
			filhos = filhos[:precisamos]
		}

		// 4) nova população = elites + filhos
		pop = append(elites, filhos...)
		sincronizarBitsDecX(pop, cfg.Bits)
	}

	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorX:         melhorGlobal.X,
		MelhorFx:        melhorGlobal.Fx,
		MelhorIndiv:     melhorGlobal,
		HistMelhorFx:    histMelhor,
		HistMediaFx:     histMedia,
		HistDiversidade: histDiv,
		Bits:            cfg.Bits,
		PopSize:         cfg.PopSize,
		ProbCruzamento:  cfg.ProbCruzamento,
		ProbMutacao:     cfg.ProbMutacao,
		Selecao:         cfg.Selecao,
		TamanhoTorneio:  cfg.TamanhoTorneio,
		PontosCorte:     cfg.PontosCorte,
		Elitismo:        cfg.Elitismo,
	}
}

func sanitizarConfig(cfg Config) Config {
	if cfg.Bits <= 0 {
		cfg.Bits = 10
	}
	if cfg.PopSize <= 1 {
		cfg.PopSize = 20
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
	if cfg.TamanhoTorneio < 2 {
		cfg.TamanhoTorneio = 2
	}
	if cfg.TamanhoTorneio > cfg.PopSize {
		cfg.TamanhoTorneio = cfg.PopSize
	}
	if cfg.PontosCorte != 1 && cfg.PontosCorte != 2 {
		cfg.PontosCorte = 1
	}
	if cfg.Elitismo < 0 {
		cfg.Elitismo = 0
	}
	if cfg.Elitismo >= cfg.PopSize {
		cfg.Elitismo = cfg.PopSize - 2 // garante espaço pra pelo menos 1 casal de filhos
	}
	return cfg
}

func clonarPop(pop []Individuo) []Individuo {
	out := make([]Individuo, len(pop))
	for i := range pop {
		out[i] = clonarIndividuo(pop[i])
	}
	return out
}
