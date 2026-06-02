package agrastrigin

import (
	"math"
	"math/rand"
	"sort"
)

// =============================================================================
// Algoritmo Genético com cromossomos REAIS — Aula 15 (Trabalho 13)
//
// Cromossomo = vetor de N números reais (1 gene por variável), em vez de bits.
// Operadores específicos:
//
//   • Inicialização:  x = a + c·(b−a),  c ∈ [0,1)
//   • Cruzamento RADCLIFF (gera 2 filhos):
//        xa(novo) = β·xa + (1−β)·xb
//        xb(novo) = (1−β)·xa + β·xb,   β ∈ (0,1) uniforme
//   • Cruzamento WRIGHT (gera 3, escolhe os 2 melhores):
//        xa(novo) = 0,5·xa + 0,5·xb
//        xb(novo) = 1,5·xa − 0,5·xb
//        xc(novo) = −0,5·xa + 1,5·xb
//     Os 2 melhores aptos (entre os 3, podando os fora de domínio) são mantidos.
//   • Mutação:  gene aleatório é substituído por novo x = a + c·(b−a)
//   • Seleção:  roleta ou torneio (idêntico ao GA binário)
//
// Função objetivo (problema do slide): Rastrigin 3D
//   f(x,y,z) = 30 + Σ (xᵢ² − 10·cos(2π·xᵢ))   com xᵢ ∈ [−5,12, +5,12]
//   Mínimo global: f(0,0,0) = 0; muitos mínimos locais (paisagem ondulada).
//
// Cada geração emite um Step pelo canal pra animar no frontend (scatter 3D
// dos indivíduos convergindo pro origem).
// =============================================================================

const (
	SelRoleta  = "roleta"
	SelTorneio = "torneio"

	CrossRadcliff = "radcliff"
	CrossWright   = "wright"

	NumVars            = 3 // (x, y, z) — fixo conforme enunciado
	dominioMinDefault  = -5.12
	dominioMaxDefault  = 5.12
)

// Config — hiperparâmetros do AG real.
type Config struct {
	PopSize        int     `json:"popSize"`
	MaxGeracoes    int     `json:"maxGeracoes"`
	ProbCruzamento float64 `json:"probCruzamento"`
	ProbMutacao    float64 `json:"probMutacao"`
	Selecao        string  `json:"selecao"`        // "roleta" | "torneio"
	TamanhoTorneio int     `json:"tamanhoTorneio"`
	Cruzamento     string  `json:"cruzamento"`     // "radcliff" | "wright"
	Elitismo       int     `json:"elitismo"`
	DominioMin     float64 `json:"dominioMin"`
	DominioMax     float64 `json:"dominioMax"`
	Seed           int64   `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		PopSize:        60,
		MaxGeracoes:    200,
		ProbCruzamento: 0.85,
		ProbMutacao:    0.05,
		Selecao:        SelTorneio,
		TamanhoTorneio: 4,
		Cruzamento:     CrossRadcliff,
		Elitismo:       2,
		DominioMin:     dominioMinDefault,
		DominioMax:     dominioMaxDefault,
	}
}

// Individuo — um ponto (x, y, z) no domínio, com sua aptidão.
type Individuo struct {
	X       []float64 `json:"x"`       // [x, y, z]
	Fx      float64   `json:"fx"`      // valor de Rastrigin(X) — quanto MENOR, melhor
	Fitness float64   `json:"fitness"` // pra seleção: (maxFx − Fx) + ε, sempre positivo
}

// Step — payload por geração via SSE.
type Step struct {
	Geracao     int         `json:"geracao"`
	MelhorFx    float64     `json:"melhorFx"`
	MelhorX     []float64   `json:"melhorX"`
	MediaFx     float64     `json:"mediaFx"`
	PiorFx      float64     `json:"piorFx"`
	Diversidade int         `json:"diversidade"`
	Populacao   []Individuo `json:"populacao"`
	MelhorGlobalFx float64  `json:"melhorGlobalFx"`
	MelhorGlobalX  []float64 `json:"melhorGlobalX"`
}

// Result — devolvido ao final do treino.
type Result struct {
	Geracoes        int       `json:"geracoes"`
	MelhorFx        float64   `json:"melhorFx"`
	MelhorX         []float64 `json:"melhorX"`
	HistMelhor      []float64 `json:"histMelhor"`
	HistMedia       []float64 `json:"histMedia"`
	HistDiversidade []int     `json:"histDiversidade"`
	Cfg             Config    `json:"cfg"`
}

// =============================================================================
// Função objetivo — Rastrigin 3D
// =============================================================================

// Rastrigin — f(x,y,z) = 30 + Σ(xᵢ² − 10·cos(2π·xᵢ)).
// Mínimo global: f(0,0,0) = 0.
func Rastrigin(x []float64) float64 {
	sum := 10.0 * float64(len(x))
	for _, xi := range x {
		sum += xi*xi - 10*math.Cos(2*math.Pi*xi)
	}
	return sum
}

// =============================================================================
// Inicialização: x = a + c·(b−a)
// =============================================================================

func gerarIndividuo(rng *rand.Rand, dMin, dMax float64) Individuo {
	x := make([]float64, NumVars)
	for i := range x {
		x[i] = dMin + rng.Float64()*(dMax-dMin)
	}
	fx := Rastrigin(x)
	return Individuo{X: x, Fx: fx}
}

func gerarPopulacaoInicial(rng *rand.Rand, n int, dMin, dMax float64) []Individuo {
	pop := make([]Individuo, n)
	for i := range pop {
		pop[i] = gerarIndividuo(rng, dMin, dMax)
	}
	return pop
}

// =============================================================================
// Seleção (igual ao GA clássico — não muda com codificação real)
// =============================================================================

// fitnessRoleta — converte Fx (minimizar) em fitness positivo (maximizar)
// usando (maxFx − Fx) + ε.
func fitnessRoleta(pop []Individuo) []float64 {
	maxFx := pop[0].Fx
	for _, ind := range pop {
		if ind.Fx > maxFx {
			maxFx = ind.Fx
		}
	}
	out := make([]float64, len(pop))
	for i, ind := range pop {
		out[i] = (maxFx - ind.Fx) + 1e-9
	}
	return out
}

func selecionarRoleta(pop []Individuo, cumul []float64, rng *rand.Rand) Individuo {
	r := rng.Float64() * cumul[len(cumul)-1]
	idx := 0
	for idx < len(cumul)-1 && cumul[idx] < r {
		idx++
	}
	return clonar(pop[idx])
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
		return pop[perm[i]].Fx < pop[perm[j]].Fx
	})
	return clonar(pop[perm[0]]), clonar(pop[perm[1]])
}

// =============================================================================
// Cruzamentos REAIS — RADCLIFF e WRIGHT
// =============================================================================

// CruzamentoRadcliff — gera 2 filhos como combinação convexa dos pais:
//   xa(novo) = β·xa + (1−β)·xb
//   xb(novo) = (1−β)·xa + β·xb
// Sempre dentro de [min(a,b), max(a,b)] gene-a-gene; nunca extrapola.
func CruzamentoRadcliff(pa, pb []float64, rng *rand.Rand) ([]float64, []float64) {
	beta := rng.Float64()
	n := len(pa)
	c1 := make([]float64, n)
	c2 := make([]float64, n)
	for i := 0; i < n; i++ {
		c1[i] = beta*pa[i] + (1-beta)*pb[i]
		c2[i] = (1-beta)*pa[i] + beta*pb[i]
	}
	return c1, c2
}

// CruzamentoWright — gera 3 filhos e devolve os 2 melhores DENTRO do domínio:
//   xa(novo) = 0,5·xa + 0,5·xb
//   xb(novo) = 1,5·xa − 0,5·xb     (extrapola, pode sair do domínio)
//   xc(novo) = −0,5·xa + 1,5·xb    (extrapola, pode sair do domínio)
// Se menos de 2 filhos forem válidos (dentro de [dMin,dMax]), completa com os
// "menos inválidos" (clamped). Sempre devolve 2 filhos.
func CruzamentoWright(pa, pb []float64, dMin, dMax float64, eval func([]float64) float64) ([]float64, []float64) {
	n := len(pa)
	c1 := make([]float64, n)
	c2 := make([]float64, n)
	c3 := make([]float64, n)
	for i := 0; i < n; i++ {
		c1[i] = 0.5*pa[i] + 0.5*pb[i]
		c2[i] = 1.5*pa[i] - 0.5*pb[i]
		c3[i] = -0.5*pa[i] + 1.5*pb[i]
	}
	cands := [][]float64{c1, c2, c3}
	// Score: válidos primeiro (menor Fx), depois inválidos clampados.
	type cand struct {
		x       []float64
		valido  bool
		score   float64
	}
	scored := make([]cand, 3)
	for i, c := range cands {
		valido := dentroDoDominio(c, dMin, dMax)
		x := c
		if !valido {
			x = clamp(c, dMin, dMax)
		}
		scored[i] = cand{x: x, valido: valido, score: eval(x)}
	}
	// ordena: válidos antes de inválidos; dentro de cada grupo, menor score primeiro
	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].valido != scored[j].valido {
			return scored[i].valido && !scored[j].valido
		}
		return scored[i].score < scored[j].score
	})
	return scored[0].x, scored[1].x
}

func dentroDoDominio(x []float64, dMin, dMax float64) bool {
	for _, xi := range x {
		if xi < dMin || xi > dMax {
			return false
		}
	}
	return true
}

func clamp(x []float64, dMin, dMax float64) []float64 {
	out := make([]float64, len(x))
	for i, xi := range x {
		if xi < dMin {
			out[i] = dMin
		} else if xi > dMax {
			out[i] = dMax
		} else {
			out[i] = xi
		}
	}
	return out
}

// =============================================================================
// Mutação — gene aleatório vira novo valor aleatório no domínio
// =============================================================================

func mutacao(x []float64, prob float64, dMin, dMax float64, rng *rand.Rand) {
	if rng.Float64() >= prob {
		return
	}
	i := rng.Intn(len(x))
	x[i] = dMin + rng.Float64()*(dMax-dMin)
}

// =============================================================================
// Helpers
// =============================================================================

func clonar(src Individuo) Individuo {
	x := make([]float64, len(src.X))
	copy(x, src.X)
	return Individuo{X: x, Fx: src.Fx, Fitness: src.Fitness}
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
		return pop[idxs[i]].Fx < pop[idxs[j]].Fx
	})
	elites := make([]Individuo, p)
	for i := 0; i < p; i++ {
		elites[i] = clonar(pop[idxs[i]])
	}
	return elites
}

// Diversidade — número de pontos "únicos" considerando arredondamento (1e-3).
// Sem quantizar, com floats todos são diferentes; quantizar dá visão da
// convergência (quantos pontos distintos a pop ainda ocupa).
func diversidade(pop []Individuo) int {
	seen := make(map[[NumVars]int64]struct{}, len(pop))
	for _, ind := range pop {
		var k [NumVars]int64
		for i := 0; i < NumVars; i++ {
			k[i] = int64(math.Round(ind.X[i] * 1000))
		}
		seen[k] = struct{}{}
	}
	return len(seen)
}

func melhorIdx(pop []Individuo) int {
	best := 0
	for i, ind := range pop {
		if ind.Fx < pop[best].Fx {
			best = i
		}
	}
	return best
}

// =============================================================================
// Treinar — orquestra o AG real, emite Step por geração via canal.
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config) Result {
	cfg = sanitizar(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	pop := gerarPopulacaoInicial(rng, cfg.PopSize, cfg.DominioMin, cfg.DominioMax)

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)
	histDiv := make([]int, 0, cfg.MaxGeracoes)

	melhorGlobal := Individuo{Fx: math.Inf(1)}

	avalEval := func(x []float64) float64 { return Rastrigin(x) }

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// estatísticas
		mi := melhorIdx(pop)
		soma := 0.0
		pior := pop[0].Fx
		for _, ind := range pop {
			soma += ind.Fx
			if ind.Fx > pior {
				pior = ind.Fx
			}
		}
		melhorDaGen := pop[mi]
		media := soma / float64(len(pop))
		div := diversidade(pop)
		histMelhor = append(histMelhor, melhorDaGen.Fx)
		histMedia = append(histMedia, media)
		histDiv = append(histDiv, div)

		if melhorDaGen.Fx < melhorGlobal.Fx {
			melhorGlobal = clonar(melhorDaGen)
		}

		if progressCh != nil {
			popSnap := make([]Individuo, len(pop))
			for i := range pop {
				popSnap[i] = clonar(pop[i])
			}
			progressCh <- Step{
				Geracao:        g + 1,
				MelhorFx:       melhorDaGen.Fx,
				MelhorX:        append([]float64(nil), melhorDaGen.X...),
				MediaFx:        media,
				PiorFx:         pior,
				Diversidade:    div,
				Populacao:      popSnap,
				MelhorGlobalFx: melhorGlobal.Fx,
				MelhorGlobalX:  append([]float64(nil), melhorGlobal.X...),
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

			var f1x, f2x []float64
			if rng.Float64() < cfg.ProbCruzamento {
				if cfg.Cruzamento == CrossWright {
					f1x, f2x = CruzamentoWright(paiA.X, paiB.X, cfg.DominioMin, cfg.DominioMax, avalEval)
				} else {
					f1x, f2x = CruzamentoRadcliff(paiA.X, paiB.X, rng)
				}
			} else {
				f1x = append([]float64(nil), paiA.X...)
				f2x = append([]float64(nil), paiB.X...)
			}

			mutacao(f1x, cfg.ProbMutacao, cfg.DominioMin, cfg.DominioMax, rng)
			mutacao(f2x, cfg.ProbMutacao, cfg.DominioMin, cfg.DominioMax, rng)

			// Garantia de domínio (RADCLIFF é convexo, mas mutação pode pôr no limite;
			// WRIGHT já clampa). Clampagem é segura.
			f1x = clamp(f1x, cfg.DominioMin, cfg.DominioMax)
			f2x = clamp(f2x, cfg.DominioMin, cfg.DominioMax)

			filhos = append(filhos,
				Individuo{X: f1x, Fx: Rastrigin(f1x)},
				Individuo{X: f2x, Fx: Rastrigin(f2x)},
			)
		}
		if len(filhos) > precisamos {
			filhos = filhos[:precisamos]
		}
		pop = append(elites, filhos...)
	}

	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorFx:        melhorGlobal.Fx,
		MelhorX:         melhorGlobal.X,
		HistMelhor:      histMelhor,
		HistMedia:       histMedia,
		HistDiversidade: histDiv,
		Cfg:             cfg,
	}
}

func sanitizar(cfg Config) Config {
	if cfg.PopSize < 4 {
		cfg.PopSize = 4
	}
	if cfg.PopSize%2 != 0 {
		cfg.PopSize++
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 200
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
	if cfg.Cruzamento != CrossRadcliff && cfg.Cruzamento != CrossWright {
		cfg.Cruzamento = CrossRadcliff
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
	if cfg.DominioMin >= cfg.DominioMax {
		cfg.DominioMin = dominioMinDefault
		cfg.DominioMax = dominioMaxDefault
	}
	return cfg
}
