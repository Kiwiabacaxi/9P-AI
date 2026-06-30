package rnaga

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
)

// =============================================================================
// Trabalho 15 — AG que descobre a melhor ARQUITETURA de uma MLP (RNA + AG).
//
// Cromossomo = vetor de 6 genes (valores inteiros e reais). O AG roda o padrão
// tradicional (geração inicial → avaliação → seleção → cruzamento → mutação →
// substituição). A avaliação de cada indivíduo TREINA a MLP definida pelo
// cromossomo e usa o MSE como fitness (menor = melhor). Limite de 100 gerações.
// =============================================================================

const numGenes = 6

// Cromossomo — vetor de 6 genes. Guarda os valores reais/inteiros direto
// (o "vetor String" do enunciado vira o método String() pra exibição).
type Cromossomo struct {
	Genes [6]float64 `json:"genes"`
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Decodificadores (conversões "no momento de empregar", como o enunciado pede).
func (c Cromossomo) Neuronios() int     { return clampInt(int(math.Round(c.Genes[0])), 2, 15) }
func (c Cromossomo) Camadas() int       { return clampInt(int(math.Round(c.Genes[1])), 2, 5) }
func (c Cromossomo) TaxaAprend() float64 { return clampF(c.Genes[2], 1e-5, 0.1) }
func (c Cromossomo) MaxEpocas() int     { return clampInt(int(math.Round(c.Genes[3])), 20, 1000) }
func (c Cromossomo) Online() bool       { return int(math.Round(c.Genes[4])) == 1 }
func (c Cromossomo) Normaliza() bool     { return int(math.Round(c.Genes[5])) == 1 }

func (c Cromossomo) epocasEfetivas(teto int) int {
	e := c.MaxEpocas()
	if teto > 0 && e > teto {
		e = teto
	}
	return e
}

func (c Cromossomo) String() string {
	on := "offline"
	if c.Online() {
		on = "online"
	}
	nm := "sem-norm"
	if c.Normaliza() {
		nm = "normaliza"
	}
	return fmt.Sprintf("%d | %d | %.5f | %d | %s | %s",
		c.Neuronios(), c.Camadas(), c.TaxaAprend(), c.MaxEpocas(), on, nm)
}

// NovoValorGene — sorteia um valor VÁLIDO para a posição dada (usado na criação
// inicial e na mutação).
func NovoValorGene(pos int, rng *rand.Rand) float64 {
	switch pos {
	case 0:
		return float64(2 + rng.Intn(14)) // neurônios 2..15
	case 1:
		return float64(2 + rng.Intn(4)) // camadas 2..5
	case 2:
		return 1e-5 + rng.Float64()*(0.1-1e-5) // taxa 1e-5..0.1
	case 3:
		return float64(20 + rng.Intn(981)) // épocas 20..1000
	case 4:
		return float64(1 + rng.Intn(2)) // online/offline
	case 5:
		return float64(1 + rng.Intn(2)) // normaliza/não
	}
	return 0
}

func CromossomoAleatorio(rng *rand.Rand) Cromossomo {
	var c Cromossomo
	for i := 0; i < numGenes; i++ {
		c.Genes[i] = NovoValorGene(i, rng)
	}
	return c
}

// Mutacao — com probabilidade prob: (a) sorteia uma posição, (b) gera um novo
// valor válido pra ela (enunciado: mutação de 5% em duas etapas).
func Mutacao(c *Cromossomo, prob float64, rng *rand.Rand) {
	if rng.Float64() >= prob {
		return
	}
	pos := rng.Intn(numGenes)
	c.Genes[pos] = NovoValorGene(pos, rng)
}

// CruzamentoUmPonto — ponto de corte único; dois filhos complementares.
func CruzamentoUmPonto(pa, pb Cromossomo, rng *rand.Rand) (Cromossomo, Cromossomo) {
	corte := 1 + rng.Intn(numGenes-1) // 1..5
	var fa, fb Cromossomo
	for k := 0; k < numGenes; k++ {
		if k < corte {
			fa.Genes[k] = pa.Genes[k]
			fb.Genes[k] = pb.Genes[k]
		} else {
			fa.Genes[k] = pb.Genes[k]
			fb.Genes[k] = pa.Genes[k]
		}
	}
	return fa, fb
}

// =============================================================================
// Dataset fictício — 100 padrões, 15 entradas (3..1457), 13 saídas (58..312).
// =============================================================================

type Dataset struct {
	X [][]float64 `json:"x"`
	Y [][]float64 `json:"y"`
}

func GerarDataset(rng *rand.Rand) Dataset {
	const P = 100
	ds := Dataset{X: make([][]float64, P), Y: make([][]float64, P)}
	for i := 0; i < P; i++ {
		ds.X[i] = make([]float64, numEntradas)
		for j := range ds.X[i] {
			ds.X[i][j] = entradaMin + rng.Float64()*(entradaMax-entradaMin)
		}
		ds.Y[i] = make([]float64, numSaidas)
		for k := range ds.Y[i] {
			ds.Y[i][k] = saidaMin + rng.Float64()*(saidaMax-saidaMin)
		}
	}
	return ds
}

// =============================================================================
// AG
// =============================================================================

type Config struct {
	PopSize     int     `json:"popSize"`
	MaxGeracoes int     `json:"maxGeracoes"`
	ProbMutacao float64 `json:"probMutacao"`
	TetoEpocas  int     `json:"tetoEpocas"` // teto p/ MaxEpocas do cromossomo (demo × completo)
	Seed        int64   `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		PopSize:     40,
		MaxGeracoes: 100,
		ProbMutacao: 0.05,
		TetoEpocas:  300,
	}
}

// IndividuoView — payload amigável de um indivíduo (genes + decodificação + MSE).
type IndividuoView struct {
	Genes     [6]float64 `json:"genes"`
	String    string     `json:"string"`
	MSE       float64    `json:"mse"`
	Neuronios int        `json:"neuronios"`
	Camadas   int        `json:"camadas"`
	Online    bool       `json:"online"`
	Normaliza bool       `json:"normaliza"`
}

func viewDe(c Cromossomo, mse float64) IndividuoView {
	return IndividuoView{
		Genes: c.Genes, String: c.String(), MSE: mse,
		Neuronios: c.Neuronios(), Camadas: c.Camadas(),
		Online: c.Online(), Normaliza: c.Normaliza(),
	}
}

type Step struct {
	Geracao          int             `json:"geracao"`
	MelhorMSE        float64         `json:"melhorMse"`
	MelhorGlobalMSE  float64         `json:"melhorGlobalMse"`
	MediaMSE         float64         `json:"mediaMse"`
	MelhorCromossomo IndividuoView   `json:"melhorCromossomo"`
	Populacao        []IndividuoView `json:"populacao"`
	// GradeMSE[neurônios−2][camadas−2] = melhor MSE já visto na célula (−1 = não visitada).
	GradeMSE [][]float64 `json:"gradeMse"`
}

type Result struct {
	Geracoes         int           `json:"geracoes"`
	MelhorCromossomo Cromossomo    `json:"melhorCromossomo"`
	MelhorView       IndividuoView `json:"melhorView"`
	MelhorMSE        float64       `json:"melhorMse"`
	HistMelhor       []float64     `json:"histMelhor"`
	HistMedia        []float64     `json:"histMedia"`
	Cfg              Config        `json:"cfg"`
}

func Treinar(progressCh chan<- Step, cfg Config) Result {
	cfg = sanitizar(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))
	ds := GerarDataset(rng)

	// memoização de fitness por arquitetura efetiva
	memo := make(map[string]float64)
	var memoMu sync.Mutex
	avalia := func(c Cromossomo) float64 {
		chave := chaveCanonica(c, cfg.TetoEpocas)
		memoMu.Lock()
		if v, ok := memo[chave]; ok {
			memoMu.Unlock()
			return v
		}
		memoMu.Unlock()
		v := AvaliarMSE(c, ds, cfg.TetoEpocas, seed)
		memoMu.Lock()
		memo[chave] = v
		memoMu.Unlock()
		return v
	}

	// avalia uma população inteira em paralelo (1 worker por CPU)
	avaliarPop := func(pop []Cromossomo) []float64 {
		mses := make([]float64, len(pop))
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i := range pop {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				sem <- struct{}{}
				mses[i] = avalia(pop[i])
				<-sem
			}(i)
		}
		wg.Wait()
		return mses
	}

	// grade de MSE por (neurônios 2..15 × camadas 2..5)
	grade := make([][]float64, 14)
	for i := range grade {
		grade[i] = make([]float64, 4)
		for j := range grade[i] {
			grade[i][j] = -1
		}
	}

	pop := make([]Cromossomo, cfg.PopSize)
	for i := range pop {
		pop[i] = CromossomoAleatorio(rng)
	}

	var histMelhor, histMedia []float64
	melhorGlobal := Cromossomo{}
	melhorGlobalMSE := math.Inf(1)

	for g := 0; g < cfg.MaxGeracoes; g++ {
		mses := avaliarPop(pop)

		soma, melhorIdx := 0.0, 0
		for i, m := range mses {
			soma += m
			if m < mses[melhorIdx] {
				melhorIdx = i
			}
			gi, gj := pop[i].Neuronios()-2, pop[i].Camadas()-2
			if grade[gi][gj] < 0 || m < grade[gi][gj] {
				grade[gi][gj] = m
			}
		}
		media := soma / float64(len(pop))
		melhorMSE := mses[melhorIdx]
		if melhorMSE < melhorGlobalMSE {
			melhorGlobalMSE = melhorMSE
			melhorGlobal = pop[melhorIdx]
		}
		histMelhor = append(histMelhor, melhorMSE)
		histMedia = append(histMedia, media)

		if progressCh != nil {
			popView := make([]IndividuoView, len(pop))
			for i := range pop {
				popView[i] = viewDe(pop[i], mses[i])
			}
			gradeCopy := make([][]float64, len(grade))
			for i := range grade {
				gradeCopy[i] = append([]float64(nil), grade[i]...)
			}
			progressCh <- Step{
				Geracao:          g + 1,
				MelhorMSE:        melhorMSE,
				MelhorGlobalMSE:  melhorGlobalMSE,
				MediaMSE:         media,
				MelhorCromossomo: viewDe(melhorGlobal, melhorGlobalMSE),
				Populacao:        popView,
				GradeMSE:         gradeCopy,
			}
		}

		pop = proximaGeracao(pop, mses, cfg, rng)
	}

	return Result{
		Geracoes:         cfg.MaxGeracoes,
		MelhorCromossomo: melhorGlobal,
		MelhorView:       viewDe(melhorGlobal, melhorGlobalMSE),
		MelhorMSE:        melhorGlobalMSE,
		HistMelhor:       histMelhor,
		HistMedia:        histMedia,
		Cfg:              cfg,
	}
}

// proximaGeracao — substituição ELITISTA: a melhor metade sobrevive; a outra
// metade vira filhos de pais escolhidos por ROLETA (menor MSE → maior chance),
// com cruzamento de 1 ponto e mutação de 5%.
func proximaGeracao(pop []Cromossomo, mses []float64, cfg Config, rng *rand.Rand) []Cromossomo {
	n := len(pop)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return mses[idx[a]] < mses[idx[b]] })

	metade := n / 2
	nova := make([]Cromossomo, 0, n)
	for i := 0; i < metade; i++ {
		nova = append(nova, pop[idx[i]])
	}

	// fitness p/ roleta: 1/(MSE+ε)
	cumul := make([]float64, n)
	s := 0.0
	for i, m := range mses {
		s += 1.0 / (m + 1e-9)
		cumul[i] = s
	}
	selecionar := func() Cromossomo {
		r := rng.Float64() * cumul[n-1]
		k := 0
		for k < n-1 && cumul[k] < r {
			k++
		}
		return pop[k]
	}

	for len(nova) < n {
		pa, pb := selecionar(), selecionar()
		fa, fb := CruzamentoUmPonto(pa, pb, rng)
		Mutacao(&fa, cfg.ProbMutacao, rng)
		Mutacao(&fb, cfg.ProbMutacao, rng)
		nova = append(nova, fa)
		if len(nova) < n {
			nova = append(nova, fb)
		}
	}
	return nova
}

func sanitizar(cfg Config) Config {
	if cfg.PopSize < 4 {
		cfg.PopSize = 4
	}
	if cfg.PopSize%2 != 0 {
		cfg.PopSize++
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 100
	}
	if cfg.MaxGeracoes > 1000 {
		cfg.MaxGeracoes = 1000
	}
	if cfg.ProbMutacao < 0 {
		cfg.ProbMutacao = 0
	}
	if cfg.ProbMutacao > 1 {
		cfg.ProbMutacao = 1
	}
	if cfg.TetoEpocas < 10 {
		cfg.TetoEpocas = 10
	}
	if cfg.TetoEpocas > 1000 {
		cfg.TetoEpocas = 1000
	}
	return cfg
}
