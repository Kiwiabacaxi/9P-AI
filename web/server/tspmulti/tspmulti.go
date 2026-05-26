package tspmulti

import (
	"math"
	"math/rand"
	"sort"
	"sync"

	"mlp-server/tsp"
)

// =============================================================================
// Algoritmo Genético Multi-populacional — Caixeiro Viajante (Trabalho 12 / Aula 14)
//
// Modelo de ILHAS: várias subpopulações ("ilhas") evoluem em paralelo
// (goroutines) usando exatamente o mesmo maquinário de GA do pacote tsp
// (seleção roleta/torneio, OX/PMX, mutação swap/inversão, elitismo). De tempos
// em tempos ocorre MIGRAÇÃO em anel — os melhores indivíduos de cada ilha são
// copiados pra ilha vizinha, substituindo os piores ("dança de cadeiras").
//
// O objetivo, conforme o slide, é manter DIVERSIDADE e escapar de mínimos
// locais onde uma população única empacaria. Por isso há um modo comparativo:
// roda em paralelo uma população única com o MESMO total de indivíduos e a
// curva dela contra a das ilhas evidencia o ganho da multipopulacional.
//
// Modelo de execução = "lockstep paralelo": o orquestrador comanda as gerações;
// a cada geração todas as ilhas evoluem 1 passo concorrentemente (WaitGroup) e
// a migração acontece em intervalos fixos. Determinístico com seed fixa — cada
// ilha tem seu próprio *rand.Rand, então o escalonamento das goroutines não
// afeta o resultado.
// =============================================================================

const TopologiaAnel = "anel"

// MultiConfig — hiperparâmetros do AG multi-populacional. Os parâmetros de GA
// em si (seleção, cruzamento, etc.) vêm de GA (tsp.Config), reaproveitados.
type MultiConfig struct {
	NumIlhas          int        `json:"numIlhas"`          // qtd de subpopulações
	TamIlha           int        `json:"tamIlha"`           // indivíduos por ilha
	MaxGeracoes       int        `json:"maxGeracoes"`       // gerações totais
	IntervaloMigracao int        `json:"intervaloMigracao"` // migra a cada N gerações
	NumMigrantes      int        `json:"numMigrantes"`      // melhores que migram por ilha
	Topologia         string     `json:"topologia"`         // "anel"
	CompararPopUnica  bool       `json:"compararPopUnica"`  // roda baseline de pop única
	Seed              int64      `json:"seed,omitempty"`
	GA                tsp.Config `json:"ga"`
}

// DefaultMultiConfig — defaults do exemplo do professor (3 ilhas × 20, migra a
// cada 10 gerações, 50 gerações, 1 migrante).
func DefaultMultiConfig() MultiConfig {
	ga := tsp.DefaultConfig()
	ga.MaxGeracoes = 50 // não usado diretamente (a orquestração controla), mas coerente
	return MultiConfig{
		NumIlhas:          3,
		TamIlha:           20,
		MaxGeracoes:       50,
		IntervaloMigracao: 10,
		NumMigrantes:      1,
		Topologia:         TopologiaAnel,
		CompararPopUnica:  true,
		GA:                ga,
	}
}

// IlhaStep — foto de uma ilha numa geração.
type IlhaStep struct {
	Ilha        int     `json:"ilha"`
	MelhorTour  []int   `json:"melhorTour"`
	MelhorDist  float64 `json:"melhorDist"`
	MelhorCusto float64 `json:"melhorCusto"`
	MediaDist   float64 `json:"mediaDist"`
	Diversidade int     `json:"diversidade"`
}

// Migracao — um movimento de migração (ilha De → ilha Para) numa geração.
type Migracao struct {
	De           int   `json:"de"`
	Para         int   `json:"para"`
	MigranteTour []int `json:"migranteTour,omitempty"` // melhor migrante (destaque do gene)
}

// MultiStep — payload por geração via SSE.
type MultiStep struct {
	Geracao            int        `json:"geracao"`
	Ilhas              []IlhaStep `json:"ilhas"`
	MelhorGlobalTour   []int      `json:"melhorGlobalTour"`
	MelhorGlobalDist   float64    `json:"melhorGlobalDist"`
	IlhaVencedora      int        `json:"ilhaVencedora"`
	GeracoesSemMelhora int        `json:"geracoesSemMelhora"`
	DiversidadeGlobal  int        `json:"diversidadeGlobal"`
	Migrou             bool       `json:"migrou"`
	Migracoes          []Migracao `json:"migracoes,omitempty"`
	RefUnicaDist       float64    `json:"refUnicaDist,omitempty"`
	RefUnicaDiv        int        `json:"refUnicaDiv,omitempty"` // diversidade da pop única (colapsa)
}

// MultiResult — resultado final.
type MultiResult struct {
	Geracoes           int         `json:"geracoes"`
	MelhorGlobalTour   []int       `json:"melhorGlobalTour"`
	MelhorGlobalDist   float64     `json:"melhorGlobalDist"`
	IlhaVencedora      int         `json:"ilhaVencedora"`
	HistGlobal         []float64   `json:"histGlobal"`
	HistIlhas          [][]float64 `json:"histIlhas"`
	HistDiversidade    []int       `json:"histDiversidade"`
	GeracoesMigracao   []int       `json:"geracoesMigracao"`
	HistRefUnica       []float64   `json:"histRefUnica,omitempty"`
	HistRefUnicaDiv    []int       `json:"histRefUnicaDiv,omitempty"`
	MelhorRefUnicaDist float64     `json:"melhorRefUnicaDist,omitempty"`
	Cfg                MultiConfig `json:"cfg"`
}

// Sanitizar — corrige uma MultiConfig pros limites válidos dado n cidades.
func Sanitizar(cfg MultiConfig, n int) MultiConfig {
	if cfg.NumIlhas < 2 {
		cfg.NumIlhas = 2
	}
	if cfg.TamIlha < 4 {
		cfg.TamIlha = 4
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 50
	}
	if cfg.IntervaloMigracao < 1 {
		cfg.IntervaloMigracao = 1
	}
	if cfg.IntervaloMigracao > cfg.MaxGeracoes {
		cfg.IntervaloMigracao = cfg.MaxGeracoes
	}
	if cfg.NumMigrantes < 1 {
		cfg.NumMigrantes = 1
	}
	if cfg.NumMigrantes > cfg.TamIlha-1 {
		cfg.NumMigrantes = cfg.TamIlha - 1
	}
	if cfg.Topologia != TopologiaAnel {
		cfg.Topologia = TopologiaAnel
	}
	// A config de GA por ilha usa PopSize = TamIlha.
	cfg.GA.PopSize = cfg.TamIlha
	cfg.GA = tsp.Sanitizar(cfg.GA, n)
	// tsp.Sanitizar pode ter ajustado PopSize (par) — reflete de volta no TamIlha.
	cfg.TamIlha = cfg.GA.PopSize
	return cfg
}

// melhorIdx — índice do indivíduo de menor custo.
func melhorIdx(pop []tsp.Individuo) int {
	best := 0
	for i, ind := range pop {
		if ind.Custo < pop[best].Custo {
			best = i
		}
	}
	return best
}

// statsIlha — melhor (menor custo), média de distância e diversidade.
func statsIlha(pop []tsp.Individuo) (melhor tsp.Individuo, mediaDist float64, div int) {
	bi := melhorIdx(pop)
	soma := 0.0
	for _, ind := range pop {
		soma += ind.Distancia
	}
	return pop[bi], soma / float64(len(pop)), tsp.Diversidade(pop)
}

// topMigrantes — devolve cópias dos `k` melhores (menor custo) de pop.
func topMigrantes(pop []tsp.Individuo, k int) []tsp.Individuo {
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(a, b int) bool {
		return pop[idxs[a]].Custo < pop[idxs[b]].Custo
	})
	if k > len(pop) {
		k = len(pop)
	}
	out := make([]tsp.Individuo, k)
	for i := 0; i < k; i++ {
		out[i] = clonar(pop[idxs[i]])
	}
	return out
}

// substituirPiores — substitui os len(migrantes) piores (maior custo) de pop
// pelos migrantes (já são cópias).
func substituirPiores(pop []tsp.Individuo, migrantes []tsp.Individuo) {
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	// piores primeiro (maior custo)
	sort.Slice(idxs, func(a, b int) bool {
		return pop[idxs[a]].Custo > pop[idxs[b]].Custo
	})
	for i, m := range migrantes {
		if i >= len(pop) {
			break
		}
		pop[idxs[i]] = m
	}
}

func clonar(src tsp.Individuo) tsp.Individuo {
	tour := make([]int, len(src.Tour))
	copy(tour, src.Tour)
	return tsp.Individuo{
		Tour:      tour,
		Distancia: src.Distancia,
		MaxLeg:    src.MaxLeg,
		TempoSec:  src.TempoSec,
		Custo:     src.Custo,
	}
}

// Treinar — orquestra o AG multi-populacional. Emite um MultiStep por geração
// no canal (se != nil) e devolve o MultiResult final.
func Treinar(progressCh chan<- MultiStep, cfg MultiConfig, matDist, matDur [][]float64) MultiResult {
	n := len(matDist)
	cfg = Sanitizar(cfg, n)

	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}

	// Uma config de GA por ilha (PopSize já ajustado em Sanitizar).
	gaIlha := cfg.GA

	// Semeia as ilhas — cada uma com seu próprio rng (determinismo independente
	// do escalonamento das goroutines).
	ilhas := make([][]tsp.Individuo, cfg.NumIlhas)
	rngs := make([]*rand.Rand, cfg.NumIlhas)
	for i := 0; i < cfg.NumIlhas; i++ {
		rngs[i] = rand.New(rand.NewSource(seed + int64(i) + 1))
		ilhas[i] = tsp.GerarPopulacaoInicial(rngs[i], cfg.TamIlha, n, matDist, matDur, gaIlha)
	}

	// Baseline de população única (testemunha "sem multipopulacional").
	var popUnica []tsp.Individuo
	var rngUnica *rand.Rand
	var gaUnica tsp.Config
	if cfg.CompararPopUnica {
		gaUnica = cfg.GA
		gaUnica.PopSize = cfg.NumIlhas * cfg.TamIlha
		gaUnica = tsp.Sanitizar(gaUnica, n)
		rngUnica = rand.New(rand.NewSource(seed))
		popUnica = tsp.GerarPopulacaoInicial(rngUnica, gaUnica.PopSize, n, matDist, matDur, gaUnica)
	}

	melhorGlobal := tsp.Individuo{Distancia: math.Inf(1), Custo: math.Inf(1)}
	ilhaVencedora := 0
	geracoesSemMelhora := 0
	refMelhor := math.Inf(1) // melhor da pop única (best-so-far)

	histGlobal := make([]float64, 0, cfg.MaxGeracoes)
	histIlhas := make([][]float64, cfg.NumIlhas)
	for i := range histIlhas {
		histIlhas[i] = make([]float64, 0, cfg.MaxGeracoes)
	}
	histDiv := make([]int, 0, cfg.MaxGeracoes)
	geracoesMigracao := make([]int, 0)
	histRef := make([]float64, 0, cfg.MaxGeracoes)
	histRefDiv := make([]int, 0, cfg.MaxGeracoes)

	for gen := 1; gen <= cfg.MaxGeracoes; gen++ {
		// 1) evolui todas as ilhas (+ pop única) concorrentemente.
		var wg sync.WaitGroup
		for i := 0; i < cfg.NumIlhas; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				ilhas[i] = tsp.EvoluirUmaGeracao(ilhas[i], gaIlha, rngs[i], matDist, matDur)
			}(i)
		}
		if cfg.CompararPopUnica {
			wg.Add(1)
			go func() {
				defer wg.Done()
				popUnica = tsp.EvoluirUmaGeracao(popUnica, gaUnica, rngUnica, matDist, matDur)
			}()
		}
		wg.Wait()

		// 2) migração em anel a cada IntervaloMigracao gerações.
		migrou := false
		var migracoes []Migracao
		if gen%cfg.IntervaloMigracao == 0 {
			migrou = true
			// Coleta os melhores de TODAS as ilhas ANTES de qualquer inserção,
			// pra migração ser simultânea ("dança de cadeiras"), não encadeada.
			bests := make([][]tsp.Individuo, cfg.NumIlhas)
			for i := 0; i < cfg.NumIlhas; i++ {
				bests[i] = topMigrantes(ilhas[i], cfg.NumMigrantes)
			}
			for i := 0; i < cfg.NumIlhas; i++ {
				dest := (i + 1) % cfg.NumIlhas
				substituirPiores(ilhas[dest], bests[i])
				migracoes = append(migracoes, Migracao{
					De:           i,
					Para:         dest,
					MigranteTour: tsp.RotateToStart(append([]int(nil), bests[i][0].Tour...), 0),
				})
			}
			geracoesMigracao = append(geracoesMigracao, gen)
		}

		// 3) estatísticas + melhor global.
		ilhaSteps := make([]IlhaStep, cfg.NumIlhas)
		melhorGenIdx := 0
		melhorGenDist := math.Inf(1)
		for i := 0; i < cfg.NumIlhas; i++ {
			melhor, mediaDist, div := statsIlha(ilhas[i])
			ilhaSteps[i] = IlhaStep{
				Ilha:        i,
				MelhorTour:  tsp.RotateToStart(append([]int(nil), melhor.Tour...), 0),
				MelhorDist:  melhor.Distancia,
				MelhorCusto: melhor.Custo,
				MediaDist:   mediaDist,
				Diversidade: div,
			}
			histIlhas[i] = append(histIlhas[i], melhor.Distancia)
			if melhor.Custo < melhorGenDist {
				melhorGenDist = melhor.Custo
				melhorGenIdx = i
			}
		}

		melhorDaGen := ilhas[melhorGenIdx][melhorIdx(ilhas[melhorGenIdx])]
		if melhorDaGen.Custo < melhorGlobal.Custo {
			melhorGlobal = clonar(melhorDaGen)
			ilhaVencedora = melhorGenIdx
			geracoesSemMelhora = 0
		} else {
			geracoesSemMelhora++
		}
		histGlobal = append(histGlobal, melhorGlobal.Distancia)

		// diversidade global — tours únicos somando todas as ilhas.
		var todos []tsp.Individuo
		for i := 0; i < cfg.NumIlhas; i++ {
			todos = append(todos, ilhas[i]...)
		}
		divGlobal := tsp.Diversidade(todos)
		histDiv = append(histDiv, divGlobal)

		// pop única (best-so-far) pra comparação justa + sua diversidade (que
		// colapsa, em contraste com as ilhas).
		refDist := 0.0
		refDiv := 0
		if cfg.CompararPopUnica {
			bu := popUnica[melhorIdx(popUnica)]
			if bu.Distancia < refMelhor {
				refMelhor = bu.Distancia
			}
			refDist = refMelhor
			refDiv = tsp.Diversidade(popUnica)
			histRef = append(histRef, refMelhor)
			histRefDiv = append(histRefDiv, refDiv)
		}

		// 4) emite o step.
		if progressCh != nil {
			progressCh <- MultiStep{
				Geracao:            gen,
				Ilhas:              ilhaSteps,
				MelhorGlobalTour:   tsp.RotateToStart(append([]int(nil), melhorGlobal.Tour...), 0),
				MelhorGlobalDist:   melhorGlobal.Distancia,
				IlhaVencedora:      ilhaVencedora,
				GeracoesSemMelhora: geracoesSemMelhora,
				DiversidadeGlobal:  divGlobal,
				Migrou:             migrou,
				Migracoes:          migracoes,
				RefUnicaDist:       refDist,
				RefUnicaDiv:        refDiv,
			}
		}
	}

	res := MultiResult{
		Geracoes:         cfg.MaxGeracoes,
		MelhorGlobalTour: tsp.RotateToStart(append([]int(nil), melhorGlobal.Tour...), 0),
		MelhorGlobalDist: melhorGlobal.Distancia,
		IlhaVencedora:    ilhaVencedora,
		HistGlobal:       histGlobal,
		HistIlhas:        histIlhas,
		HistDiversidade:  histDiv,
		GeracoesMigracao: geracoesMigracao,
		Cfg:              cfg,
	}
	if cfg.CompararPopUnica {
		res.HistRefUnica = histRef
		res.HistRefUnicaDiv = histRefDiv
		res.MelhorRefUnicaDist = refMelhor
	}
	return res
}
