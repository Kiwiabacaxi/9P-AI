package matching

import (
	"math/rand"
	"sort"
	"time"
)

// initialPopulation gera N cromossomos com matching aleatório (incluindo -1).
func initialPopulation(s Scenario, popSize int, rng *rand.Rand) []Chromosome {
	M := len(s.Traders)
	N := len(s.Lots)
	pop := make([]Chromosome, popSize)
	for i := range pop {
		c := make(Chromosome, N)
		for k := range c {
			r := rng.Intn(M + 1)
			if r == M {
				c[k] = -1
			} else {
				c[k] = r
			}
		}
		pop[i] = c
	}
	return pop
}

// torneio retorna o melhor de k indivíduos sorteados.
func torneio(pop []Chromosome, fits []float64, k int, rng *rand.Rand) Chromosome {
	bestIdx := rng.Intn(len(pop))
	bestFit := fits[bestIdx]
	for i := 1; i < k; i++ {
		idx := rng.Intn(len(pop))
		if fits[idx] > bestFit {
			bestFit = fits[idx]
			bestIdx = idx
		}
	}
	return cloneChrom(pop[bestIdx])
}

func cloneChrom(c Chromosome) Chromosome {
	cp := make(Chromosome, len(c))
	copy(cp, c)
	return cp
}

// crossoverUniforme produz 1 filho com gene[i] do pai aleatoriamente.
func crossoverUniforme(p1, p2 Chromosome, rng *rand.Rand) Chromosome {
	N := len(p1)
	child := make(Chromosome, N)
	for i := 0; i < N; i++ {
		if rng.Intn(2) == 0 {
			child[i] = p1[i]
		} else {
			child[i] = p2[i]
		}
	}
	return child
}

// repair: pra cada trader em overcapacity, remove iterativamente o lote menos rentável até caber.
// "Menos rentável" = menor (preço_pago - preço_reserva) por saca.
func repair(s Scenario, c Chromosome) Chromosome {
	M := len(s.Traders)
	for {
		// soma volumes por trader
		volPorTrader := make([]float64, M)
		for i, j := range c {
			if j < 0 || j >= M {
				continue
			}
			volPorTrader[j] += s.Lots[i].VolumeT
		}
		// acha primeiro trader em overflow
		troubled := -1
		for j := 0; j < M; j++ {
			if volPorTrader[j] > s.Traders[j].CapacidadeT {
				troubled = j
				break
			}
		}
		if troubled == -1 {
			return c
		}
		// remove o lote menos rentável atribuído a este trader
		worstIdx := -1
		worstMargin := 0.0
		first := true
		for i, j := range c {
			if j != troubled {
				continue
			}
			margin := PrecoPago(s, i, j) - s.Lots[i].PrecoReserva
			if first || margin < worstMargin {
				worstMargin = margin
				worstIdx = i
				first = false
			}
		}
		if worstIdx == -1 {
			return c
		}
		c[worstIdx] = -1
	}
}

// mutar aplica mutação composta com probMutacao por gene.
func mutar(c Chromosome, M int, probMutacao float64, rng *rand.Rand) {
	N := len(c)
	for i := 0; i < N; i++ {
		if rng.Float64() >= probMutacao {
			continue
		}
		r := rng.Float64()
		switch {
		case r < 0.50:
			// swap com outro gene
			j := rng.Intn(N)
			c[i], c[j] = c[j], c[i]
		case r < 0.80:
			// reassign aleatório
			c[i] = rng.Intn(M)
		default:
			// force unmatch
			c[i] = -1
		}
	}
}

// elitismo: extrai os top-p indivíduos.
func elites(pop []Chromosome, fits []float64, p int) []Chromosome {
	if p <= 0 {
		return nil
	}
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(a, b int) bool {
		return fits[idxs[a]] > fits[idxs[b]]
	})
	if p > len(idxs) {
		p = len(idxs)
	}
	out := make([]Chromosome, p)
	for i := 0; i < p; i++ {
		out[i] = cloneChrom(pop[idxs[i]])
	}
	return out
}

// Treinar roda o GA. progressCh recebe Steps a cada geração; quando feito, retorna Result.
func Treinar(progressCh chan<- Step, s Scenario, cfg Config) Result {
	cfg = sanitizeCfg(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	M := len(s.Traders)
	pop := initialPopulation(s, cfg.PopSize, rng)
	// reparo inicial
	for i := range pop {
		pop[i] = repair(s, pop[i])
	}

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)

	var melhorGlobal Chromosome
	var melhorFitGlobal float64
	primeiro := true

	for gen := 0; gen < cfg.MaxGeracoes; gen++ {
		// avaliação
		fits := make([]float64, len(pop))
		breakdowns := make([]FitnessBreakdown, len(pop))
		for i, c := range pop {
			br := Evaluate(s, c, cfg)
			fits[i] = br.Fitness
			breakdowns[i] = br
		}
		// stats
		var bestIdx int
		bestFit := fits[0]
		var sum, worst float64 = 0, fits[0]
		for i, f := range fits {
			sum += f
			if f > bestFit {
				bestFit = f
				bestIdx = i
			}
			if f < worst {
				worst = f
			}
		}
		mean := sum / float64(len(fits))
		bestBr := breakdowns[bestIdx]

		if primeiro || bestFit > melhorFitGlobal {
			melhorFitGlobal = bestFit
			melhorGlobal = cloneChrom(pop[bestIdx])
			primeiro = false
		}

		histMelhor = append(histMelhor, bestFit)
		histMedia = append(histMedia, mean)

		if progressCh != nil {
			progressCh <- Step{
				Geracao:         gen,
				MelhorCrom:      cloneChrom(pop[bestIdx]),
				MelhorFitness:   bestFit,
				MediaFitness:    mean,
				PiorFitness:     worst,
				MelhorSuperavit: bestBr.SuperavitTotal,
				MelhorViolacoes: bestBr.Violacoes,
				TraderStats:     bestBr.TraderStats,
				NumMatched:      bestBr.NumMatched,
			}
		}

		// próxima geração
		newPop := elites(pop, fits, cfg.Elitismo)
		for len(newPop) < cfg.PopSize {
			p1 := torneio(pop, fits, cfg.TamanhoTorneio, rng)
			p2 := torneio(pop, fits, cfg.TamanhoTorneio, rng)
			var child Chromosome
			if rng.Float64() < cfg.ProbCruzamento {
				child = crossoverUniforme(p1, p2, rng)
			} else {
				child = cloneChrom(p1)
			}
			mutar(child, M, cfg.ProbMutacao, rng)
			child = repair(s, child)
			newPop = append(newPop, child)
		}
		pop = newPop
	}

	finalBr := Evaluate(s, melhorGlobal, cfg)
	return Result{
		Geracoes:        cfg.MaxGeracoes,
		MelhorCrom:      melhorGlobal,
		MelhorFitness:   melhorFitGlobal,
		MelhorViolacoes: finalBr.Violacoes,
		TraderStats:     finalBr.TraderStats,
		HistMelhor:      histMelhor,
		HistMedia:       histMedia,
		Cfg:             cfg,
		ScenarioID:      s.ID,
	}
}

func sanitizeCfg(cfg Config) Config {
	if cfg.PopSize <= 1 {
		cfg.PopSize = 80
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 200
	}
	if cfg.TamanhoTorneio <= 1 {
		cfg.TamanhoTorneio = 4
	}
	if cfg.Elitismo < 0 {
		cfg.Elitismo = 0
	}
	if cfg.MBig <= 0 {
		cfg.MBig = 1e6
	}
	return cfg
}
