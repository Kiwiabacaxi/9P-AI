package matching

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

// ObjVector representa os objetivos otimizados pelo NSGA-II.
// Todos são MAXIMIZE (Diversidade = 1 - HHI já invertido).
type ObjVector struct {
	Superavit   float64 `json:"superavit"`
	Inclusao    float64 `json:"inclusao"`    // fração de lotes matched (0..1)
	Diversidade float64 `json:"diversidade"` // 1 - HHI (0..1, 1=perfeitamente diverso)
	Violacoes   int     `json:"violacoes"`   // para constraint dominance
}

// EvaluateMulti avalia um cromossomo retornando os 3 objetivos + nº violações.
// HHI = soma de (volume_j / volume_total)² → 1 trader = 1.0, distribuído = baixo.
func EvaluateMulti(s Scenario, c Chromosome, cfg Config) (ObjVector, FitnessBreakdown) {
	br := Evaluate(s, c, cfg)
	N := len(s.Lots)
	inclusao := 0.0
	if N > 0 {
		inclusao = float64(br.NumMatched) / float64(N)
	}
	var volTotal float64
	for _, st := range br.TraderStats {
		volTotal += st.VolumeAlocadoT
	}
	hhi := 0.0
	if volTotal > 0 {
		for _, st := range br.TraderStats {
			share := st.VolumeAlocadoT / volTotal
			hhi += share * share
		}
	} else {
		hhi = 1.0 // tudo zero = degenerado, máxima concentração
	}
	return ObjVector{
		Superavit:   br.SuperavitTotal,
		Inclusao:    inclusao,
		Diversidade: 1.0 - hhi,
		Violacoes:   br.Violacoes,
	}, br
}

// dominates retorna true se a domina b sob constraint-domination de Deb.
// Regras:
//  1. Se a é feasible e b é infeasible, a domina.
//  2. Se ambos infeasible, o de menos violações domina.
//  3. Se ambos feasible, dominância usual em todos os objetivos
//     (a ≥ b em todos, a > b em pelo menos um).
func dominates(a, b ObjVector) bool {
	aFeas := a.Violacoes == 0
	bFeas := b.Violacoes == 0
	if aFeas && !bFeas {
		return true
	}
	if !aFeas && bFeas {
		return false
	}
	if !aFeas && !bFeas {
		return a.Violacoes < b.Violacoes
	}
	// ambos feasible
	betterIn := false
	if a.Superavit < b.Superavit || a.Inclusao < b.Inclusao || a.Diversidade < b.Diversidade {
		return false
	}
	if a.Superavit > b.Superavit || a.Inclusao > b.Inclusao || a.Diversidade > b.Diversidade {
		betterIn = true
	}
	return betterIn
}

// nonDominatedSort separa pop em fronteiras (rank 0, 1, 2...).
// Retorna slice de fronteiras (cada uma é slice de índices em pop).
func nonDominatedSort(objs []ObjVector) [][]int {
	n := len(objs)
	S := make([][]int, n)        // soluções dominadas por i
	np := make([]int, n)         // quantas dominam i
	var fronts [][]int

	first := []int{}
	for i := 0; i < n; i++ {
		S[i] = nil
		np[i] = 0
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if dominates(objs[i], objs[j]) {
				S[i] = append(S[i], j)
			} else if dominates(objs[j], objs[i]) {
				np[i]++
			}
		}
		if np[i] == 0 {
			first = append(first, i)
		}
	}
	fronts = append(fronts, first)

	k := 0
	for len(fronts[k]) > 0 {
		var next []int
		for _, i := range fronts[k] {
			for _, j := range S[i] {
				np[j]--
				if np[j] == 0 {
					next = append(next, j)
				}
			}
		}
		k++
		fronts = append(fronts, next)
	}
	// remove fronteira final vazia
	if len(fronts[len(fronts)-1]) == 0 {
		fronts = fronts[:len(fronts)-1]
	}
	return fronts
}

// crowdingDistance calcula distância de crowding pra cada índice na fronteira.
// Retorna map idx→distance; idx fora da fronteira não estão no map.
func crowdingDistance(front []int, objs []ObjVector) map[int]float64 {
	dist := make(map[int]float64, len(front))
	if len(front) == 0 {
		return dist
	}
	for _, i := range front {
		dist[i] = 0
	}
	if len(front) <= 2 {
		for _, i := range front {
			dist[i] = math.Inf(1)
		}
		return dist
	}

	// 3 objetivos — itera cada um
	type objAccessor func(o ObjVector) float64
	accessors := []objAccessor{
		func(o ObjVector) float64 { return o.Superavit },
		func(o ObjVector) float64 { return o.Inclusao },
		func(o ObjVector) float64 { return o.Diversidade },
	}
	for _, get := range accessors {
		sorted := make([]int, len(front))
		copy(sorted, front)
		sort.Slice(sorted, func(a, b int) bool {
			return get(objs[sorted[a]]) < get(objs[sorted[b]])
		})
		lo := get(objs[sorted[0]])
		hi := get(objs[sorted[len(sorted)-1]])
		span := hi - lo
		if span <= 1e-12 {
			continue // todos iguais; sem contribuição
		}
		dist[sorted[0]] = math.Inf(1)
		dist[sorted[len(sorted)-1]] = math.Inf(1)
		for k := 1; k < len(sorted)-1; k++ {
			prev := get(objs[sorted[k-1]])
			next := get(objs[sorted[k+1]])
			dist[sorted[k]] += (next - prev) / span
		}
	}
	return dist
}

// tournamentNSGA seleciona o melhor por (rank ASC, crowd DESC).
func tournamentNSGA(rng *rand.Rand, ranks []int, crowds []float64) int {
	a := rng.Intn(len(ranks))
	b := rng.Intn(len(ranks))
	if ranks[a] < ranks[b] {
		return a
	}
	if ranks[b] < ranks[a] {
		return b
	}
	if crowds[a] > crowds[b] {
		return a
	}
	return b
}

// StepNSGA é o snapshot por geração emitido via SSE.
type StepNSGA struct {
	Geracao        int           `json:"geracao"`
	FrontSize      int           `json:"frontSize"`     // tamanho da rank-0 atual
	Front          []FrontPoint  `json:"front"`         // pareto front (rank 0)
	BestSuperavit  float64       `json:"bestSuperavit"` // melhor superávit individual (mesmo se não feasible)
	BestInclusao   float64       `json:"bestInclusao"`
	BestDiversidade float64      `json:"bestDiversidade"`
	NumFeasible    int           `json:"numFeasible"`
}

// FrontPoint expõe um ponto da Pareto front pro frontend.
type FrontPoint struct {
	Chrom       Chromosome    `json:"chrom"`
	Superavit   float64       `json:"superavit"`
	Inclusao    float64       `json:"inclusao"`
	Diversidade float64       `json:"diversidade"`
	Violacoes   int           `json:"violacoes"`
	NumMatched  int           `json:"numMatched"`
	TraderStats []TraderStats `json:"traderStats"`
}

// ResultNSGA é o estado final.
type ResultNSGA struct {
	Geracoes   int          `json:"geracoes"`
	Front      []FrontPoint `json:"front"`
	Cfg        Config       `json:"cfg"`
	ScenarioID string       `json:"scenarioId"`
}

// TreinarNSGA roda NSGA-II. Streama Steps por geração; retorna ResultNSGA com a fronteira final.
func TreinarNSGA(progressCh chan<- StepNSGA, s Scenario, cfg Config) ResultNSGA {
	cfg = sanitizeCfg(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))
	M := len(s.Traders)

	pop := initialPopulation(s, cfg.PopSize, rng)
	for i := range pop {
		pop[i] = repair(s, pop[i])
	}

	// avaliação inicial
	objs := make([]ObjVector, len(pop))
	brks := make([]FitnessBreakdown, len(pop))
	for i, c := range pop {
		objs[i], brks[i] = EvaluateMulti(s, c, cfg)
	}

	for gen := 0; gen < cfg.MaxGeracoes; gen++ {
		// ranks + crowding
		fronts := nonDominatedSort(objs)
		ranks := make([]int, len(pop))
		for r, f := range fronts {
			for _, i := range f {
				ranks[i] = r
			}
		}
		crowds := make([]float64, len(pop))
		for _, f := range fronts {
			cd := crowdingDistance(f, objs)
			for i, d := range cd {
				crowds[i] = d
			}
		}

		// emit progress (front 0)
		if progressCh != nil {
			emitFrontStep(progressCh, gen, fronts[0], pop, objs, brks)
		}

		// próxima população: torneio + ops + repair
		offspring := make([]Chromosome, 0, cfg.PopSize)
		for len(offspring) < cfg.PopSize {
			ia := tournamentNSGA(rng, ranks, crowds)
			ib := tournamentNSGA(rng, ranks, crowds)
			var child Chromosome
			if rng.Float64() < cfg.ProbCruzamento {
				child = crossoverUniforme(pop[ia], pop[ib], rng)
			} else {
				child = cloneChrom(pop[ia])
			}
			mutar(child, M, cfg.ProbMutacao, rng)
			child = repair(s, child)
			offspring = append(offspring, child)
		}

		// avalia offspring
		offObjs := make([]ObjVector, len(offspring))
		offBrks := make([]FitnessBreakdown, len(offspring))
		for i, c := range offspring {
			offObjs[i], offBrks[i] = EvaluateMulti(s, c, cfg)
		}

		// combina + seleciona top N por (rank, crowd)
		combined := append([]Chromosome(nil), pop...)
		combined = append(combined, offspring...)
		combinedObjs := append([]ObjVector(nil), objs...)
		combinedObjs = append(combinedObjs, offObjs...)
		combinedBrks := append([]FitnessBreakdown(nil), brks...)
		combinedBrks = append(combinedBrks, offBrks...)

		combFronts := nonDominatedSort(combinedObjs)
		combRanks := make([]int, len(combined))
		for r, f := range combFronts {
			for _, i := range f {
				combRanks[i] = r
			}
		}
		combCrowds := make([]float64, len(combined))
		for _, f := range combFronts {
			cd := crowdingDistance(f, combinedObjs)
			for i, d := range cd {
				combCrowds[i] = d
			}
		}
		// ordena por (rank ASC, crowd DESC), pega top PopSize
		idxs := make([]int, len(combined))
		for i := range idxs {
			idxs[i] = i
		}
		sort.Slice(idxs, func(a, b int) bool {
			if combRanks[idxs[a]] != combRanks[idxs[b]] {
				return combRanks[idxs[a]] < combRanks[idxs[b]]
			}
			return combCrowds[idxs[a]] > combCrowds[idxs[b]]
		})
		newPop := make([]Chromosome, cfg.PopSize)
		newObjs := make([]ObjVector, cfg.PopSize)
		newBrks := make([]FitnessBreakdown, cfg.PopSize)
		for k := 0; k < cfg.PopSize; k++ {
			newPop[k] = combined[idxs[k]]
			newObjs[k] = combinedObjs[idxs[k]]
			newBrks[k] = combinedBrks[idxs[k]]
		}
		pop = newPop
		objs = newObjs
		brks = newBrks
	}

	// fronteira final = rank-0 da última pop
	fronts := nonDominatedSort(objs)
	finalFront := buildFrontPoints(fronts[0], pop, objs, brks)
	return ResultNSGA{
		Geracoes:   cfg.MaxGeracoes,
		Front:      finalFront,
		Cfg:        cfg,
		ScenarioID: s.ID,
	}
}

func emitFrontStep(ch chan<- StepNSGA, gen int, front []int, pop []Chromosome, objs []ObjVector, brks []FitnessBreakdown) {
	pts := buildFrontPoints(front, pop, objs, brks)
	var bestSup, bestInc, bestDiv float64
	var feas int
	first := true
	for _, o := range objs {
		if o.Violacoes == 0 {
			feas++
		}
		if first || o.Superavit > bestSup {
			bestSup = o.Superavit
		}
		if first || o.Inclusao > bestInc {
			bestInc = o.Inclusao
		}
		if first || o.Diversidade > bestDiv {
			bestDiv = o.Diversidade
		}
		first = false
	}
	ch <- StepNSGA{
		Geracao:         gen,
		FrontSize:       len(front),
		Front:           pts,
		BestSuperavit:   bestSup,
		BestInclusao:    bestInc,
		BestDiversidade: bestDiv,
		NumFeasible:     feas,
	}
}

func buildFrontPoints(front []int, pop []Chromosome, objs []ObjVector, brks []FitnessBreakdown) []FrontPoint {
	pts := make([]FrontPoint, 0, len(front))
	for _, i := range front {
		pts = append(pts, FrontPoint{
			Chrom:       cloneChrom(pop[i]),
			Superavit:   objs[i].Superavit,
			Inclusao:    objs[i].Inclusao,
			Diversidade: objs[i].Diversidade,
			Violacoes:   objs[i].Violacoes,
			NumMatched:  brks[i].NumMatched,
			TraderStats: brks[i].TraderStats,
		})
	}
	// ordena por superávit decrescente pro frontend ficar previsível
	sort.Slice(pts, func(a, b int) bool { return pts[a].Superavit > pts[b].Superavit })
	return pts
}
