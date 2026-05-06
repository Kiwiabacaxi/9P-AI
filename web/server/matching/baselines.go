package matching

import "sort"

// GreedyByReserve atribui lotes em ordem de preço_reserva DESC ao trader que paga mais
// e ainda tem capacidade.
func GreedyByReserve(s Scenario) (Chromosome, FitnessBreakdown) {
	N := len(s.Lots)
	M := len(s.Traders)
	c := make(Chromosome, N)
	for i := range c {
		c[i] = -1
	}

	// indices ordenados por preço_reserva DESC
	idxs := make([]int, N)
	for i := range idxs {
		idxs[i] = i
	}
	sort.SliceStable(idxs, func(a, b int) bool {
		return s.Lots[idxs[a]].PrecoReserva > s.Lots[idxs[b]].PrecoReserva
	})

	usadoT := make([]float64, M)
	for _, i := range idxs {
		bestJ := -1
		bestMargin := 0.0
		first := true
		for j := 0; j < M; j++ {
			if usadoT[j]+s.Lots[i].VolumeT > s.Traders[j].CapacidadeT {
				continue
			}
			margin := PrecoPago(s, i, j) - s.Lots[i].PrecoReserva
			if first || margin > bestMargin {
				bestMargin = margin
				bestJ = j
				first = false
			}
		}
		if bestJ != -1 {
			c[i] = bestJ
			usadoT[bestJ] += s.Lots[i].VolumeT
		}
	}

	br := Evaluate(s, c, DefaultConfig())
	return c, br
}
