package matching

import "math"

const (
	SacasPorTonelada = 1000.0 / 60.0 // 16.667
)

// HaversineKm calcula distância em km entre dois pontos lat/lng.
func HaversineKm(latA, lngA, latB, lngB float64) float64 {
	const R = 6371.0
	rad := math.Pi / 180.0
	dLat := (latB - latA) * rad
	dLng := (lngB - lngA) * rad
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(latA*rad)*math.Cos(latB*rad)*math.Sin(dLng/2)*math.Sin(dLng/2)
	return 2 * R * math.Asin(math.Sqrt(a))
}

// premioProteina: +R$/saca por ponto de proteína acima de 36, capado em 4.
func premioProteina(proteina float64) float64 {
	delta := proteina - 36.0
	if delta < 0 {
		return 0
	}
	if delta > 4 {
		delta = 4
	}
	return delta * 1.0
}

// descontoLog: R$/saca = 0.05 * dist_km (≈ R$3/saca em 60km, escala ok pra demo).
func descontoLog(distKm float64) float64 {
	return 0.05 * distKm
}

// PrecoPago calcula o preço por saca que trader j pagaria pelo lote i.
// Capado em PrecoMaximo do trader.
func PrecoPago(s Scenario, lotIdx, traderIdx int) float64 {
	lot := s.Lots[lotIdx]
	trader := s.Traders[traderIdx]
	prod := findProducer(s.Producers, lot.ProducerID)
	dist := HaversineKm(prod.Lat, prod.Lng, trader.HubLat, trader.HubLng)
	preco := s.PrecoBase + premioProteina(lot.Proteina) - descontoLog(dist)
	if preco > trader.PrecoMaximo {
		preco = trader.PrecoMaximo
	}
	return preco
}

func findProducer(prods []Producer, id int) Producer {
	for _, p := range prods {
		if p.ID == id {
			return p
		}
	}
	return Producer{}
}

// EvaluateBreakdown avalia um cromossomo e retorna métricas detalhadas.
type FitnessBreakdown struct {
	Fitness        float64
	SuperavitTotal float64
	CustoLogTotal  float64
	PenalidadeQual float64
	Violacoes      int
	NumMatched     int
	TraderStats    []TraderStats
}

func Evaluate(s Scenario, c Chromosome, cfg Config) FitnessBreakdown {
	M := len(s.Traders)
	N := len(s.Lots)
	stats := make([]TraderStats, M)
	for j := range stats {
		stats[j].TraderID = s.Traders[j].ID
	}

	// blend acumuladores
	weightedProt := make([]float64, M)
	weightedUmid := make([]float64, M)
	weightedImp := make([]float64, M)

	var superavit, custoLog float64
	var matched int

	for i := 0; i < N; i++ {
		j := c[i]
		if j < 0 || j >= M {
			continue
		}
		matched++
		lot := s.Lots[i]
		trader := s.Traders[j]
		prod := findProducer(s.Producers, lot.ProducerID)
		dist := HaversineKm(prod.Lat, prod.Lng, trader.HubLat, trader.HubLng)
		preco := PrecoPago(s, i, j)
		sacas := lot.VolumeT * SacasPorTonelada
		// superávit: (preco - reserva) * sacas; pode ser negativo se reserva > preço
		superavit += (preco - lot.PrecoReserva) * sacas
		// custo logístico: dist * volume (custo de transporte; quanto maior, pior)
		custoLog += dist * lot.VolumeT
		stats[j].VolumeAlocadoT += lot.VolumeT
		stats[j].NumLotes++
		weightedProt[j] += lot.Proteina * lot.VolumeT
		weightedUmid[j] += lot.Umidade * lot.VolumeT
		weightedImp[j] += lot.Impurezas * lot.VolumeT
	}

	// finaliza blends
	for j := range stats {
		v := stats[j].VolumeAlocadoT
		if v > 0 {
			stats[j].BlendProteina = weightedProt[j] / v
			stats[j].BlendUmidade = weightedUmid[j] / v
			stats[j].BlendImpurezas = weightedImp[j] / v
		}
	}

	// violações hard + penalidade qual
	var penalQual float64
	var violacoes int
	for j, t := range s.Traders {
		st := &stats[j]
		// capacidade
		if st.VolumeAlocadoT > t.CapacidadeT {
			st.OverCapacity = true
			violacoes++
		}
		if st.NumLotes == 0 {
			continue
		}
		// blend mínimo de proteína (penalidade quadrática)
		if st.BlendProteina < t.ProteinaMin {
			st.UnderSpec = true
			violacoes++
			d := t.ProteinaMin - st.BlendProteina
			penalQual += d * d * st.VolumeAlocadoT
		}
		// blend max umidade
		if st.BlendUmidade > t.UmidadeMax {
			st.UnderSpec = true
			violacoes++
			d := st.BlendUmidade - t.UmidadeMax
			penalQual += d * d * st.VolumeAlocadoT
		}
		// impurezas
		if st.BlendImpurezas > t.ImpurezasMax {
			st.UnderSpec = true
			violacoes++
			d := st.BlendImpurezas - t.ImpurezasMax
			penalQual += d * d * st.VolumeAlocadoT
		}
	}

	fit := superavit - cfg.LambdaLog*custoLog - cfg.LambdaQual*penalQual - cfg.MBig*float64(violacoes)
	return FitnessBreakdown{
		Fitness:        fit,
		SuperavitTotal: superavit,
		CustoLogTotal:  custoLog,
		PenalidadeQual: penalQual,
		Violacoes:      violacoes,
		NumMatched:     matched,
		TraderStats:    stats,
	}
}
