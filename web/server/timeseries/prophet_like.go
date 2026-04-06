package timeseries

// =============================================================================
// Prophet-like — Decomposição Tendência + Sazonalidade (Go puro)
//
// Versão simplificada do Prophet da Meta:
//   y(t) = trend(t) + seasonal(t) + noise(t)
//
//   Tendência: regressão linear nos preços
//   Sazonalidade: padrão semanal (5 dias úteis)
//   Predição: trend + seasonal extrapolados
//
// Não é o Prophet real (que usa decomposição bayesiana + Stan),
// mas demonstra o conceito de forma didática.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/stat"
)

// TreinarProphetLike implementa decomposição trend + seasonality.
func TreinarProphetLike(cfg Config, data NormalizedData) TimeSeriesResult {
	start := time.Now()
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	closes := data.AllClose
	n := len(closes)
	if n < 10 {
		res.TempoMs = time.Since(start).Milliseconds()
		return res
	}

	// Passo 1: Tendência via regressão linear (apenas últimos 30 dias para evitar viés)
	trendWindow := 30
	if trendWindow > n { trendWindow = n }
	trendCloses := closes[n-trendWindow:]
	trendXs := make([]float64, trendWindow)
	for i := range trendWindow { trendXs[i] = float64(n - trendWindow + i) }

	alpha, beta := stat.LinearRegression(trendXs, trendCloses, nil, false)
	// trend(t) = alpha + beta * t
	// Damping: limitar beta para não extrapolar demais
	maxBeta := 0.05 // máximo ~5 centavos por dia
	if beta > maxBeta { beta = maxBeta }
	if beta < -maxBeta { beta = -maxBeta }

	// Passo 2: Remover tendência para extrair sazonalidade
	detrended := make([]float64, n)
	for i := range n {
		detrended[i] = closes[i] - (alpha + beta*float64(i))
	}

	// Passo 3: Sazonalidade semanal (período = 5 dias úteis)
	period := 5
	seasonal := make([]float64, period)
	counts := make([]float64, period)
	for i := range n {
		idx := i % period
		seasonal[idx] += detrended[i]
		counts[idx]++
	}
	for i := range period {
		if counts[i] > 0 { seasonal[i] /= counts[i] }
	}

	// Centralizar sazonalidade (soma = 0)
	sMean := stat.Mean(seasonal, nil)
	for i := range period { seasonal[i] -= sMean }

	// Passo 4: Gerar predições
	validDays := cfg.ValidDays
	if cfg.ValidPct > 0 { validDays = int(float64(n) * cfg.ValidPct) }
	if validDays <= 0 { validDays = 7 }
	trainEnd := n - validDays
	if trainEnd < 2 { trainEnd = 2 }

	var reaisValid, preditosValid []float64

	for i := 1; i < n; i++ {
		trend := alpha + beta*float64(i)
		seas := seasonal[i%period]
		pred := trend + seas

		date := ""
		if i-cfg.WindowSize >= 0 && i-cfg.WindowSize < len(data.Dates) {
			date = data.Dates[i-cfg.WindowSize]
		} else if i < len(data.Dates) {
			date = data.Dates[i]
		}

		pt := TimeSeriesPoint{Data: date, Preco: closes[i], Predito: pred}
		res.Pontos = append(res.Pontos, pt)

		if i >= trainEnd {
			res.PontosValid = append(res.PontosValid, pt)
			reaisValid = append(reaisValid, closes[i])
			preditosValid = append(preditosValid, pred)
		}
	}

	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(reaisValid, preditosValid)

	// VizData: decomposição para visualização
	trendVals := make([]float64, n)
	seasonalVals := make([]float64, n)
	residualVals := make([]float64, n)
	for i := range n {
		trendVals[i] = alpha + beta*float64(i)
		seasonalVals[i] = seasonal[i%period]
		residualVals[i] = closes[i] - trendVals[i] - seasonalVals[i]
	}
	res.VizData = map[string]any{
		"trend":     trendVals,
		"seasonal":  seasonalVals,
		"residual":  residualVals,
		"period":    period,
		"alpha":     alpha,
		"beta":      beta,
		"seasonalPattern": seasonal,
	}

	// Forecast: extrapolar trend + seasonal
	fd := cfg.ForecastDays; if fd <= 0 { fd = 7 }
	conf := res.RmseFinal; if conf < 0.01 { conf = 0.01 }

	for d := 1; d <= fd; d++ {
		t := float64(n + d - 1)
		trend := alpha + beta*t
		seas := seasonal[(n+d-1)%period]
		pred := trend + seas
		spread := conf * math.Sqrt(float64(d))

		res.Forecast = append(res.Forecast, ForecastPoint{
			Dia: d, Predito: pred, Upper: pred + spread, Lower: pred - spread,
		})
		if d == 1 { res.PredicaoAmanha = pred }
	}

	res.TempoMs = time.Since(start).Milliseconds()
	return res
}
