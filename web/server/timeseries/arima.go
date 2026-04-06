package timeseries

// =============================================================================
// ARIMA — AutoRegressive Integrated Moving Average
//
// Modelo estatístico clássico para séries temporais:
//   AR(p): regressão linear nos p valores anteriores
//   I(d):  diferenciação d vezes para tornar estacionária
//   MA(q): média móvel dos q erros anteriores
//
// Implementação simplificada: AR(p) com diferenciação de ordem 1.
// Os coeficientes AR são estimados por Yule-Walker (autocorrelação).
//
// ARIMA(p,1,0) — foco no componente autoregressivo.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// TreinarARIMA implementa ARIMA(p,1,0) simplificado.
// p = windowSize (ordem AR), d = 1 (uma diferenciação).
func TreinarARIMA(cfg Config, data NormalizedData) TimeSeriesResult {
	start := time.Now()
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	// Usar preços originais para diferenciação
	closes := data.AllClose
	n := len(closes)
	if n < cfg.WindowSize+2 {
		res.TempoMs = time.Since(start).Milliseconds()
		return res
	}

	// Passo 1: Diferenciação (d=1) — tornar série estacionária
	// diff[i] = closes[i+1] - closes[i]
	diff := make([]float64, n-1)
	for i := range n - 1 {
		diff[i] = closes[i+1] - closes[i]
	}

	// Passo 2: Estimar coeficientes AR(p) via Yule-Walker
	p := cfg.WindowSize
	if p > len(diff)-1 { p = len(diff) - 1 }
	arCoeffs := estimateAR(diff, p)

	// Passo 3: Gerar predições para treino + validação
	validDays := cfg.ValidDays
	if cfg.ValidPct > 0 {
		validDays = int(float64(n) * cfg.ValidPct)
	}
	if validDays <= 0 { validDays = 7 }

	trainEnd := n - validDays
	if trainEnd < p+1 { trainEnd = p + 1 }

	var reaisValid, preditosValid []float64

	for i := p + 1; i < n; i++ {
		// Predição: somar diferenças preditas ao preço anterior
		predDiff := 0.0
		for j := range p {
			if i-1-j >= 0 && i-1-j < len(diff) {
				predDiff += arCoeffs[j] * diff[i-1-j]
			}
		}
		predPrice := closes[i-1] + predDiff
		realPrice := closes[i]

		date := ""
		if i < len(data.AllClose) && i-cfg.WindowSize >= 0 && i-cfg.WindowSize < len(data.Dates) {
			date = data.Dates[i-cfg.WindowSize]
		}

		pt := TimeSeriesPoint{Data: date, Preco: realPrice, Predito: predPrice}
		res.Pontos = append(res.Pontos, pt)

		if i >= trainEnd {
			res.PontosValid = append(res.PontosValid, pt)
			reaisValid = append(reaisValid, realPrice)
			preditosValid = append(preditosValid, predPrice)
		}
	}

	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(reaisValid, preditosValid)

	// Forecast
	forecastDays := cfg.ForecastDays
	if forecastDays <= 0 { forecastDays = 7 }

	confidence := res.RmseFinal
	if confidence < 0.01 { confidence = 0.01 }

	// Estender a série de diferenças para forecast
	extDiff := make([]float64, len(diff))
	copy(extDiff, diff)
	lastPrice := closes[n-1]

	for d := 1; d <= forecastDays; d++ {
		predDiff := 0.0
		for j := range p {
			idx := len(extDiff) - 1 - j
			if idx >= 0 {
				predDiff += arCoeffs[j] * extDiff[idx]
			}
		}
		lastPrice += predDiff
		extDiff = append(extDiff, predDiff)

		spread := confidence * math.Sqrt(float64(d))
		res.Forecast = append(res.Forecast, ForecastPoint{
			Dia: d, Predito: lastPrice, Upper: lastPrice + spread, Lower: lastPrice - spread,
		})
		if d == 1 { res.PredicaoAmanha = lastPrice }
	}

	res.TempoMs = time.Since(start).Milliseconds()
	return res
}

// estimateAR estima coeficientes AR(p) usando Yule-Walker.
// Resolve: R * φ = r, onde R é a matriz de autocorrelação e r o vetor.
func estimateAR(series []float64, p int) []float64 {
	n := len(series)
	if n <= p || p <= 0 {
		return make([]float64, p)
	}

	mean := stat.Mean(series, nil)

	// Calcular autocorrelações
	autocorr := make([]float64, p+1)
	for lag := range p + 1 {
		sum := 0.0
		for i := lag; i < n; i++ {
			sum += (series[i] - mean) * (series[i-lag] - mean)
		}
		autocorr[lag] = sum / float64(n)
	}

	// Evitar divisão por zero
	if autocorr[0] < 1e-12 {
		return make([]float64, p)
	}

	// Construir sistema Toeplitz: R * φ = r
	// R[i][j] = autocorr[|i-j|]
	R := mat.NewDense(p, p, nil)
	r := mat.NewVecDense(p, nil)
	for i := range p {
		r.SetVec(i, autocorr[i+1])
		for j := range p {
			idx := i - j
			if idx < 0 { idx = -idx }
			R.Set(i, j, autocorr[idx])
		}
	}

	// Resolver R * φ = r
	var phi mat.VecDense
	err := phi.SolveVec(R, r)
	if err != nil {
		// Fallback: coeficientes simples baseados em autocorrelação
		coeffs := make([]float64, p)
		for i := range p {
			coeffs[i] = autocorr[i+1] / autocorr[0]
			if math.IsNaN(coeffs[i]) { coeffs[i] = 0 }
		}
		return coeffs
	}

	coeffs := make([]float64, p)
	for i := range p {
		coeffs[i] = phi.AtVec(i)
		if math.IsNaN(coeffs[i]) { coeffs[i] = 0 }
	}
	return coeffs
}
