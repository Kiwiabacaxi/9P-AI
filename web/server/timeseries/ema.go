package timeseries

// =============================================================================
// EMA — Exponential Moving Average (Média Móvel Exponencial)
//
// Variação do SMA que dá mais peso aos dados recentes.
// α (alpha) controla o decaimento: α = 2/(N+1)
//
// Fórmula: EMA(t) = α * preço(t) + (1-α) * EMA(t-1)
//
// Melhor que SMA para capturar tendências recentes, mas ainda é baseline.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/stat"
)

// TreinarEMA calcula predições usando média móvel exponencial.
func TreinarEMA(cfg Config, data NormalizedData) TimeSeriesResult {
	start := time.Now()
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	alpha := 2.0 / (float64(cfg.WindowSize) + 1.0)

	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)

	var reaisValid, preditosValid []float64

	for i := range len(allX) {
		// EMA: começar com SMA da janela, depois aplicar exponencial
		window := allX[i]
		ema := stat.Mean(window, nil) // seed com SMA
		for _, v := range window {
			ema = alpha*v + (1-alpha)*ema
		}

		precoReal := Desnormalizar(allY[i], data.MinPrice, data.MaxPrice)
		precoPred := Desnormalizar(ema, data.MinPrice, data.MaxPrice)

		pt := TimeSeriesPoint{Data: data.Dates[i], Preco: precoReal, Predito: precoPred}
		res.Pontos = append(res.Pontos, pt)

		if i >= trainLen {
			res.PontosValid = append(res.PontosValid, pt)
			reaisValid = append(reaisValid, precoReal)
			preditosValid = append(preditosValid, precoPred)
		}
	}

	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(reaisValid, preditosValid)

	// Forecast
	forecastDays := cfg.ForecastDays
	if forecastDays <= 0 { forecastDays = 7 }
	rangeP := data.MaxPrice - data.MinPrice
	if rangeP < 0.0001 { rangeP = 1 }

	closes := data.AllClose
	if len(closes) >= cfg.WindowSize {
		window := make([]float64, cfg.WindowSize)
		for j := range cfg.WindowSize {
			window[j] = (closes[len(closes)-cfg.WindowSize+j] - data.MinPrice) / rangeP
		}

		confidence := res.RmseFinal
		if confidence < 0.01 { confidence = 0.01 }

		for d := 1; d <= forecastDays; d++ {
			ema := stat.Mean(window, nil)
			for _, v := range window {
				ema = alpha*v + (1-alpha)*ema
			}
			predPrice := Desnormalizar(ema, data.MinPrice, data.MaxPrice)
			spread := confidence * math.Sqrt(float64(d))

			res.Forecast = append(res.Forecast, ForecastPoint{
				Dia: d, Predito: predPrice, Upper: predPrice + spread, Lower: predPrice - spread,
			})
			if d == 1 { res.PredicaoAmanha = predPrice }
			window = append(window[1:], ema)
		}
	}

	res.TempoMs = time.Since(start).Milliseconds()
	return res
}
