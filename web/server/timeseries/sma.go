package timeseries

// =============================================================================
// SMA — Simple Moving Average (Média Móvel Simples)
//
// Modelo baseline mais simples: predição = média dos últimos N dias.
// Sem treinamento — apenas calcula médias sobre janelas deslizantes.
//
// Fórmula: SMA(t) = (1/N) * Σ(preço(t-i)) para i=1..N
//
// Útil como baseline para comparar com modelos mais sofisticados.
// Se um LSTM não bater o SMA, algo está errado.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/stat"
)

// TreinarSMA calcula predições usando média móvel simples.
// Não há treinamento real — apenas cálculo de médias.
func TreinarSMA(cfg Config, data NormalizedData) TimeSeriesResult {
	start := time.Now()
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)

	var reaisValid, preditosValid []float64

	for i := range len(allX) {
		// SMA: média da janela de entrada
		predNorm := stat.Mean(allX[i], nil)
		precoReal := Desnormalizar(allY[i], data.MinPrice, data.MaxPrice)
		precoPred := Desnormalizar(predNorm, data.MinPrice, data.MaxPrice)

		pt := TimeSeriesPoint{Data: data.Dates[i], Preco: precoReal, Predito: precoPred}
		res.Pontos = append(res.Pontos, pt)

		if i >= trainLen {
			res.PontosValid = append(res.PontosValid, pt)
			reaisValid = append(reaisValid, precoReal)
			preditosValid = append(preditosValid, precoPred)
		}
	}

	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(reaisValid, preditosValid)

	// Forecast: deslizar janela com predições anteriores
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
			predNorm := stat.Mean(window, nil)
			predPrice := Desnormalizar(predNorm, data.MinPrice, data.MaxPrice)
			spread := confidence * math.Sqrt(float64(d))

			res.Forecast = append(res.Forecast, ForecastPoint{
				Dia: d, Predito: predPrice, Upper: predPrice + spread, Lower: predPrice - spread,
			})
			if d == 1 { res.PredicaoAmanha = predPrice }
			window = append(window[1:], predNorm)
		}
	}

	res.TempoMs = time.Since(start).Milliseconds()
	return res
}
