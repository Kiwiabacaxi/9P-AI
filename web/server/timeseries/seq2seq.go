package timeseries

// =============================================================================
// Seq2Seq — Sequence to Sequence (Encoder-Decoder)
//
// Encoder LSTM processa a janela inteira → produz context (h, c).
// Decoder usa o context para gerar a predição.
//
// Simplificação: o decoder é uma camada densa sobre o hidden state
// do encoder (equivalente a um decoder de 1 step sem recorrência).
// Isso evita instabilidade do decoder LSTM autoregressivo em Go puro.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/floats"
)

type Seq2SeqNet struct {
	Encoder    *LSTMNet
	// Decoder simplificado: dense layer sobre o context do encoder
	Wd         []float64 // [hiddenSize] → 1
	Bd         float64
	HiddenSize int
}

func NewSeq2Seq(inputSize, hiddenSize int) *Seq2SeqNet {
	enc := NewLSTM(inputSize, hiddenSize)
	// Decoder dense separado (não usa o dense do encoder)
	wd := make([]float64, hiddenSize)
	for j := range hiddenSize {
		wd[j] = enc.Wd[j] // inicializar igual
	}
	return &Seq2SeqNet{
		Encoder: enc, Wd: wd, Bd: 0, HiddenSize: hiddenSize,
	}
}

func (net *Seq2SeqNet) Predict(sequence []float64) float64 {
	// Encoder: processar toda a sequência
	_, encStates := net.Encoder.Forward(sequence)
	hContext := encStates[len(encStates)-1].Hidden

	// Decoder: dense sobre context
	return net.Bd + floats.Dot(net.Wd, hContext)
}

func TreinarSeq2Seq(cfg Config, data NormalizedData, ch chan<- TimeSeriesStep) (*Seq2SeqNet, TimeSeriesResult) {
	start := time.Now()
	hidSize := cfg.HiddenSize
	if hidSize <= 0 { hidSize = 16 }
	lr := cfg.Alfa
	if lr <= 0 { lr = 0.001 }
	maxCiclo := cfg.MaxCiclo
	if maxCiclo <= 0 { maxCiclo = 1500 }

	net := NewSeq2Seq(1, hidSize)
	nTrain := len(data.TrainX)
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		mse := 0.0
		for i := range nTrain {
			// Encoder forward
			_, encStates := net.Encoder.Forward(data.TrainX[i])
			hCtx := encStates[len(encStates)-1].Hidden

			// Decoder: dense output
			output := net.Bd + floats.Dot(net.Wd, hCtx)

			d := data.TrainY[i] - output
			mse += d * d

			// Backward decoder dense
			dOutput := data.TrainY[i] - output
			for j := range net.HiddenSize {
				net.Wd[j] += lr * clip1(dOutput*hCtx[j])
			}
			net.Bd += lr * clip1(dOutput)

			// Backward encoder (propagar gradiente do dense ao hidden state)
			net.Encoder.BackwardAndUpdate(encStates, data.TrainY[i], output, lr)
		}
		mse /= float64(nTrain)
		res.MseHistorico = append(res.MseHistorico, mse)

		if ch != nil && ciclo%100 == 0 {
			mseV := 0.0
			for i := range len(data.ValidX) {
				o := net.Predict(data.ValidX[i])
				d := data.ValidY[i] - o
				mseV += d * d
			}
			if len(data.ValidX) > 0 { mseV /= float64(len(data.ValidX)) }
			select { case ch <- TimeSeriesStep{Ciclo: ciclo, MseTreino: mse, MseValid: mseV}: default: }
		}
	}

	// Predições
	res.Ciclos = maxCiclo
	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)
	var rv, pv []float64
	for i := range len(allX) {
		o := net.Predict(allX[i])
		r := Desnormalizar(allY[i], data.MinPrice, data.MaxPrice)
		p := Desnormalizar(o, data.MinPrice, data.MaxPrice)
		pt := TimeSeriesPoint{Data: data.Dates[i], Preco: r, Predito: p}
		res.Pontos = append(res.Pontos, pt)
		if i >= trainLen { res.PontosValid = append(res.PontosValid, pt); rv = append(rv, r); pv = append(pv, p) }
	}
	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(rv, pv)

	// Forecast
	fd := cfg.ForecastDays; if fd <= 0 { fd = 7 }
	rP := data.MaxPrice - data.MinPrice; if rP < 0.0001 { rP = 1 }
	cls := data.AllClose
	if len(cls) >= cfg.WindowSize {
		w := make([]float64, cfg.WindowSize)
		for j := range cfg.WindowSize { w[j] = (cls[len(cls)-cfg.WindowSize+j] - data.MinPrice) / rP }
		conf := res.RmseFinal; if conf < 0.01 { conf = 0.01 }
		for d := 1; d <= fd; d++ {
			pn := net.Predict(w)
			pp := Desnormalizar(pn, data.MinPrice, data.MaxPrice)
			sp := conf * math.Sqrt(float64(d))
			res.Forecast = append(res.Forecast, ForecastPoint{Dia: d, Predito: pp, Upper: pp + sp, Lower: pp - sp})
			if d == 1 { res.PredicaoAmanha = pp }
			w = append(w[1:], pn)
		}
	}
	res.TempoMs = time.Since(start).Milliseconds()
	return net, res
}
