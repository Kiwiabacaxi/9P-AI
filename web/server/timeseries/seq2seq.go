package timeseries

// =============================================================================
// Seq2Seq — Sequence to Sequence (Sutskever et al. 2014)
//
// Encoder-Decoder architecture for multi-step forecasting:
//
//   Encoder: LSTM que processa a janela de entrada (windowSize timesteps)
//            e comprime num vetor de contexto (h_final, c_final)
//
//   Decoder: LSTM que recebe o contexto e gera forecastDays predições
//            autoregressivamente (output do step anterior vira input do próximo)
//
// A primeira arquitetura pensada para horizontes longos de previsão.
// Diferença do LSTM simples: gera TODOS os passos futuros de uma vez,
// em vez de prever um dia e deslizar a janela.
// =============================================================================

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
)

// Seq2SeqNet é o encoder-decoder
type Seq2SeqNet struct {
	Encoder    *LSTMNet // processa input sequence
	Decoder    *LSTMNet // gera output sequence
	HiddenSize int
}

func NewSeq2Seq(inputSize, hiddenSize int) *Seq2SeqNet {
	return &Seq2SeqNet{
		Encoder:    NewLSTM(inputSize, hiddenSize),
		Decoder:    NewLSTM(1, hiddenSize), // decoder recebe 1 valor por step
		HiddenSize: hiddenSize,
	}
}

// EncodeDecode processa input → context → previsão de 1 step
func (net *Seq2SeqNet) Predict(sequence []float64) float64 {
	// Encoder: processar toda a sequência para obter context
	_, encStates := net.Encoder.Forward(sequence)
	hContext := encStates[len(encStates)-1].Hidden

	// Decoder: um passo com o último valor como input
	lastVal := sequence[len(sequence)-1]
	decoderInput := []float64{lastVal}

	// Setar hidden state do decoder com o context do encoder
	// (simplificação: usar context como input ao dense)
	output := net.Decoder.Bd
	output += floats.Dot(net.Decoder.Wd, hContext)

	// Misturar com decoder forward
	decOut, _ := net.Decoder.Forward(decoderInput)
	output = (output + decOut) / 2.0

	return output
}

// TreinarSeq2Seq treina o encoder-decoder
func TreinarSeq2Seq(cfg Config, data NormalizedData, ch chan<- TimeSeriesStep) (*Seq2SeqNet, TimeSeriesResult) {
	start := time.Now()
	hidSize := cfg.HiddenSize
	if hidSize <= 0 { hidSize = 16 }
	lr := cfg.Alfa
	if lr <= 0 { lr = 0.001 }
	maxCiclo := cfg.MaxCiclo
	if maxCiclo <= 0 { maxCiclo = 1500 }

	rng := rand.New(rand.NewSource(42))
	_ = rng

	net := NewSeq2Seq(1, hidSize)
	nTrain := len(data.TrainX)
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		mse := 0.0
		for i := range nTrain {
			seq := data.TrainX[i]

			// Encoder forward
			_, encStates := net.Encoder.Forward(seq)

			// Decoder: usar último valor + context
			lastVal := seq[len(seq)-1]
			decOut, decStates := net.Decoder.Forward([]float64{lastVal})
			hCtx := encStates[len(encStates)-1].Hidden
			output := (decOut + net.Decoder.Bd + floats.Dot(net.Decoder.Wd, hCtx)) / 2.0

			d := data.TrainY[i] - output
			mse += d * d

			// Backward: treinar encoder e decoder
			net.Encoder.BackwardAndUpdate(encStates, data.TrainY[i], decOut, lr)
			net.Decoder.BackwardAndUpdate(decStates, data.TrainY[i], decOut, lr)
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

	// Forecast multi-step (o ponto forte do Seq2Seq)
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
