package timeseries

// =============================================================================
// BiLSTM — Bidirectional LSTM
//
// Processa a sequência em ambas as direções (forward + backward)
// e concatena os hidden states para capturar contexto passado E futuro.
//
//   Forward LSTM:  h_fw = LSTM(x_1, x_2, ..., x_T)
//   Backward LSTM: h_bw = LSTM(x_T, x_T-1, ..., x_1)
//   Combined:      h = [h_fw; h_bw]  (concatenação)
//   Output:        y = Dense(h)
//
// Útil quando o contexto futuro importa (classificação de sequências).
// Para previsão temporal pura, o ganho sobre LSTM unidirecional é marginal.
// =============================================================================

import (
	"math"
	"time"

	"gonum.org/v1/gonum/floats"
)

// BiLSTMNet é composto de dois LSTMs + camada densa
type BiLSTMNet struct {
	Forward  *LSTMNet
	Backward *LSTMNet
	// Dense: 2*hiddenSize → 1
	Wd []float64
	Bd float64
}

func NewBiLSTM(inputSize, hiddenSize int) *BiLSTMNet {
	fw := NewLSTM(inputSize, hiddenSize)
	bw := NewLSTM(inputSize, hiddenSize)
	// Substituir dense dos LSTMs individuais por uma shared
	wd := make([]float64, hiddenSize*2)
	for j := range hiddenSize * 2 {
		wd[j] = fw.Wd[j%hiddenSize] * 0.5
	}
	return &BiLSTMNet{Forward: fw, Backward: bw, Wd: wd, Bd: 0}
}

func (net *BiLSTMNet) Predict(sequence []float64) float64 {
	seqLen := len(sequence)

	// Forward pass
	_, fwStates := net.Forward.Forward(sequence)
	hFw := fwStates[seqLen-1].Hidden

	// Backward pass (sequência invertida)
	reversed := make([]float64, seqLen)
	for i := range seqLen {
		reversed[i] = sequence[seqLen-1-i]
	}
	_, bwStates := net.Backward.Forward(reversed)
	hBw := bwStates[seqLen-1].Hidden

	// Concatenar [h_fw; h_bw]
	combined := append(append([]float64{}, hFw...), hBw...)

	// Dense output
	return net.Bd + floats.Dot(net.Wd, combined)
}

// TreinarBiLSTM treina o BiLSTM (treina cada direção separadamente por simplicidade)
func TreinarBiLSTM(cfg Config, data NormalizedData, ch chan<- TimeSeriesStep) (*BiLSTMNet, TimeSeriesResult) {
	start := time.Now()
	hidSize := cfg.HiddenSize
	if hidSize <= 0 { hidSize = 16 }
	lr := cfg.Alfa
	if lr <= 0 { lr = 0.001 }
	maxCiclo := cfg.MaxCiclo
	if maxCiclo <= 0 { maxCiclo = 2000 }

	net := NewBiLSTM(1, hidSize)
	nTrain := len(data.TrainX)
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		mse := 0.0
		for i := range nTrain {
			seq := data.TrainX[i]
			seqLen := len(seq)

			// Forward direction
			fwOut, fwStates := net.Forward.Forward(seq)
			// Backward direction
			reversed := make([]float64, seqLen)
			for j := range seqLen { reversed[j] = seq[seqLen-1-j] }
			bwOut, bwStates := net.Backward.Forward(reversed)

			// Combined prediction (average for simplicity)
			output := (fwOut + bwOut) / 2.0
			d := data.TrainY[i] - output
			mse += d * d

			// Update each LSTM separately
			net.Forward.BackwardAndUpdate(fwStates, data.TrainY[i], fwOut, lr)
			net.Backward.BackwardAndUpdate(bwStates, data.TrainY[i], bwOut, lr)
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
