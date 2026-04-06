package timeseries

// =============================================================================
// LSTM — Long Short-Term Memory (Hochreiter & Schmidhuber 1997)
//
// Rede recorrente com mecanismo de gates que resolve o problema de
// vanishing gradient das RNNs simples. Tem 4 gates:
//
//   Forget gate:  f = σ(Wf·[h,x] + bf)    — o que esquecer do estado
//   Input gate:   i = σ(Wi·[h,x] + bi)    — o que adicionar ao estado
//   Cell cand.:   g = tanh(Wg·[h,x] + bg) — candidato a novo estado
//   Output gate:  o = σ(Wo·[h,x] + bo)    — o que expor como saída
//
//   Cell state:   c = f*c_prev + i*g       — memória de longo prazo
//   Hidden state: h = o * tanh(c)          — saída do timestep
//
// Treinamento: BPTT (Backpropagation Through Time) com gradient clipping.
// Usa gonum/mat para multiplicações de matrizes nos gates.
// =============================================================================

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
)

// LSTMNet é a rede LSTM completa com camada densa de saída
type LSTMNet struct {
	InputSize  int
	HiddenSize int
	// Gate weights: [combinedSize][hiddenSize] onde combinedSize = inputSize + hiddenSize
	Wf, Wi, Wg, Wo [][]float64 // forget, input, candidate, output
	Bf, Bi, Bg, Bo []float64   // biases
	// Dense output layer: hidden → 1
	Wd []float64 // [hiddenSize]
	Bd float64
}

// LSTMState armazena o estado interno em cada timestep (para BPTT)
type LSTMState struct {
	X         []float64 // input deste timestep
	Combined  []float64 // [h_prev, x] concatenado
	F, I, G, O []float64 // gate activations
	CellPrev  []float64 // cell state anterior
	Cell      []float64 // cell state atual
	Hidden    []float64 // hidden state atual
	TanhC     []float64 // tanh(cell) — necessário para backward
}

// NewLSTM cria uma LSTM com pesos inicializados (Xavier)
func NewLSTM(inputSize, hiddenSize int) *LSTMNet {
	rng := rand.New(rand.NewSource(42))
	combined := inputSize + hiddenSize
	scale := math.Sqrt(2.0 / float64(combined))

	net := &LSTMNet{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wf: randMatrix(rng, combined, hiddenSize, scale),
		Wi: randMatrix(rng, combined, hiddenSize, scale),
		Wg: randMatrix(rng, combined, hiddenSize, scale),
		Wo: randMatrix(rng, combined, hiddenSize, scale),
		Bf: make([]float64, hiddenSize),
		Bi: make([]float64, hiddenSize),
		Bg: make([]float64, hiddenSize),
		Bo: make([]float64, hiddenSize),
		Wd: make([]float64, hiddenSize),
		Bd: 0,
	}

	// Forget gate bias inicializado em 1.0 (recomendação Jozefowicz 2015)
	for j := range hiddenSize {
		net.Bf[j] = 1.0
		net.Wd[j] = rng.NormFloat64() * math.Sqrt(1.0/float64(hiddenSize))
	}

	return net
}

func randMatrix(rng *rand.Rand, rows, cols int, scale float64) [][]float64 {
	m := make([][]float64, rows)
	for i := range rows {
		m[i] = make([]float64, cols)
		for j := range cols {
			m[i][j] = rng.NormFloat64() * scale
		}
	}
	return m
}

// sigmoid e suas derivadas
func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// matVecMul calcula W·x + b usando gonum floats para eficiência
func matVecMul(W [][]float64, x, b []float64) []float64 {
	rows := len(W[0]) // output dim
	out := make([]float64, rows)
	copy(out, b)
	for i, xi := range x {
		floats.AddScaled(out, xi, W[i])
	}
	return out
}

// Forward processa uma sequência e retorna output + estados para BPTT
func (net *LSTMNet) Forward(sequence []float64) (float64, []LSTMState) {
	seqLen := len(sequence)
	states := make([]LSTMState, seqLen)

	h := make([]float64, net.HiddenSize)
	c := make([]float64, net.HiddenSize)

	for t := range seqLen {
		x := []float64{sequence[t]} // inputSize = 1

		// Concatenar [h, x]
		combined := make([]float64, net.HiddenSize+net.InputSize)
		copy(combined, h)
		copy(combined[net.HiddenSize:], x)

		// Gates
		fRaw := matVecMul(net.Wf, combined, net.Bf)
		iRaw := matVecMul(net.Wi, combined, net.Bi)
		gRaw := matVecMul(net.Wg, combined, net.Bg)
		oRaw := matVecMul(net.Wo, combined, net.Bo)

		f := make([]float64, net.HiddenSize)
		ig := make([]float64, net.HiddenSize)
		g := make([]float64, net.HiddenSize)
		o := make([]float64, net.HiddenSize)
		newC := make([]float64, net.HiddenSize)
		tanhC := make([]float64, net.HiddenSize)
		newH := make([]float64, net.HiddenSize)

		for j := range net.HiddenSize {
			f[j] = sigmoid(fRaw[j])
			ig[j] = sigmoid(iRaw[j])
			g[j] = math.Tanh(gRaw[j])
			o[j] = sigmoid(oRaw[j])
			newC[j] = f[j]*c[j] + ig[j]*g[j]
			tanhC[j] = math.Tanh(newC[j])
			newH[j] = o[j] * tanhC[j]
		}

		states[t] = LSTMState{
			X: x, Combined: combined,
			F: f, I: ig, G: g, O: o,
			CellPrev: append([]float64{}, c...),
			Cell: newC, Hidden: newH, TanhC: tanhC,
		}

		h = newH
		c = newC
	}

	// Dense output: y = Wd·h + Bd
	output := net.Bd
	output += floats.Dot(net.Wd, h)

	return output, states
}

// BackwardAndUpdate implementa BPTT com gradient clipping
func (net *LSTMNet) BackwardAndUpdate(states []LSTMState, target, output, alfa float64) {
	seqLen := len(states)

	// Delta da camada dense
	dOutput := target - output

	// Gradientes dense
	hFinal := states[seqLen-1].Hidden
	dWd := make([]float64, net.HiddenSize)
	dH := make([]float64, net.HiddenSize)
	for j := range net.HiddenSize {
		dWd[j] = dOutput * hFinal[j]
		dH[j] = dOutput * net.Wd[j]
	}

	// BPTT: propagar de trás pra frente
	dC := make([]float64, net.HiddenSize) // gradiente do cell state

	combined := net.HiddenSize + net.InputSize

	// Acumuladores de gradientes para pesos
	dWf := make([][]float64, combined)
	dWi := make([][]float64, combined)
	dWg := make([][]float64, combined)
	dWo := make([][]float64, combined)
	for i := range combined {
		dWf[i] = make([]float64, net.HiddenSize)
		dWi[i] = make([]float64, net.HiddenSize)
		dWg[i] = make([]float64, net.HiddenSize)
		dWo[i] = make([]float64, net.HiddenSize)
	}
	dBf := make([]float64, net.HiddenSize)
	dBi := make([]float64, net.HiddenSize)
	dBg := make([]float64, net.HiddenSize)
	dBo := make([]float64, net.HiddenSize)

	for t := seqLen - 1; t >= 0; t-- {
		st := states[t]

		// Gradiente do hidden state → cell state
		for j := range net.HiddenSize {
			dC[j] += dH[j] * st.O[j] * (1 - st.TanhC[j]*st.TanhC[j])
		}

		// Gradientes dos gates
		dF := make([]float64, net.HiddenSize)
		dI := make([]float64, net.HiddenSize)
		dG := make([]float64, net.HiddenSize)
		dO := make([]float64, net.HiddenSize)

		for j := range net.HiddenSize {
			dO[j] = dH[j] * st.TanhC[j] * st.O[j] * (1 - st.O[j])
			dF[j] = dC[j] * st.CellPrev[j] * st.F[j] * (1 - st.F[j])
			dI[j] = dC[j] * st.G[j] * st.I[j] * (1 - st.I[j])
			dG[j] = dC[j] * st.I[j] * (1 - st.G[j]*st.G[j])
		}

		// Acumular gradientes dos pesos
		for i := range combined {
			for j := range net.HiddenSize {
				dWf[i][j] += dF[j] * st.Combined[i]
				dWi[i][j] += dI[j] * st.Combined[i]
				dWg[i][j] += dG[j] * st.Combined[i]
				dWo[i][j] += dO[j] * st.Combined[i]
			}
		}
		for j := range net.HiddenSize {
			dBf[j] += dF[j]
			dBi[j] += dI[j]
			dBg[j] += dG[j]
			dBo[j] += dO[j]
		}

		// Propagar dH para o timestep anterior (via combined = [h_prev, x])
		dHPrev := make([]float64, net.HiddenSize)
		for i := range net.HiddenSize {
			for j := range net.HiddenSize {
				dHPrev[i] += dF[j]*net.Wf[i][j] + dI[j]*net.Wi[i][j] + dG[j]*net.Wg[i][j] + dO[j]*net.Wo[i][j]
			}
		}

		// Propagar dC para timestep anterior (via forget gate)
		newDC := make([]float64, net.HiddenSize)
		for j := range net.HiddenSize {
			newDC[j] = dC[j] * st.F[j]
		}

		dH = dHPrev
		dC = newDC
	}

	// Gradient clipping [-1, 1]
	clipGrad := func(g [][]float64) {
		for i := range g {
			for j := range g[i] {
				g[i][j] = math.Max(-1, math.Min(1, g[i][j]))
			}
		}
	}
	clipVec := func(g []float64) {
		for i := range g {
			g[i] = math.Max(-1, math.Min(1, g[i]))
		}
	}
	clipGrad(dWf); clipGrad(dWi); clipGrad(dWg); clipGrad(dWo)
	clipVec(dBf); clipVec(dBi); clipVec(dBg); clipVec(dBo)
	clipVec(dWd)

	// Atualizar pesos
	for i := range combined {
		for j := range net.HiddenSize {
			net.Wf[i][j] += alfa * dWf[i][j]
			net.Wi[i][j] += alfa * dWi[i][j]
			net.Wg[i][j] += alfa * dWg[i][j]
			net.Wo[i][j] += alfa * dWo[i][j]
		}
	}
	for j := range net.HiddenSize {
		net.Bf[j] += alfa * dBf[j]
		net.Bi[j] += alfa * dBi[j]
		net.Bg[j] += alfa * dBg[j]
		net.Bo[j] += alfa * dBo[j]
		net.Wd[j] += alfa * dWd[j]
	}
	net.Bd += alfa * dOutput
}

// TreinarLSTM treina uma LSTM nos dados de série temporal
func TreinarLSTM(cfg Config, data NormalizedData, ch chan<- TimeSeriesStep) (*LSTMNet, TimeSeriesResult) {
	start := time.Now()
	hidSize := cfg.HiddenSize
	if hidSize <= 0 { hidSize = 16 }
	lr := cfg.Alfa
	if lr <= 0 { lr = 0.001 }
	maxCiclo := cfg.MaxCiclo
	if maxCiclo <= 0 { maxCiclo = 2000 }

	net := NewLSTM(1, hidSize)
	nTrain := len(data.TrainX)

	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		mseTreino := 0.0
		for i := range nTrain {
			output, states := net.Forward(data.TrainX[i])
			d := data.TrainY[i] - output
			mseTreino += d * d
			net.BackwardAndUpdate(states, data.TrainY[i], output, lr)
		}
		mseTreino /= float64(nTrain)
		res.MseHistorico = append(res.MseHistorico, mseTreino)

		if ch != nil && ciclo%100 == 0 {
			mseValid := 0.0
			for i := range len(data.ValidX) {
				output, _ := net.Forward(data.ValidX[i])
				d := data.ValidY[i] - output
				mseValid += d * d
			}
			if len(data.ValidX) > 0 { mseValid /= float64(len(data.ValidX)) }
			select {
			case ch <- TimeSeriesStep{Ciclo: ciclo, MseTreino: mseTreino, MseValid: mseValid}:
			default:
			}
		}
	}

	// Gerar predições
	res.Ciclos = maxCiclo
	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)
	var reaisValid, preditosValid []float64

	for i := range len(allX) {
		output, _ := net.Forward(allX[i])
		precoReal := Desnormalizar(allY[i], data.MinPrice, data.MaxPrice)
		precoPred := Desnormalizar(output, data.MinPrice, data.MaxPrice)
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
			predNorm, _ := net.Forward(window)
			predPrice := Desnormalizar(predNorm, data.MinPrice, data.MaxPrice)
			spread := confidence * math.Sqrt(float64(d))
			res.Forecast = append(res.Forecast, ForecastPoint{Dia: d, Predito: predPrice, Upper: predPrice + spread, Lower: predPrice - spread})
			if d == 1 { res.PredicaoAmanha = predPrice }
			window = append(window[1:], predNorm)
		}
	}

	res.TempoMs = time.Since(start).Milliseconds()
	return net, res
}
