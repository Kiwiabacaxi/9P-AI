package timeseries

// =============================================================================
// GRU — Gated Recurrent Unit (Cho et al. 2014)
//
// Versão simplificada do LSTM com apenas 2 gates:
//   Reset gate:  r = σ(Wr·[h,x] + br)   — quanto do estado anterior usar
//   Update gate: z = σ(Wz·[h,x] + bz)   — balanceia estado antigo vs novo
//   Candidate:   h̃ = tanh(Wh·[r*h,x] + bh)
//   Output:      h = (1-z)*h_prev + z*h̃
//
// Menos parâmetros que LSTM → treina mais rápido.
// Performance comparável ao LSTM na maioria dos casos.
// =============================================================================

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
)

// GRUNet é a rede GRU com camada densa de saída
type GRUNet struct {
	InputSize  int
	HiddenSize int
	Wr, Wz, Wh [][]float64 // [combined][hidden] — reset, update, candidate
	Br, Bz, Bh []float64   // biases
	Wd         []float64   // dense: hidden → 1
	Bd         float64
}

type GRUState struct {
	Combined []float64
	R, Z     []float64 // gate activations
	HCand    []float64 // candidate hidden
	HPrev    []float64 // previous hidden
	Hidden   []float64 // new hidden
}

func NewGRU(inputSize, hiddenSize int) *GRUNet {
	rng := rand.New(rand.NewSource(42))
	combined := inputSize + hiddenSize
	scale := math.Sqrt(2.0 / float64(combined))
	net := &GRUNet{
		InputSize: inputSize, HiddenSize: hiddenSize,
		Wr: randMatrix(rng, combined, hiddenSize, scale),
		Wz: randMatrix(rng, combined, hiddenSize, scale),
		Wh: randMatrix(rng, combined, hiddenSize, scale),
		Br: make([]float64, hiddenSize),
		Bz: make([]float64, hiddenSize),
		Bh: make([]float64, hiddenSize),
		Wd: make([]float64, hiddenSize),
	}
	for j := range hiddenSize {
		net.Wd[j] = rng.NormFloat64() * math.Sqrt(1.0/float64(hiddenSize))
	}
	return net
}

func (net *GRUNet) Forward(sequence []float64) (float64, []GRUState) {
	seqLen := len(sequence)
	states := make([]GRUState, seqLen)
	h := make([]float64, net.HiddenSize)

	for t := range seqLen {
		x := []float64{sequence[t]}
		combined := append(append([]float64{}, h...), x...)

		rGate := matVecMul(net.Wr, combined, net.Br)
		zGate := matVecMul(net.Wz, combined, net.Bz)

		r := make([]float64, net.HiddenSize)
		z := make([]float64, net.HiddenSize)
		for j := range net.HiddenSize {
			r[j] = sigmoid(rGate[j])
			z[j] = sigmoid(zGate[j])
		}

		// Candidate: h̃ = tanh(Wh·[r*h, x] + bh)
		rhCombined := make([]float64, net.HiddenSize+net.InputSize)
		for j := range net.HiddenSize {
			rhCombined[j] = r[j] * h[j]
		}
		copy(rhCombined[net.HiddenSize:], x)
		hCandRaw := matVecMul(net.Wh, rhCombined, net.Bh)
		hCand := make([]float64, net.HiddenSize)
		for j := range net.HiddenSize {
			hCand[j] = math.Tanh(hCandRaw[j])
		}

		// h = (1-z)*h_prev + z*h̃
		newH := make([]float64, net.HiddenSize)
		for j := range net.HiddenSize {
			newH[j] = (1-z[j])*h[j] + z[j]*hCand[j]
		}

		states[t] = GRUState{Combined: combined, R: r, Z: z, HCand: hCand, HPrev: append([]float64{}, h...), Hidden: newH}
		h = newH
	}

	output := net.Bd + floats.Dot(net.Wd, h)
	return output, states
}

func (net *GRUNet) BackwardAndUpdate(states []GRUState, target, output, alfa float64) {
	seqLen := len(states)
	dOutput := target - output
	hFinal := states[seqLen-1].Hidden

	dH := make([]float64, net.HiddenSize)
	for j := range net.HiddenSize {
		dH[j] = dOutput * net.Wd[j]
		net.Wd[j] += alfa * clip1(dOutput*hFinal[j])
	}
	net.Bd += alfa * clip1(dOutput)

	combined := net.HiddenSize + net.InputSize

	for t := seqLen - 1; t >= 0; t-- {
		st := states[t]
		dZ := make([]float64, net.HiddenSize)
		dHCand := make([]float64, net.HiddenSize)
		dHPrev := make([]float64, net.HiddenSize)

		for j := range net.HiddenSize {
			dZ[j] = dH[j] * (st.HCand[j] - st.HPrev[j]) * st.Z[j] * (1 - st.Z[j])
			dHCand[j] = dH[j] * st.Z[j] * (1 - st.HCand[j]*st.HCand[j])
			dHPrev[j] = dH[j] * (1 - st.Z[j])
		}

		// Update Wz, Wr, Wh
		for i := range combined {
			for j := range net.HiddenSize {
				net.Wz[i][j] += alfa * clip1(dZ[j]*st.Combined[i])
				net.Wr[i][j] += alfa * clip1(dHCand[j]*st.R[j]*st.Combined[i])
			}
		}
		for j := range net.HiddenSize {
			net.Bz[j] += alfa * clip1(dZ[j])
			net.Bh[j] += alfa * clip1(dHCand[j])
		}

		// Propagar para timestep anterior
		for i := range net.HiddenSize {
			for j := range net.HiddenSize {
				dHPrev[i] += dZ[j]*net.Wz[i][j] + dHCand[j]*net.Wr[i][j]
			}
		}
		dH = dHPrev
	}
}

func clip1(x float64) float64 { return math.Max(-1, math.Min(1, x)) }

// TreinarGRU treina uma GRU nos dados de série temporal
func TreinarGRU(cfg Config, data NormalizedData, ch chan<- TimeSeriesStep) (*GRUNet, TimeSeriesResult) {
	start := time.Now()
	hidSize := cfg.HiddenSize
	if hidSize <= 0 { hidSize = 16 }
	lr := cfg.Alfa
	if lr <= 0 { lr = 0.001 }
	maxCiclo := cfg.MaxCiclo
	if maxCiclo <= 0 { maxCiclo = 2000 }

	net := NewGRU(1, hidSize)
	nTrain := len(data.TrainX)
	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		mse := 0.0
		for i := range nTrain {
			output, states := net.Forward(data.TrainX[i])
			d := data.TrainY[i] - output
			mse += d * d
			net.BackwardAndUpdate(states, data.TrainY[i], output, lr)
		}
		mse /= float64(nTrain)
		res.MseHistorico = append(res.MseHistorico, mse)
		if ch != nil && ciclo%100 == 0 {
			mseV := 0.0
			for i := range len(data.ValidX) {
				o, _ := net.Forward(data.ValidX[i])
				d := data.ValidY[i] - o
				mseV += d * d
			}
			if len(data.ValidX) > 0 { mseV /= float64(len(data.ValidX)) }
			select { case ch <- TimeSeriesStep{Ciclo: ciclo, MseTreino: mse, MseValid: mseV}: default: }
		}
	}

	// Predições + forecast (mesmo padrão do LSTM/MLP)
	res.Ciclos = maxCiclo
	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)
	var rv, pv []float64
	for i := range len(allX) {
		o, _ := net.Forward(allX[i])
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
			pn, _ := net.Forward(w)
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
