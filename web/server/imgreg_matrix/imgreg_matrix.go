package imgreg_matrix

import (
	"context"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Config struct {
	HiddenLayers    int     `json:"hiddenLayers"`
	NeuronsPerLayer int     `json:"neuronsPerLayer"`
	LearningRate    float64 `json:"learningRate"`
	Imagem          string  `json:"imagem"`
	MaxEpocas       int     `json:"maxEpocas"`
}

type Step struct {
	Epoca         int          `json:"epoca"`
	MaxEpocas     int          `json:"maxEpocas"`
	Loss          float64      `json:"loss"`
	OutputPixels  [][3]float64 `json:"outputPixels"`
	ActiveLayer   int          `json:"activeLayer"`
	Done          bool         `json:"done"`
	Convergiu     bool         `json:"convergiu"`
	LossHistorico []float64    `json:"lossHistorico"`
	ElapsedMs     int64        `json:"elapsedMs"`
	EpochMs       int64        `json:"epochMs"`
}

// Net stores weights as Go slices (for JSON serialization and initialization).
// During training, Wm/Bvec hold persistent gonum Dense matrices to avoid
// per-epoch allocation and data copying.
type Net struct {
	LayerSizes []int         `json:"layerSizes"`
	W          [][][]float64 `json:"-"`
	B          [][]float64   `json:"-"`
	// Persistent gonum matrices, populated by buildMatrices()
	Wm   []*mat.Dense `json:"-"`
	Bvec [][]float64  `json:"-"` // bias per layer (plain slice, added row-by-row)
}

func relu(x float64) float64      { if x > 0 { return x }; return 0 }
func reluDeriv(z float64) float64 { if z > 0 { return 1 }; return 0 }
func sigmoid(x float64) float64 {
	if x < -500 {
		return 0
	}
	if x > 500 {
		return 1
	}
	return 1.0 / (1.0 + math.Exp(-x))
}
func sigmoidDeriv(y float64) float64 { return y * (1 - y) }

// =============================================================================
// INICIALIZAÇÃO DA REDE
// =============================================================================

func inicializar(cfg Config, rng *rand.Rand) Net {
	sizes := make([]int, 0, cfg.HiddenLayers+2)
	sizes = append(sizes, 2)
	for i := 0; i < cfg.HiddenLayers; i++ {
		sizes = append(sizes, cfg.NeuronsPerLayer)
	}
	sizes = append(sizes, 3)

	nLayers := len(sizes) - 1
	W := make([][][]float64, nLayers)
	B := make([][]float64, nLayers)

	for l := 0; l < nLayers; l++ {
		fanIn := sizes[l]
		fanOut := sizes[l+1]
		scale := math.Sqrt(2.0 / float64(fanIn))
		W[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			W[l][i] = make([]float64, fanOut)
			for j := 0; j < fanOut; j++ {
				W[l][i][j] = rng.NormFloat64() * scale
			}
		}
		B[l] = make([]float64, fanOut)
	}

	net := Net{LayerSizes: sizes, W: W, B: B}
	net.buildMatrices()
	return net
}

// buildMatrices creates persistent gonum Dense matrices from W/B slices.
// Called once after initialization (and after weight updates) so that Mul
// operates directly on the matrix data without per-epoch copying.
func (n *Net) buildMatrices() {
	nT := len(n.W)
	n.Wm = make([]*mat.Dense, nT)
	n.Bvec = make([][]float64, nT)
	for l := 0; l < nT; l++ {
		fanIn := len(n.W[l])
		fanOut := len(n.W[l][0])
		data := make([]float64, fanIn*fanOut)
		for i := 0; i < fanIn; i++ {
			for j := 0; j < fanOut; j++ {
				data[i*fanOut+j] = n.W[l][i][j]
			}
		}
		n.Wm[l] = mat.NewDense(fanIn, fanOut, data)
		b := make([]float64, fanOut)
		copy(b, n.B[l])
		n.Bvec[l] = b
	}
}

// syncFromMatrices writes Wm/Bvec back to W/B slices (for predict/JSON).
func (n *Net) syncFromMatrices() {
	for l, Wl := range n.Wm {
		fanIn := len(n.W[l])
		fanOut := len(n.W[l][0])
		for i := 0; i < fanIn; i++ {
			for j := 0; j < fanOut; j++ {
				n.W[l][i][j] = Wl.At(i, j)
			}
		}
		copy(n.B[l], n.Bvec[l])
	}
}

// =============================================================================
// GERAÇÃO DE IMAGENS-ALVO (16×16 pixels, proceduralmente)
// =============================================================================

func GetTarget(nome string) [][3]float64 {
	switch nome {
	case "smiley":
		return imgSmiley()
	case "radial":
		return imgRadial()
	case "brasil":
		return imgBrasil()
	default:
		return imgCoracao()
	}
}

func imgCoracao() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)
			cx, cy := x*1.3, y*1.3
			x2, y2 := cx*cx, cy*cy
			f := (x2+y2-1)*(x2+y2-1)*(x2+y2-1) - x2*cy*y2
			var r, g, b float64
			if f <= 0 {
				intensity := 1.0 - math.Sqrt(x2+y2)*0.3
				intensity = math.Max(0.6, math.Min(1.0, intensity))
				r = intensity; g = 0.05; b = 0.1
			} else {
				r = 0.04; g = 0.05; b = 0.08
			}
			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

func imgSmiley() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)
			dist := math.Sqrt(x*x + y*y)
			var r, g, b float64
			r, g, b = 0.05, 0.08, 0.25
			if dist < 0.85 { r, g, b = 0.95, 0.85, 0.05 }
			if math.Sqrt((x+0.28)*(x+0.28)+(y-0.28)*(y-0.28)) < 0.14 { r, g, b = 0.1, 0.07, 0.05 }
			if math.Sqrt((x-0.28)*(x-0.28)+(y-0.28)*(y-0.28)) < 0.14 { r, g, b = 0.1, 0.07, 0.05 }
			dS := math.Sqrt(x*x + (y+0.1)*(y+0.1))
			if dS > 0.38 && dS < 0.55 && y < -0.05 { r, g, b = 0.1, 0.07, 0.05 }
			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

func imgRadial() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0
			dist := math.Sqrt(x*x + y*y)
			angle := math.Atan2(y, x)
			wave := math.Sin(dist*8.0)*0.5 + 0.5
			angWave := math.Cos(angle*3.0+dist*4.0)*0.3 + 0.7
			pixels[py*16+px] = [3]float64{wave * angWave, math.Sin(dist*6.0+1.0)*0.5 + 0.5, math.Cos(dist*10.0)*0.5 + 0.5}
		}
	}
	return pixels
}

func imgBrasil() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)
			var r, g, b float64
			r, g, b = 0.0, 0.5, 0.15
			if math.Abs(x)*0.9+math.Abs(y) < 0.65 { r, g, b = 0.95, 0.80, 0.0 }
			dist := math.Sqrt(x*x + y*y)
			if dist < 0.35 { r, g, b = 0.0, 0.2, 0.7 }
			if dist < 0.35 && math.Abs(y) < 0.06 { r, g, b = 0.95, 0.95, 0.95 }
			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

// =============================================================================
// PREDICT (slice-based, single-sample)
// =============================================================================

func predict(net Net) [][3]float64 {
	pixels := make([][3]float64, 256)
	nLayers := len(net.LayerSizes)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0
			acts := make([][]float64, nLayers)
			acts[0] = []float64{x, y}
			for l := 0; l < len(net.W); l++ {
				fanIn, fanOut := len(net.W[l]), len(net.W[l][0])
				z := make([]float64, fanOut)
				a := make([]float64, fanOut)
				for j := 0; j < fanOut; j++ {
					z[j] = net.B[l][j]
					for i := 0; i < fanIn; i++ {
						z[j] += acts[l][i] * net.W[l][i][j]
					}
					if l == len(net.W)-1 {
						a[j] = sigmoid(z[j])
					} else {
						a[j] = relu(z[j])
					}
				}
				acts[l+1] = a
			}
			out := acts[nLayers-1]
			pixels[py*16+px] = [3]float64{out[0], out[1], out[2]}
		}
	}
	return pixels
}

// =============================================================================
// TREINAR — forward/backward matricial com gonum (matrizes persistentes)
// =============================================================================

func Treinar(ctx context.Context, cfg Config, progressCh chan<- Step) Net {
	rng := rand.New(rand.NewSource(42))
	if cfg.MaxEpocas <= 0 {
		cfg.MaxEpocas = 2000
	}
	if cfg.HiddenLayers < 1 {
		cfg.HiddenLayers = 2
	}
	if cfg.NeuronsPerLayer < 4 {
		cfg.NeuronsPerLayer = 16
	}

	net := inicializar(cfg, rng)
	target := GetTarget(cfg.Imagem)
	start := time.Now()

	// Input matrix A0: 256×2 — fixo, nunca muda
	inputData := make([]float64, 256*2)
	targetData := make([]float64, 256*3)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			idx := py*16 + px
			inputData[idx*2+0] = (float64(px)/15.0)*2.0 - 1.0
			inputData[idx*2+1] = (float64(py)/15.0)*2.0 - 1.0
			targetData[idx*3+0] = target[idx][0]
			targetData[idx*3+1] = target[idx][1]
			targetData[idx*3+2] = target[idx][2]
		}
	}
	A0 := mat.NewDense(256, 2, inputData)
	T := mat.NewDense(256, 3, targetData)

	nT := len(net.Wm)

	// Pré-aloca buffers reutilizáveis para forward/backward (sem alloc por época)
	acts := make([]*mat.Dense, nT+1)
	preActs := make([]*mat.Dense, nT+1)
	acts[0] = A0
	preActs[0] = A0
	for l := 0; l < nT; l++ {
		fanOut := net.LayerSizes[l+1]
		acts[l+1] = mat.NewDense(256, fanOut, nil)
		preActs[l+1] = mat.NewDense(256, fanOut, nil)
	}
	dZ := make([]*mat.Dense, nT+1)
	for l := 1; l <= nT; l++ {
		fanOut := net.LayerSizes[l]
		dZ[l] = mat.NewDense(256, fanOut, nil)
	}
	dW := make([]*mat.Dense, nT)
	for l := 0; l < nT; l++ {
		dW[l] = mat.NewDense(net.LayerSizes[l], net.LayerSizes[l+1], nil)
	}
	// dAl: buffer próprio por camada para evitar problema de stride de views
	dAl := make([]*mat.Dense, nT)
	for l := 0; l < nT; l++ {
		dAl[l] = mat.NewDense(256, net.LayerSizes[l], nil)
	}

	lossHistorico := make([]float64, 0, cfg.MaxEpocas)

	// epochStep proporcional: no máximo ~50 steps para nunca saturar o canal
	epochStep := cfg.MaxEpocas / 50
	if epochStep < 1 {
		epochStep = 1
	}

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		select {
		case <-ctx.Done():
			close(progressCh)
			return net
		default:
		}

		epochStart := time.Now()

		// === FORWARD PASS — usa matrizes persistentes net.Wm[l] ===
		for l := 0; l < nT; l++ {
			fanOut := net.LayerSizes[l+1]

			// Z = A_{l} * W_{l}  (BLAS dgemm via gonum)
			preActs[l+1].Mul(acts[l], net.Wm[l])

			// Z += bias (row-by-row, não há broadcast em gonum)
			b := net.Bvec[l]
			raw := preActs[l+1].RawMatrix().Data
			for row := 0; row < 256; row++ {
				off := row * fanOut
				for j := 0; j < fanOut; j++ {
					raw[off+j] += b[j]
				}
			}

			// Ativação element-wise no buffer acts[l+1]
			isOut := (l == nT-1)
			zRaw := preActs[l+1].RawMatrix().Data
			aRaw := acts[l+1].RawMatrix().Data
			if isOut {
				for k, v := range zRaw {
					aRaw[k] = sigmoid(v)
				}
			} else {
				for k, v := range zRaw {
					aRaw[k] = relu(v)
				}
			}
		}

		// === LOSS (MSE) ===
		outRaw := acts[nT].RawMatrix().Data
		tRaw := T.RawMatrix().Data
		var lossTotal float64
		for k, a := range outRaw {
			d := tRaw[k] - a
			lossTotal += d * d
		}
		lossMedia := lossTotal * 0.5 / float64(256*3)
		lossHistorico = append(lossHistorico, lossMedia)

		// === BACKWARD PASS ===
		// dZ_out = (T - A_out) * sigmoidDeriv(A_out)  — sem normalizar por N
		dzRaw := dZ[nT].RawMatrix().Data
		for k, a := range outRaw {
			dzRaw[k] = (tRaw[k] - a) * sigmoidDeriv(a)
		}

		// Propagate delta backwards
		for l := nT - 1; l >= 1; l-- {
			fanIn2 := net.LayerSizes[l]

			// dA_l = dZ_{l+1} * W_l^T  (BLAS dgemm)
			// Usa buffer dedicado (não view) para garantir stride contíguo
			dAl[l].Mul(dZ[l+1], net.Wm[l].T())

			// dZ_l = dA_l * reluDeriv(preActs[l]) element-wise
			dzlRaw := dZ[l].RawMatrix().Data
			daRaw := dAl[l].RawMatrix().Data
			zRaw := preActs[l].RawMatrix().Data
			for k := range dzlRaw {
				dzlRaw[k] = daRaw[k] * reluDeriv(zRaw[k])
			}
			_ = fanIn2
		}

		// === WEIGHT UPDATE ===
		// Batch GD: gradiente acumulado sem normalizar por N, lr aplicado direto.
		// Equivale ao SGD online da versão slice (que atualiza 256x com lr por sample).
		lr := cfg.LearningRate
		for l := 0; l < nT; l++ {
			fanOut3 := net.LayerSizes[l+1]

			// dW via BLAS: A_{l}^T * dZ_{l+1}
			dW[l].Mul(acts[l].T(), dZ[l+1])

			// Update Wm in-place
			wRaw := net.Wm[l].RawMatrix().Data
			dwRaw := dW[l].RawMatrix().Data
			for k := range wRaw {
				wRaw[k] += lr * dwRaw[k]
			}

			// dB = soma das linhas de dZ[l+1]
			dzColRaw := dZ[l+1].RawMatrix().Data
			b := net.Bvec[l]
			for j := 0; j < fanOut3; j++ {
				var sum float64
				for row := 0; row < 256; row++ {
					sum += dzColRaw[row*fanOut3+j]
				}
				b[j] += lr * sum
			}
		}

		epochMs := time.Since(epochStart).Milliseconds()

		if epoca%epochStep == 0 || epoca == 1 || epoca == cfg.MaxEpocas {
			// Sync gonum matrices back to slices for predict()
			net.syncFromMatrices()
			pixels := predict(net)
			step := Step{
				Epoca:        epoca,
				MaxEpocas:    cfg.MaxEpocas,
				Loss:         lossMedia,
				OutputPixels: pixels,
				ActiveLayer:  (epoca / epochStep) % len(net.W),
				ElapsedMs:    time.Since(start).Milliseconds(),
				EpochMs:      epochMs,
			}
			select {
			case progressCh <- step:
			default:
			}
		}
	}

	net.syncFromMatrices()
	finalPixels := predict(net)
	convergiu := lossHistorico[len(lossHistorico)-1] < 0.01
	select {
	case progressCh <- Step{
		Done:          true,
		Convergiu:     convergiu,
		LossHistorico: lossHistorico,
		Epoca:         cfg.MaxEpocas,
		MaxEpocas:     cfg.MaxEpocas,
		Loss:          lossHistorico[len(lossHistorico)-1],
		OutputPixels:  finalPixels,
		ActiveLayer:   -1,
		ElapsedMs:     time.Since(start).Milliseconds(),
	}:
	case <-ctx.Done():
	}
	close(progressCh)
	return net
}
