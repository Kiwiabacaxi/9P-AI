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

// Net stores weights as Go slices (for JSON serialization and initialization)
// but converts to gonum.Dense internally during training.
type Net struct {
	LayerSizes []int         `json:"layerSizes"`
	W          [][][]float64 `json:"-"`
	B          [][]float64   `json:"-"`
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

	return Net{LayerSizes: sizes, W: W, B: B}
}

// =============================================================================
// GERAÇÃO DE IMAGENS-ALVO (16×16 pixels, proceduralmente)
// =============================================================================

// GetTarget — retorna os 256 pixels RGB [0,1] da imagem escolhida
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

// imgCoracao — coração vermelho com gradiente sobre fundo escuro
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
				r = intensity
				g = 0.05
				b = 0.1
			} else {
				r = 0.04
				g = 0.05
				b = 0.08
			}
			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

// imgSmiley — rosto amarelo com olhos e sorriso em fundo azul escuro
func imgSmiley() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)

			dist := math.Sqrt(x*x + y*y)

			var r, g, b float64
			r, g, b = 0.05, 0.08, 0.25

			if dist < 0.85 {
				r, g, b = 0.95, 0.85, 0.05
			}

			olhoEsqX, olhoEsqY := -0.28, 0.28
			if math.Sqrt((x-olhoEsqX)*(x-olhoEsqX)+(y-olhoEsqY)*(y-olhoEsqY)) < 0.14 {
				r, g, b = 0.1, 0.07, 0.05
			}

			olhoDirX, olhoDirY := 0.28, 0.28
			if math.Sqrt((x-olhoDirX)*(x-olhoDirX)+(y-olhoDirY)*(y-olhoDirY)) < 0.14 {
				r, g, b = 0.1, 0.07, 0.05
			}

			sorrisoX, sorrisoY := 0.0, -0.1
			distSorriso := math.Sqrt((x-sorrisoX)*(x-sorrisoX) + (y-sorrisoY)*(y-sorrisoY))
			if distSorriso > 0.38 && distSorriso < 0.55 && y < sorrisoY+0.05 {
				r, g, b = 0.1, 0.07, 0.05
			}

			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

// imgRadial — padrão de ondas concêntricas coloridas (sin/cos)
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

			r := wave * angWave
			g := math.Sin(dist*6.0+1.0)*0.5 + 0.5
			b := math.Cos(dist*10.0)*0.5 + 0.5

			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

// imgBrasil — bandeira do Brasil simplificada (verde, losango amarelo, círculo azul)
func imgBrasil() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)

			var r, g, b float64

			r, g, b = 0.0, 0.5, 0.15

			if math.Abs(x)*0.9+math.Abs(y) < 0.65 {
				r, g, b = 0.95, 0.80, 0.0
			}

			dist := math.Sqrt(x*x + y*y)
			if dist < 0.35 {
				r, g, b = 0.0, 0.2, 0.7
			}

			if dist < 0.35 && math.Abs(y) < 0.06 {
				r, g, b = 0.95, 0.95, 0.95
			}

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
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0
			// predict does a single-sample forward pass using the slice-based Net
			// (not matrix ops, since it's just one sample)
			nLayers := len(net.LayerSizes)
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
// TREINAR — forward/backward matricial com gonum
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
	const epochStep = 5
	start := time.Now()

	// Prepara a matrix de inputs: 256x2 (coordenadas normalizadas)
	// e a matrix de targets: 256x3 (RGB)
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
	A0 := mat.NewDense(256, 2, inputData) // input fixed
	T := mat.NewDense(256, 3, targetData)

	nT := len(net.W)
	lossHistorico := make([]float64, 0, cfg.MaxEpocas)

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		select {
		case <-ctx.Done():
			close(progressCh)
			return net
		default:
		}

		epochStart := time.Now()

		// === FORWARD PASS MATRICIAL ===
		// acts[l] = 256xsizes[l], preActs[l] = 256xsizes[l]
		acts := make([]*mat.Dense, nT+1)
		preActs := make([]*mat.Dense, nT+1)
		acts[0] = A0
		preActs[0] = A0

		for l := 0; l < nT; l++ {
			fanIn, fanOut := len(net.W[l]), len(net.W[l][0])

			// Build weight matrix Wl: fanIn x fanOut
			wData := make([]float64, fanIn*fanOut)
			for i := 0; i < fanIn; i++ {
				for j := 0; j < fanOut; j++ {
					wData[i*fanOut+j] = net.W[l][i][j]
				}
			}
			Wl := mat.NewDense(fanIn, fanOut, wData)

			// Z = A * W  (256xfanIn x fanInxfanOut = 256xfanOut)
			Z := mat.NewDense(256, fanOut, nil)
			Z.Mul(acts[l], Wl)

			// Add bias row by row (gonum does not broadcast)
			for row := 0; row < 256; row++ {
				for j := 0; j < fanOut; j++ {
					Z.Set(row, j, Z.At(row, j)+net.B[l][j])
				}
			}
			preActs[l+1] = Z

			// Apply activation element-wise
			A := mat.NewDense(256, fanOut, nil)
			isOut := (l == nT-1)
			for row := 0; row < 256; row++ {
				for j := 0; j < fanOut; j++ {
					z := Z.At(row, j)
					if isOut {
						A.Set(row, j, sigmoid(z))
					} else {
						A.Set(row, j, relu(z))
					}
				}
			}
			acts[l+1] = A
		}

		// === LOSS (MSE) ===
		var lossTotal float64
		outA := acts[nT] // 256x3
		for row := 0; row < 256; row++ {
			for k := 0; k < 3; k++ {
				d := T.At(row, k) - outA.At(row, k)
				lossTotal += 0.5 * d * d
			}
		}
		lossMedia := lossTotal / float64(256*3)
		lossHistorico = append(lossHistorico, lossMedia)

		// === BACKWARD PASS MATRICIAL ===
		// dZ_out: 256x3 = (T - outA) * sigmoidDeriv(outA) element-wise
		dZ := make([]*mat.Dense, nT+1)
		dZout := mat.NewDense(256, 3, nil)
		for row := 0; row < 256; row++ {
			for k := 0; k < 3; k++ {
				y := outA.At(row, k)
				dZout.Set(row, k, (T.At(row, k)-y)*sigmoidDeriv(y))
			}
		}
		dZ[nT] = dZout

		// Propagate delta backwards through hidden layers
		for l := nT - 1; l >= 1; l-- {
			fanIn2, fanOut2 := len(net.W[l]), len(net.W[l][0])
			wData := make([]float64, fanIn2*fanOut2)
			for i := 0; i < fanIn2; i++ {
				for j := 0; j < fanOut2; j++ {
					wData[i*fanOut2+j] = net.W[l][i][j]
				}
			}
			Wl := mat.NewDense(fanIn2, fanOut2, wData)

			// dA_l = dZ_{l+1} * W_l^T  (256xfanOut2 x fanOut2xfanIn2 = 256xfanIn2)
			dAl := mat.NewDense(256, fanIn2, nil)
			dAl.Mul(dZ[l+1], Wl.T())

			// dZ_l = dA_l * reluDeriv(preActs[l]) element-wise
			dZl := mat.NewDense(256, fanIn2, nil)
			for row := 0; row < 256; row++ {
				for j := 0; j < fanIn2; j++ {
					dZl.Set(row, j, dAl.At(row, j)*reluDeriv(preActs[l].At(row, j)))
				}
			}
			dZ[l] = dZl
		}

		// === WEIGHT UPDATE ===
		// dW_l = (A_{l-1}^T * dZ_l) / 256  (mean over 256 pixels)
		for l := 0; l < nT; l++ {
			fanIn3, fanOut3 := len(net.W[l]), len(net.W[l][0])
			dWl := mat.NewDense(fanIn3, fanOut3, nil)
			dWl.Mul(acts[l].T(), dZ[l+1])

			for i := 0; i < fanIn3; i++ {
				for j := 0; j < fanOut3; j++ {
					net.W[l][i][j] += cfg.LearningRate * dWl.At(i, j) / 256.0
				}
			}
			// dB_l = mean of rows of dZ_{l+1}
			for j := 0; j < fanOut3; j++ {
				var sum float64
				for row := 0; row < 256; row++ {
					sum += dZ[l+1].At(row, j)
				}
				net.B[l][j] += cfg.LearningRate * sum / 256.0
			}
		}

		epochMs := time.Since(epochStart).Milliseconds()

		if epoca%epochStep == 0 || epoca == 1 || epoca == cfg.MaxEpocas {
			pixels := predict(net)
			step := Step{
				Epoca:       epoca,
				MaxEpocas:   cfg.MaxEpocas,
				Loss:        lossMedia,
				OutputPixels: pixels,
				ActiveLayer: epoca % len(net.W),
				ElapsedMs:   time.Since(start).Milliseconds(),
				EpochMs:     epochMs,
			}
			select {
			case progressCh <- step:
			default:
			}
		}
	}

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
