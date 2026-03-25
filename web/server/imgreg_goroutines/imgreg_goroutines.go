package imgreg_goroutines

import (
	"context"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
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

type Net struct {
	LayerSizes []int         `json:"layerSizes"`
	W          [][][]float64 `json:"w"`
	B          [][]float64   `json:"b"`
}

type gradResult struct {
	gradW [][][]float64
	gradB [][]float64
	loss  float64
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDeriv(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

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
		fanIn, fanOut := sizes[l], sizes[l+1]
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

func forward(net Net, x, y float64) ([][]float64, [][]float64) {
	nLayers := len(net.LayerSizes)
	acts := make([][]float64, nLayers)
	preActs := make([][]float64, nLayers)
	acts[0] = []float64{x, y}
	preActs[0] = []float64{x, y}
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
		preActs[l+1] = z
		acts[l+1] = a
	}
	return acts, preActs
}

func backward(net Net, acts, preActs [][]float64, target [3]float64) ([][][]float64, [][]float64) {
	nT := len(net.W)
	gradW := make([][][]float64, nT)
	gradB := make([][]float64, nT)
	for l := 0; l < nT; l++ {
		fanIn, fanOut := len(net.W[l]), len(net.W[l][0])
		gradW[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			gradW[l][i] = make([]float64, fanOut)
		}
		gradB[l] = make([]float64, fanOut)
	}
	nLayers := len(acts)
	outLayer := acts[nLayers-1]
	deltas := make([][]float64, nLayers)
	dOut := make([]float64, len(outLayer))
	for k := 0; k < len(outLayer); k++ {
		dOut[k] = (target[k] - outLayer[k]) * sigmoidDeriv(outLayer[k])
	}
	deltas[nLayers-1] = dOut
	for l := nT - 1; l >= 1; l-- {
		fanOut := len(net.W[l])
		dHid := make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			var soma float64
			for k := 0; k < len(deltas[l+1]); k++ {
				soma += deltas[l+1][k] * net.W[l][j][k]
			}
			dHid[j] = soma * reluDeriv(preActs[l][j])
		}
		deltas[l] = dHid
	}
	for l := 0; l < nT; l++ {
		fanIn, fanOut := len(net.W[l]), len(net.W[l][0])
		for i := 0; i < fanIn; i++ {
			for j := 0; j < fanOut; j++ {
				gradW[l][i][j] = deltas[l+1][j] * acts[l][i]
			}
		}
		for j := 0; j < fanOut; j++ {
			gradB[l][j] = deltas[l+1][j]
		}
	}
	return gradW, gradB
}

func atualizarPesos(net Net, gradW [][][]float64, gradB [][]float64, lr float64) Net {
	for l := range net.W {
		for i := range net.W[l] {
			for j := range net.W[l][i] {
				net.W[l][i][j] += lr * gradW[l][i][j]
			}
		}
		for j := range net.B[l] {
			net.B[l][j] += lr * gradB[l][j]
		}
	}
	return net
}

// imgCoracao — coração vermelho com gradiente sobre fundo escuro
// Usa a equação do coração: (x²+y²-1)³ - x²y³ ≤ 0
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

func predict(net Net) [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0
			acts, _ := forward(net, x, y)
			out := acts[len(acts)-1]
			pixels[py*16+px] = [3]float64{out[0], out[1], out[2]}
		}
	}
	return pixels
}

func zeroGrads(net Net) ([][][]float64, [][]float64) {
	nT := len(net.W)
	gW := make([][][]float64, nT)
	gB := make([][]float64, nT)
	for l := 0; l < nT; l++ {
		fanIn, fanOut := len(net.W[l]), len(net.W[l][0])
		gW[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			gW[l][i] = make([]float64, fanOut)
		}
		gB[l] = make([]float64, fanOut)
	}
	return gW, gB
}

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

	indices := make([]int, 256)
	for i := range indices {
		indices[i] = i
	}

	lossHistorico := make([]float64, 0, cfg.MaxEpocas)
	epochStep := cfg.MaxEpocas / 50
	if epochStep < 1 {
		epochStep = 1
	}
	start := time.Now()

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		select {
		case <-ctx.Done():
			close(progressCh)
			return net
		default:
		}
		runtime.Gosched()

		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		epochStart := time.Now()

		// Canal buffered de tamanho 256: cada goroutine envia seu gradResult
		gradCh := make(chan gradResult, 256)
		var wg sync.WaitGroup

		// Captura snapshot dos pesos para leitura concorrente segura
		// netSnap is a shallow struct copy: netSnap.W shares the same backing arrays as net.W.
		// Goroutines may safely READ these concurrently because atualizarPesos only WRITES
		// after wg.Wait() has returned. Do not call atualizarPesos or modify net.W before wg.Wait().
		netSnap := net

		for _, idx := range indices {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				py, px := idx/16, idx%16
				x := (float64(px)/15.0)*2.0 - 1.0
				y := (float64(py)/15.0)*2.0 - 1.0
				t := target[idx]
				acts, preActs := forward(netSnap, x, y)
				out := acts[len(acts)-1]
				var l float64
				for k := 0; k < 3; k++ {
					d := t[k] - out[k]
					l += 0.5 * d * d
				}
				var tArr [3]float64
				tArr[0], tArr[1], tArr[2] = t[0], t[1], t[2]
				gW, gB := backward(netSnap, acts, preActs, tArr)
				gradCh <- gradResult{gradW: gW, gradB: gB, loss: l}
			}(idx)
		}

		// Aguarda todas as goroutines terminarem ANTES de ler o canal.
		// Assim o canal sempre tem exatamente 256 itens quando começamos a drenar.
		wg.Wait()
		close(gradCh)

		// Acumula gradientes na goroutine principal (sem mutex)
		accumW, accumB := zeroGrads(net)
		var lossTotal float64
		for gr := range gradCh {
			lossTotal += gr.loss
			for l := range accumW {
				for i := range accumW[l] {
					for j := range accumW[l][i] {
						accumW[l][i][j] += gr.gradW[l][i][j]
					}
				}
				for j := range accumB[l] {
					accumB[l][j] += gr.gradB[l][j]
				}
			}
		}

		// Update único ao final da época
		// NOTE: accumW/accumB contain the SUM of gradients for all 256 pixels (not the mean).
		// This means the effective weight update is: Δw = lr * Σ grad_i (batch GD with sum, not mean).
		// The learning rate semantics differ from the Standard SGD backend — use a smaller lr
		// (e.g., divide by 256) if you want numerically equivalent updates per epoch.
		net = atualizarPesos(net, accumW, accumB, cfg.LearningRate)

		epochMs := time.Since(epochStart).Milliseconds()
		lossMedia := lossTotal / float64(256*3)
		lossHistorico = append(lossHistorico, lossMedia)

		if epoca%epochStep == 0 || epoca == 1 || epoca == cfg.MaxEpocas {
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
