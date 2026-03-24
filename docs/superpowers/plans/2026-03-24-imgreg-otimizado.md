# Image Regression Otimizada Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar 3 backends otimizados (goroutines, matrix, mini-batch) para o treino de MLP image regression, mais um painel de benchmark comparativo, com accordion na sidebar e views idênticas ao painel Standard existente.

**Architecture:** Quatro novos packages Go independentes (`imgreg_goroutines`, `imgreg_matrix`, `imgreg_minibatch`, `imgreg_bench`) em `web/server/`, cada um com seus próprios tipos e função `Treinar`. O `main.go` recebe novas rotas e estado global para cada backend. O frontend (`index.html`) converte o item `IMG_REGRESSION` da sidebar em accordion com 5 filhos e adiciona 4 novas views.

**Tech Stack:** Go 1.24, `gonum.org/v1/gonum/mat` (para `imgreg_matrix`), SSE streaming, vanilla JS/HTML/CSS (sem frameworks).

---

## File Map

**Criar:**
- `web/server/imgreg_goroutines/imgreg_goroutines.go` — backend goroutines (batch GD paralelo)
- `web/server/imgreg_matrix/imgreg_matrix.go` — backend gonum/mat (forward/backward matricial)
- `web/server/imgreg_minibatch/imgreg_minibatch.go` — backend mini-batch com worker pool
- `web/server/imgreg_bench/imgreg_bench.go` — orquestrador benchmark (SSE multiplexado)

**Modificar:**
- `web/server/go.mod` — adicionar `gonum.org/v1/gonum`
- `web/server/main.go` — novos globals, handlers e rotas para os 4 backends
- `web/static/index.html` — sidebar accordion, 4 novas views, CSS novo, JS novo

**Não modificar:**
- `web/server/imgreg/imgreg.go` — backend original intacto

---

## Task 1: Adicionar gonum ao go.mod

**Files:**
- Modify: `web/server/go.mod`
- Modify: `web/server/go.sum` (gerado automaticamente)

- [ ] **Step 1: Rodar go get para adicionar gonum**

```bash
cd web/server
go get gonum.org/v1/gonum/mat
```

Expected: `go.mod` atualizado com `require gonum.org/v1/gonum v0.x.x`, `go.sum` gerado.

- [ ] **Step 2: Verificar que o módulo foi adicionado**

```bash
grep gonum web/server/go.mod
```

Expected: linha `gonum.org/v1/gonum v0.x.x`

- [ ] **Step 3: Commit**

```bash
git add web/server/go.mod web/server/go.sum
git commit -m "chore: add gonum dependency for matrix backend"
```

---

## Task 2: Backend `imgreg_goroutines`

**Files:**
- Create: `web/server/imgreg_goroutines/imgreg_goroutines.go`

Este backend é idêntico ao `imgreg` original em estrutura, mas paraleliza o forward+backward de cada pixel com goroutines e faz um único update de pesos ao final da época (batch GD).

- [ ] **Step 1: Criar o arquivo com tipos e funções de ativação**

Crie `web/server/imgreg_goroutines/imgreg_goroutines.go`:

```go
package imgreg_goroutines

import (
	"context"
	"math"
	"math/rand"
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
	LayerSizes []int       `json:"layerSizes"`
	W          [][][]float64 `json:"w"`
	B          [][]float64   `json:"b"`
}

type gradResult struct {
	gradW [][][]float64
	gradB [][]float64
	loss  float64
}

func relu(x float64) float64 {
	if x > 0 { return x }
	return 0
}

func reluDeriv(z float64) float64 {
	if z > 0 { return 1 }
	return 0
}

func sigmoid(x float64) float64 {
	if x < -500 { return 0 }
	if x > 500 { return 1 }
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDeriv(y float64) float64 { return y * (1 - y) }
```

- [ ] **Step 2: Adicionar `inicializar`, `forward`, `backward` e `atualizarPesos`**

Acrescente ao mesmo arquivo (copie a lógica do `imgreg` original, ajustando o package):

```go
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
		for i := 0; i < fanIn; i++ { gradW[l][i] = make([]float64, fanOut) }
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
```

- [ ] **Step 3: Adicionar `GetTarget` e `predict` (cópia do original)**

```go
func GetTarget(nome string) [][3]float64 {
	switch nome {
	case "smiley":  return imgSmiley()
	case "radial":  return imgRadial()
	case "brasil":  return imgBrasil()
	default:        return imgCoracao()
	}
}

// Copie as 4 funções imgCoracao, imgSmiley, imgRadial, imgBrasil
// de web/server/imgreg/imgreg.go sem modificação — apenas ajuste o package name.

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
```

- [ ] **Step 4: Adicionar a função `Treinar` com goroutines paralelas**

```go
func zeroGrads(net Net) ([][][]float64, [][]float64) {
	nT := len(net.W)
	gW := make([][][]float64, nT)
	gB := make([][]float64, nT)
	for l := 0; l < nT; l++ {
		fanIn, fanOut := len(net.W[l]), len(net.W[l][0])
		gW[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ { gW[l][i] = make([]float64, fanOut) }
		gB[l] = make([]float64, fanOut)
	}
	return gW, gB
}

func Treinar(ctx context.Context, cfg Config, progressCh chan<- Step) Net {
	rng := rand.New(rand.NewSource(42))
	if cfg.MaxEpocas <= 0 { cfg.MaxEpocas = 2000 }
	if cfg.HiddenLayers < 1 { cfg.HiddenLayers = 2 }
	if cfg.NeuronsPerLayer < 4 { cfg.NeuronsPerLayer = 16 }

	net := inicializar(cfg, rng)
	target := GetTarget(cfg.Imagem)

	indices := make([]int, 256)
	for i := range indices { indices[i] = i }

	lossHistorico := make([]float64, 0, cfg.MaxEpocas)
	const epochStep = 5
	start := time.Now()

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		select {
		case <-ctx.Done():
			close(progressCh)
			return net
		default:
		}

		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		epochStart := time.Now()

		// Canal buffered de tamanho 256: cada goroutine envia seu gradResult
		gradCh := make(chan gradResult, 256)
		var wg sync.WaitGroup

		// Captura snapshot dos pesos para leitura concorrente segura
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
				ActiveLayer:  epoca % len(net.W),
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
```

- [ ] **Step 5: Verificar que compila**

```bash
cd web/server && go build ./imgreg_goroutines/...
```

Expected: sem erros.

- [ ] **Step 6: Commit**

```bash
git add web/server/imgreg_goroutines/
git commit -m "feat: add imgreg_goroutines backend (parallel batch GD)"
```

---

## Task 3: Backend `imgreg_matrix`

**Files:**
- Create: `web/server/imgreg_matrix/imgreg_matrix.go`

Reescreve forward/backward usando `gonum/mat`. Processa todos os 256 pixels por época como operações matriciais.

- [ ] **Step 1: Criar arquivo com tipos, ativações e `Net`**

Crie `web/server/imgreg_matrix/imgreg_matrix.go`:

```go
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

// Net armazena pesos como slices Go (para serialização JSON e inicialização)
// mas os converte para gonum.Dense internamente no treino.
type Net struct {
	LayerSizes []int         `json:"layerSizes"`
	W          [][][]float64 `json:"w"`
	B          [][]float64   `json:"b"`
}

func relu(x float64) float64       { if x > 0 { return x }; return 0 }
func reluDeriv(z float64) float64  { if z > 0 { return 1 }; return 0 }
func sigmoid(x float64) float64 {
	if x < -500 { return 0 }
	if x > 500 { return 1 }
	return 1.0 / (1.0 + math.Exp(-x))
}
func sigmoidDeriv(y float64) float64 { return y * (1 - y) }
```

- [ ] **Step 2: Adicionar `inicializar`, `GetTarget` e funções de imagem**

Copie `inicializar` e as 4 funções de imagem de `imgreg_goroutines` (mesma lógica), ajustando o package para `imgreg_matrix`. Copie também `GetTarget` e `predict`.

- [ ] **Step 3: Adicionar `Treinar` com forward/backward matricial**

```go
func Treinar(ctx context.Context, cfg Config, progressCh chan<- Step) Net {
	rng := rand.New(rand.NewSource(42))
	if cfg.MaxEpocas <= 0 { cfg.MaxEpocas = 2000 }
	if cfg.HiddenLayers < 1 { cfg.HiddenLayers = 2 }
	if cfg.NeuronsPerLayer < 4 { cfg.NeuronsPerLayer = 16 }

	net := inicializar(cfg, rng)
	target := GetTarget(cfg.Imagem)
	const epochStep = 5
	start := time.Now()

	// Prepara a matrix de inputs: 256×2 (coordenadas normalizadas)
	// e a matrix de targets: 256×3 (RGB)
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
	A0 := mat.NewDense(256, 2, inputData) // input fixo
	T  := mat.NewDense(256, 3, targetData)

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
		// acts[l] = 256×sizes[l], preActs[l] = 256×sizes[l]
		acts := make([]*mat.Dense, nT+1)
		preActs := make([]*mat.Dense, nT+1)
		acts[0] = A0

		for l := 0; l < nT; l++ {
			fanIn, fanOut := len(net.W[l]), len(net.W[l][0])

			// Constrói matrix de pesos W_l: fanIn×fanOut
			wData := make([]float64, fanIn*fanOut)
			for i := 0; i < fanIn; i++ {
				for j := 0; j < fanOut; j++ {
					wData[i*fanOut+j] = net.W[l][i][j]
				}
			}
			Wl := mat.NewDense(fanIn, fanOut, wData)

			// Z = A * W  (256×fanIn × fanIn×fanOut = 256×fanOut)
			Z := mat.NewDense(256, fanOut, nil)
			Z.Mul(acts[l], Wl)

			// Adiciona bias linha a linha (gonum não faz broadcast)
			for row := 0; row < 256; row++ {
				for j := 0; j < fanOut; j++ {
					Z.Set(row, j, Z.At(row, j)+net.B[l][j])
				}
			}
			preActs[l+1] = Z

			// Aplica ativação elemento a elemento
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
		outA := acts[nT] // 256×3
		for row := 0; row < 256; row++ {
			for k := 0; k < 3; k++ {
				d := T.At(row, k) - outA.At(row, k)
				lossTotal += 0.5 * d * d
			}
		}
		lossMedia := lossTotal / float64(256*3)
		lossHistorico = append(lossHistorico, lossMedia)

		// === BACKWARD PASS MATRICIAL ===
		// dZ_out: 256×3 = (T - outA) * sigmoidDeriv(outA) elemento a elemento
		dZ := make([]*mat.Dense, nT+1)
		dZout := mat.NewDense(256, 3, nil)
		for row := 0; row < 256; row++ {
			for k := 0; k < 3; k++ {
				y := outA.At(row, k)
				dZout.Set(row, k, (T.At(row, k)-y)*sigmoidDeriv(y))
			}
		}
		dZ[nT] = dZout

		// Propaga delta pelas camadas ocultas de trás para frente
		for l := nT - 1; l >= 1; l-- {
			fanIn2, fanOut2 := len(net.W[l]), len(net.W[l][0])
			wData := make([]float64, fanIn2*fanOut2)
			for i := 0; i < fanIn2; i++ {
				for j := 0; j < fanOut2; j++ {
					wData[i*fanOut2+j] = net.W[l][i][j]
				}
			}
			Wl := mat.NewDense(fanIn2, fanOut2, wData)

			// dA_l = dZ_{l+1} * W_lᵀ  (256×fanOut2 × fanOut2×fanIn2 = 256×fanIn2)
			dAl := mat.NewDense(256, fanIn2, nil)
			dAl.Mul(dZ[l+1], Wl.T())

			// dZ_l = dA_l * reluDeriv(preActs[l])  elemento a elemento
			dZl := mat.NewDense(256, fanIn2, nil)
			for row := 0; row < 256; row++ {
				for j := 0; j < fanIn2; j++ {
					dZl.Set(row, j, dAl.At(row, j)*reluDeriv(preActs[l].At(row, j)))
				}
			}
			dZ[l] = dZl
		}

		// === UPDATE DE PESOS ===
		// dW_l = (A_{l-1}ᵀ * dZ_l) / 256  (média sobre os 256 pixels)
		for l := 0; l < nT; l++ {
			fanIn3, fanOut3 := len(net.W[l]), len(net.W[l][0])
			dWl := mat.NewDense(fanIn3, fanOut3, nil)
			dWl.Mul(acts[l].T(), dZ[l+1])

			for i := 0; i < fanIn3; i++ {
				for j := 0; j < fanOut3; j++ {
					net.W[l][i][j] += cfg.LearningRate * dWl.At(i, j) / 256.0
				}
			}
			// dB_l = média das linhas de dZ_{l+1}
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
				Epoca:        epoca,
				MaxEpocas:    cfg.MaxEpocas,
				Loss:         lossMedia,
				OutputPixels: pixels,
				ActiveLayer:  epoca % len(net.W),
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
```

- [ ] **Step 4: Verificar que compila**

```bash
cd web/server && go build ./imgreg_matrix/...
```

Expected: sem erros.

- [ ] **Step 5: Commit**

```bash
git add web/server/imgreg_matrix/
git commit -m "feat: add imgreg_matrix backend (gonum vectorized forward/backward)"
```

---

## Task 4: Backend `imgreg_minibatch`

**Files:**
- Create: `web/server/imgreg_minibatch/imgreg_minibatch.go`

Mini-batch com worker pool: os 256 pixels são divididos em batches, processados em paralelo por N workers.

- [ ] **Step 1: Criar arquivo com tipos (Config estendida)**

Crie `web/server/imgreg_minibatch/imgreg_minibatch.go`:

```go
package imgreg_minibatch

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
	BatchSize       int     `json:"batchSize"`
	NumWorkers      int     `json:"numWorkers"`
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

type batchJob struct {
	indices []int
	net     Net
	target  [][3]float64
}

type batchResult struct {
	gradW [][][]float64
	gradB [][]float64
	loss  float64
}
```

- [ ] **Step 2: Adicionar funções auxiliares (ativações, inicializar, forward, backward, atualizarPesos)**

Copie de `imgreg_goroutines` as funções: `relu`, `reluDeriv`, `sigmoid`, `sigmoidDeriv`, `inicializar`, `forward`, `backward`, `atualizarPesos`, `zeroGrads`, `GetTarget`, `predict`, e as 4 funções de imagem. Ajuste apenas o package name.

- [ ] **Step 3: Adicionar função `processBatch` e `Treinar`**

```go
func processBatch(job batchJob) batchResult {
	accumW, accumB := zeroGrads(job.net)
	var lossTotal float64
	for _, idx := range job.indices {
		py, px := idx/16, idx%16
		x := (float64(px)/15.0)*2.0 - 1.0
		y := (float64(py)/15.0)*2.0 - 1.0
		t := job.target[idx]
		acts, preActs := forward(job.net, x, y)
		out := acts[len(acts)-1]
		for k := 0; k < 3; k++ {
			d := t[k] - out[k]
			lossTotal += 0.5 * d * d
		}
		var tArr [3]float64
		tArr[0], tArr[1], tArr[2] = t[0], t[1], t[2]
		gW, gB := backward(job.net, acts, preActs, tArr)
		for l := range accumW {
			for i := range accumW[l] {
				for j := range accumW[l][i] {
					accumW[l][i][j] += gW[l][i][j]
				}
			}
			for j := range accumB[l] {
				accumB[l][j] += gB[l][j]
			}
		}
	}
	return batchResult{gradW: accumW, gradB: accumB, loss: lossTotal}
}

func Treinar(ctx context.Context, cfg Config, progressCh chan<- Step) Net {
	rng := rand.New(rand.NewSource(42))
	if cfg.MaxEpocas <= 0 { cfg.MaxEpocas = 2000 }
	if cfg.HiddenLayers < 1 { cfg.HiddenLayers = 2 }
	if cfg.NeuronsPerLayer < 4 { cfg.NeuronsPerLayer = 16 }
	if cfg.BatchSize <= 0 { cfg.BatchSize = 32 }
	if cfg.NumWorkers <= 0 { cfg.NumWorkers = runtime.NumCPU() }

	net := inicializar(cfg, rng)
	target := GetTarget(cfg.Imagem)

	indices := make([]int, 256)
	for i := range indices { indices[i] = i }

	lossHistorico := make([]float64, 0, cfg.MaxEpocas)
	const epochStep = 5
	start := time.Now()

	// Canal de jobs e results para o worker pool
	jobCh := make(chan batchJob, cfg.NumWorkers*2)
	resCh := make(chan batchResult, 256/cfg.BatchSize+1)

	// Inicia worker pool permanente
	var workerWg sync.WaitGroup
	for w := 0; w < cfg.NumWorkers; w++ {
		workerWg.Add(1)
		go func() {
			defer workerWg.Done()
			for job := range jobCh {
				resCh <- processBatch(job)
			}
		}()
	}

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		select {
		case <-ctx.Done():
			close(jobCh)
			// Drena resCh concorrentemente para não bloquear workers que já receberam jobs
			go func() {
				workerWg.Wait()
				close(resCh)
			}()
			for range resCh {} // esvazia para liberar os workers
			close(progressCh)
			return net
		default:
		}

		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		epochStart := time.Now()

		// Divide em batches e envia para os workers
		var numBatches int
		netSnap := net
		for start2 := 0; start2 < 256; start2 += cfg.BatchSize {
			end := start2 + cfg.BatchSize
			if end > 256 { end = 256 }
			batch := make([]int, end-start2)
			copy(batch, indices[start2:end])
			jobCh <- batchJob{indices: batch, net: netSnap, target: target}
			numBatches++
		}

		// Coleta resultados e acumula gradientes
		accumW, accumB := zeroGrads(net)
		var lossTotal float64
		for i := 0; i < numBatches; i++ {
			res := <-resCh
			lossTotal += res.loss
			for l := range accumW {
				for ii := range accumW[l] {
					for j := range accumW[l][ii] {
						accumW[l][ii][j] += res.gradW[l][ii][j]
					}
				}
				for j := range accumB[l] {
					accumB[l][j] += res.gradB[l][j]
				}
			}
		}

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
				ActiveLayer:  epoca % len(net.W),
				ElapsedMs:    time.Since(start).Milliseconds(),
				EpochMs:      epochMs,
			}
			select {
			case progressCh <- step:
			default:
			}
		}
	}

	close(jobCh)
	workerWg.Wait()

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
```

- [ ] **Step 4: Verificar que compila**

```bash
cd web/server && go build ./imgreg_minibatch/...
```

Expected: sem erros.

- [ ] **Step 5: Commit**

```bash
git add web/server/imgreg_minibatch/
git commit -m "feat: add imgreg_minibatch backend (worker pool mini-batch)"
```

---

## Task 5: Backend `imgreg_bench`

**Files:**
- Create: `web/server/imgreg_bench/imgreg_bench.go`

Orquestra os 3 backends em paralelo ou sequencial, multiplexando seus steps num único canal com tag `backend`.

- [ ] **Step 1: Criar arquivo com tipos**

Crie `web/server/imgreg_bench/imgreg_bench.go`:

```go
package imgreg_bench

import (
	"context"
	"sync"

	gor "mlp-server/imgreg_goroutines"
	mat "mlp-server/imgreg_matrix"
	mb  "mlp-server/imgreg_minibatch"
)

type BenchConfig struct {
	HiddenLayers    int     `json:"hiddenLayers"`
	NeuronsPerLayer int     `json:"neuronsPerLayer"`
	LearningRate    float64 `json:"learningRate"`
	Imagem          string  `json:"imagem"`
	MaxEpocas       int     `json:"maxEpocas"`
	BatchSize       int     `json:"batchSize"`
	NumWorkers      int     `json:"numWorkers"`
	Parallel        bool    `json:"parallel"`
}

// Step é uma cópia local do Step dos backends (sem import circular)
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

type BenchStep struct {
	Backend string `json:"backend"` // "goroutines" | "matrix" | "minibatch"
	Step    Step   `json:"step"`
}
```

- [ ] **Step 2: Adicionar funções de conversão de Step e `Rodar`**

```go
func fromGor(s gor.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}
func fromMat(s mat.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}
func fromMb(s mb.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}

func Rodar(ctx context.Context, cfg BenchConfig, ch chan<- BenchStep) {
	gorCfg := gor.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas}
	matCfg := mat.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas}
	mbCfg := mb.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas,
		BatchSize: cfg.BatchSize, NumWorkers: cfg.NumWorkers}

	runGor := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan gor.Step, 64)
		go gor.Treinar(ctx, gorCfg, c)
		for s := range c {
			out <- BenchStep{Backend: "goroutines", Step: fromGor(s)}
		}
	}
	runMat := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan mat.Step, 64)
		go mat.Treinar(ctx, matCfg, c)
		for s := range c {
			out <- BenchStep{Backend: "matrix", Step: fromMat(s)}
		}
	}
	runMb := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan mb.Step, 64)
		go mb.Treinar(ctx, mbCfg, c)
		for s := range c {
			out <- BenchStep{Backend: "minibatch", Step: fromMb(s)}
		}
	}

	if cfg.Parallel {
		// Todos os 3 backends em paralelo; canal intermediário para serializar
		proxy := make(chan BenchStep, 192)
		var wg sync.WaitGroup
		for _, fn := range []func(context.Context, chan<- BenchStep){runGor, runMat, runMb} {
			wg.Add(1)
			fn := fn
			go func() {
				defer wg.Done()
				fn(ctx, proxy)
			}()
		}
		go func() {
			wg.Wait()
			close(proxy)
		}()
		for bs := range proxy {
			ch <- bs
		}
	} else {
		// Sequencial: goroutines → matrix → minibatch
		runGor(ctx, ch)
		runMat(ctx, ch)
		runMb(ctx, ch)
	}
	close(ch)
}
```

- [ ] **Step 3: Verificar que compila**

```bash
cd web/server && go build ./imgreg_bench/...
```

Expected: sem erros.

- [ ] **Step 4: Commit**

```bash
git add web/server/imgreg_bench/
git commit -m "feat: add imgreg_bench orchestrator (parallel/sequential SSE mux)"
```

---

## Task 6: Rotas e estado global no `main.go`

**Files:**
- Modify: `web/server/main.go`

Adiciona imports, estado global e handlers para os 4 novos backends.

- [ ] **Step 1: Adicionar imports e estado global**

No topo do `main.go`, adicione nos imports:

```go
igoroutines "mlp-server/imgreg_goroutines"
imatrix     "mlp-server/imgreg_matrix"
iminibatch  "mlp-server/imgreg_minibatch"
ibench      "mlp-server/imgreg_bench"
```

Na seção de estado global (junto com os outros vars), adicione:

```go
// imgreg_goroutines
igorRede     *igoroutines.Net
igorTraining bool
igorCfg      *igoroutines.Config
igorCancel   context.CancelFunc

// imgreg_matrix
imatRede     *imatrix.Net
imatTraining bool
imatCfg      *imatrix.Config
imatCancel   context.CancelFunc

// imgreg_minibatch
imbRede      *iminibatch.Net
imbTraining  bool
imbCfg       *iminibatch.Config
imbCancel    context.CancelFunc

// imgreg_bench
benchCfg     *ibench.BenchConfig
benchRunning bool
benchCancel  context.CancelFunc
```

- [ ] **Step 2: Adicionar handlers para `imgreg_goroutines`**

Adicione ao `main.go` (copie o padrão dos handlers `handleImgreg*` existentes, substituindo tipos e prefixos):

```go
func handleIgorConfig(w http.ResponseWriter, r *http.Request) {
	var cfg igoroutines.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); igorCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleIgorTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if igorTraining { mu.Unlock(); errJSON(w, http.StatusConflict, "treino em andamento"); return }
	cfg := igorCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	igorTraining = true
	ctx, cancel := context.WithCancel(r.Context())
	igorCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); igorTraining = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	progressCh := make(chan igoroutines.Step, 64)
	go func() {
		defer cancel()
		rede := igoroutines.Treinar(ctx, *cfg, progressCh)
		mu.Lock(); igorRede = &rede; igorTraining = false; igorCancel = nil; mu.Unlock()
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleIgorReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if igorCancel != nil { igorCancel(); igorCancel = nil }
	igorRede = nil; igorCfg = nil; igorTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleIgorStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	training, trained := igorTraining, igorRede != nil
	mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]any{"training": training, "trained": trained})
}
```

- [ ] **Step 3: Adicionar handlers idênticos para `imgreg_matrix` e `imgreg_minibatch`**

Repita o padrão do Step 2 para os outros dois backends, substituindo:
- `igoroutines` → `imatrix` / `iminibatch`
- prefixo `Igor` → `Imat` / `Imb`
- variáveis globais correspondentes

Para `imgreg_minibatch`, o `Config` tem `BatchSize` e `NumWorkers` extras — o mesmo `json.Decode` os captura automaticamente.

- [ ] **Step 4: Adicionar handlers do benchmark**

```go
func handleBenchConfig(w http.ResponseWriter, r *http.Request) {
	var cfg ibench.BenchConfig
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); benchCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleBenchTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if benchRunning { mu.Unlock(); errJSON(w, http.StatusConflict, "benchmark em andamento"); return }
	cfg := benchCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	benchRunning = true
	ctx, cancel := context.WithCancel(r.Context())
	benchCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); benchRunning = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	benchCh := make(chan ibench.BenchStep, 192)
	go func() {
		defer cancel()
		ibench.Rodar(ctx, *cfg, benchCh)
		mu.Lock(); benchRunning = false; benchCancel = nil; mu.Unlock()
	}()

	for step := range benchCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleBenchReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if benchCancel != nil { benchCancel(); benchCancel = nil }
	benchCfg = nil; benchRunning = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}
```

- [ ] **Step 5: Registrar todas as rotas no `main()`**

Dentro de `func main()`, após as rotas `imgreg` existentes, adicione:

```go
// imgreg_goroutines
mux.HandleFunc("/api/imgreg-goroutines/config", cors(handleIgorConfig))
mux.HandleFunc("/api/imgreg-goroutines/train",  cors(handleIgorTrain))
mux.HandleFunc("/api/imgreg-goroutines/reset",  cors(handleIgorReset))
mux.HandleFunc("/api/imgreg-goroutines/status", cors(handleIgorStatus))

// imgreg_matrix
mux.HandleFunc("/api/imgreg-matrix/config", cors(handleImatConfig))
mux.HandleFunc("/api/imgreg-matrix/train",  cors(handleImatTrain))
mux.HandleFunc("/api/imgreg-matrix/reset",  cors(handleImatReset))
mux.HandleFunc("/api/imgreg-matrix/status", cors(handleImatStatus))

// imgreg_minibatch
mux.HandleFunc("/api/imgreg-minibatch/config", cors(handleImbConfig))
mux.HandleFunc("/api/imgreg-minibatch/train",  cors(handleImbTrain))
mux.HandleFunc("/api/imgreg-minibatch/reset",  cors(handleImbReset))
mux.HandleFunc("/api/imgreg-minibatch/status", cors(handleImbStatus))

// benchmark
mux.HandleFunc("/api/imgreg-bench/config", cors(handleBenchConfig))
mux.HandleFunc("/api/imgreg-bench/train",  cors(handleBenchTrain))
mux.HandleFunc("/api/imgreg-bench/reset",  cors(handleBenchReset))
```

- [ ] **Step 6: Verificar que compila**

```bash
cd web/server && go build .
```

Expected: sem erros.

- [ ] **Step 7: Commit**

```bash
git add web/server/main.go
git commit -m "feat: wire up imgreg_goroutines/matrix/minibatch/bench routes in main.go"
```

---

## Task 7: Frontend — CSS e sidebar accordion

**Files:**
- Modify: `web/static/index.html` (seção `<style>` e nav sidebar)

- [ ] **Step 1: Adicionar CSS para accordion e novas views**

Na seção `<style>` do `index.html`, após os estilos `.imgreg-*` existentes (por volta da linha 170), adicione:

```css
/* ── ACCORDION SIDEBAR ── */
.nav-accordion-header { display: flex; align-items: center; gap: 10px; padding: 9px 20px; cursor: pointer; font-family: var(--font-mono); font-size: 11px; color: var(--on-surface); transition: color 100ms, background 100ms; user-select: none; }
.nav-accordion-header:hover { color: var(--primary); background: var(--surface); }
.nav-accordion-header.open { color: var(--primary); }
.nav-accordion-arrow { margin-left: auto; font-size: 9px; transition: transform 150ms; }
.nav-accordion-header.open .nav-accordion-arrow { transform: rotate(90deg); }
.nav-accordion-children { display: none; padding-left: 12px; }
.nav-accordion-children.open { display: block; }
.nav-accordion-children .nav-item { font-size: 10px; padding: 7px 20px; }
.nav-accordion-children .nav-item.active::before { background: var(--cyan); }

/* ── TIMING STATS ── */
.imgreg-stat-bar .stat-val.time { color: var(--pink); }

/* ── BENCHMARK VIEW ── */
.bench-mode-toggle { display: flex; gap: 8px; }
.bench-mode-btn { font-family: var(--font-mono); font-size: 10px; padding: 6px 14px; background: var(--surface-high); color: var(--on-surface); border: none; cursor: pointer; }
.bench-mode-btn.active { background: var(--primary-glow); color: #013a00; }
.bench-cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.bench-col-header { font-family: var(--font-mono); font-size: 10px; color: var(--cyan); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; text-align: center; }
.bench-result-cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 16px; }
.bench-result-card { background: var(--surface); padding: 16px 20px; }
.bench-result-card .metric-label { font-family: var(--font-mono); font-size: 9px; color: var(--on-surface); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.bench-result-card .metric-val { font-family: var(--font-mono); font-size: 18px; font-weight: 700; color: var(--primary-glow); }
.bench-result-card.winner .metric-val { color: var(--cyan); }
```

- [ ] **Step 2: Substituir o item `IMG_REGRESSION` na sidebar pelo accordion**

Encontre no HTML (por volta da linha 295):

```html
    <div class="nav-item" data-view="imgreg">
      <svg ...>...</svg>
      IMG_REGRESSION
      <span class="nav-badge" id="badge-imgreg">OFF</span>
    </div>
```

Substitua por:

```html
    <!-- IMG_REGRESSION accordion -->
    <div class="nav-accordion-header" id="accordion-imgreg">
      <svg class="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="1" y="1" width="6" height="6"/><rect x="9" y="1" width="6" height="6"/><rect x="1" y="9" width="6" height="6"/><rect x="9" y="9" width="6" height="6"/><line x1="4" y1="4" x2="12" y2="4" stroke-dasharray="1,1"/><line x1="4" y1="12" x2="12" y2="12" stroke-dasharray="1,1"/></svg>
      IMG_REGRESSION
      <span class="nav-accordion-arrow">▶</span>
    </div>
    <div class="nav-accordion-children" id="accordion-imgreg-children">
      <div class="nav-item" data-view="imgreg">
        Standard
        <span class="nav-badge" id="badge-imgreg">OFF</span>
      </div>
      <div class="nav-item" data-view="imgreg-goroutines">
        Goroutines
        <span class="nav-badge" id="badge-igor">OFF</span>
      </div>
      <div class="nav-item" data-view="imgreg-matrix">
        Matrix
        <span class="nav-badge" id="badge-imat">OFF</span>
      </div>
      <div class="nav-item" data-view="imgreg-minibatch">
        Mini-batch
        <span class="nav-badge" id="badge-imb">OFF</span>
      </div>
      <div class="nav-item" data-view="imgreg-bench">
        Benchmark
        <span class="nav-badge" id="badge-bench">—</span>
      </div>
    </div>
```

- [ ] **Step 3: Adicionar JS para o accordion no final do `<script>`**

Antes do `</script>` final, adicione:

```js
// ── Accordion IMG_REGRESSION ──
const accordionHeader = document.getElementById('accordion-imgreg');
const accordionChildren = document.getElementById('accordion-imgreg-children');
accordionHeader.addEventListener('click', () => {
  const open = accordionChildren.classList.toggle('open');
  accordionHeader.classList.toggle('open', open);
});
// Abre o accordion automaticamente se uma view filha estiver ativa
function ensureAccordionOpen(viewId) {
  const childViews = ['imgreg','imgreg-goroutines','imgreg-matrix','imgreg-minibatch','imgreg-bench'];
  if (childViews.includes(viewId)) {
    accordionChildren.classList.add('open');
    accordionHeader.classList.add('open');
  }
}
```

Não existe uma função `showView` — a navegação é um `forEach` anônimo. Encontre o bloco que contém `querySelectorAll('[data-view]')` e `.classList.add('active')` (por volta da linha 990). O bloco se parece com:

```js
document.querySelectorAll('.nav-item[data-view]').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('view-' + item.dataset.view).classList.add('active');
    // ... outras linhas
  });
});
```

Adicione `ensureAccordionOpen(item.dataset.view);` logo após a linha `document.getElementById('view-' + item.dataset.view).classList.add('active');`. Exemplo:

```js
document.getElementById('view-' + item.dataset.view).classList.add('active');
ensureAccordionOpen(item.dataset.view); // ← adicione esta linha
```

Além disso, os filhos do accordion (`.nav-accordion-children .nav-item`) também precisam do mesmo event listener de navegação. Como eles têm `data-view`, o `querySelectorAll('.nav-item[data-view]')` já os captura — não é necessário handler separado.

- [ ] **Step 4: Verificar visualmente**

Abra `http://localhost:8080` no browser. Verifique:
- O item IMG_REGRESSION na sidebar agora é um accordion clicável
- Ao clicar, os 5 filhos aparecem/somem
- Clicar em "Standard" ainda abre a view imgreg existente

- [ ] **Step 5: Commit**

```bash
git add web/static/index.html
git commit -m "feat: convert IMG_REGRESSION sidebar item to accordion with 5 children"
```

---

## Task 8: Frontend — Views dos 3 backends otimizados

**Files:**
- Modify: `web/static/index.html` (adicionar 3 novas views após `view-imgreg`)

Cada view é uma cópia do `view-imgreg` com IDs renomeados e prefixo diferente. Goroutines usa prefixo `igor-`, Matrix usa `imat-`, Mini-batch usa `imb-`.

- [ ] **Step 1: Adicionar view `imgreg-goroutines`**

Após o fechamento de `</div>` do `view-imgreg` (por volta da linha 816), insira:

```html
<!-- ══════════════════════ VIEW: IMGREG GOROUTINES ══════════════════════ -->
<div class="view" id="view-imgreg-goroutines">
  <div class="page-header">
    <div>
      <div class="page-title">MLP <span>Image Regression</span> · Goroutines</div>
      <div class="page-sub">Batch GD paralelo · 256 goroutines/época · sem dependências externas</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <button class="btn btn-ghost" id="btn-igor-reset" style="padding:8px 16px;font-size:10px">RESET</button>
      <button class="btn btn-ghost" id="btn-igor-init" style="padding:8px 16px;font-size:10px">INICIALIZAR</button>
      <button class="btn btn-primary" id="btn-igor-train">
        <span class="spin" id="spin-igor" style="display:none"></span>
        TREINAR
      </button>
    </div>
  </div>

  <!-- Configuração (idêntica ao Standard) -->
  <div class="grid-3" style="margin-bottom:12px">
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">Imagem Alvo</div>
      <select class="imgreg-select" id="sel-igor-img">
        <option value="coracao">♥ Coração</option>
        <option value="smiley">☺ Smiley</option>
        <option value="radial">◎ Ondas Radiais</option>
        <option value="brasil">◈ Brasil</option>
      </select>
    </div>
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">Camadas Ocultas × Neurônios</div>
      <div style="display:flex;gap:8px">
        <select class="imgreg-select" id="sel-igor-layers" style="flex:1">
          <option value="2">2 camadas</option>
          <option value="3" selected>3 camadas</option>
          <option value="4">4 camadas</option>
          <option value="5">5 camadas</option>
        </select>
        <select class="imgreg-select" id="sel-igor-neurons" style="flex:1">
          <option value="16">16 neurônios</option>
          <option value="32" selected>32 neurônios</option>
          <option value="64">64 neurônios</option>
          <option value="128">128 neurônios</option>
        </select>
      </div>
    </div>
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">Learning Rate · Épocas</div>
      <div style="display:flex;gap:8px">
        <select class="imgreg-select" id="sel-igor-lr" style="flex:1">
          <option value="0.001">α 0.001</option>
          <option value="0.005">α 0.005</option>
          <option value="0.01" selected>α 0.01</option>
          <option value="0.02">α 0.02</option>
          <option value="0.05">α 0.05</option>
        </select>
        <select class="imgreg-select" id="sel-igor-epocas" style="flex:1">
          <option value="500">500</option>
          <option value="1000">1000</option>
          <option value="2000" selected>2000</option>
          <option value="5000">5000</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Stat bar com timing -->
  <div class="imgreg-stat-bar" id="igor-stat-bar">
    <span class="stat-key">ARCH</span><span class="stat-val" id="igor-arch">—</span>
    <span class="stat-sep">·</span>
    <span class="stat-key">ÉPOCA</span><span class="stat-val cyan" id="igor-epoca">0/0</span>
    <span class="stat-sep">·</span>
    <span class="stat-key">LOSS</span><span class="stat-val" id="igor-loss">—</span>
    <span class="stat-sep">·</span>
    <span class="stat-key">ms/ép</span><span class="stat-val time" id="igor-epochms">—</span>
    <span class="stat-sep">·</span>
    <span class="stat-key">px/s</span><span class="stat-val" id="igor-throughput">—</span>
    <div style="flex:1"></div>
    <div style="width:160px">
      <div class="imgreg-progress-bar"><div class="imgreg-progress-fill" id="igor-progress"></div></div>
    </div>
  </div>

  <!-- Visualização -->
  <div class="card" style="margin-bottom:16px;padding:24px">
    <div class="card-title">Visualização — Goroutines Paralelas</div>
    <div class="card-pulse"></div>
    <div class="imgreg-center-panel" id="igor-center-panel">
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div class="imgreg-label" style="color:var(--cyan)">TARGET · original</div>
        <div class="imgreg-canvas-wrap">
          <div class="imgreg-hud imgreg-hud-tl">INPUT_IMG</div>
          <div class="imgreg-grid" id="igor-target-grid"></div>
          <div class="imgreg-hud imgreg-hud-br">16×16 · 256px</div>
        </div>
      </div>
      <div style="flex:1;min-width:280px;display:flex;flex-direction:column;gap:8px">
        <div class="imgreg-label" style="color:var(--primary-glow)">NEURAL NETWORK · live</div>
        <div id="igor-net-svg" style="width:100%;"></div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <div class="imgreg-label" style="color:var(--primary-glow)">MLP OUTPUT · predição</div>
        <div class="imgreg-canvas-wrap">
          <div class="imgreg-hud imgreg-hud-tl">OUTPUT_MLP</div>
          <div class="imgreg-grid" id="igor-output-grid"></div>
          <div class="imgreg-hud imgreg-hud-br" id="igor-output-epoch">época 0</div>
        </div>
        <div class="imgreg-label" id="igor-output-loss" style="color:var(--on-surface)">LOSS: —</div>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Loss Curve</div>
      <div class="chart-wrap"><canvas id="igor-chart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Training Log</div>
      <div class="log-panel" id="igor-log"><div class="log-line dim">// aguardando...</div></div>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Adicionar view `imgreg-matrix`**

Copie a view `imgreg-goroutines` inteira, substituindo:
- `igor` → `imat`
- `Goroutines` → `Matrix`
- subtítulo: `Batch GD matricial · gonum/mat · forward/backward vetorizado`

- [ ] **Step 3: Adicionar view `imgreg-minibatch`**

Copie a view `imgreg-goroutines`, substituindo:
- `igor` → `imb`
- `Goroutines` → `Mini-batch`
- subtítulo: `Worker pool · mini-batches paralelos · batch size + workers configuráveis`

Adicione no card de configuração um 4º card para batch size e workers:

```html
<div class="card" style="padding:16px 20px">
  <div class="imgreg-select-label">Batch Size · Workers</div>
  <div style="display:flex;gap:8px">
    <select class="imgreg-select" id="sel-imb-batch" style="flex:1">
      <option value="8">8 px/batch</option>
      <option value="16">16 px/batch</option>
      <option value="32" selected>32 px/batch</option>
      <option value="64">64 px/batch</option>
    </select>
    <select class="imgreg-select" id="sel-imb-workers" style="flex:1">
      <option value="2">2 workers</option>
      <option value="4" selected>4 workers</option>
      <option value="8">8 workers</option>
      <option value="0">NumCPU</option>
    </select>
  </div>
</div>
```

- [ ] **Step 4: Verificar que as views aparecem no DOM**

No browser, usar devtools para confirmar que os 3 `div.view` foram adicionados com IDs corretos.

- [ ] **Step 5: Commit**

```bash
git add web/static/index.html
git commit -m "feat: add imgreg-goroutines, imgreg-matrix, imgreg-minibatch views"
```

---

## Task 9: Frontend — View Benchmark

**Files:**
- Modify: `web/static/index.html` (adicionar view `imgreg-bench`)

- [ ] **Step 1: Adicionar a view benchmark após as outras 3 views**

```html
<!-- ══════════════════════ VIEW: IMGREG BENCHMARK ══════════════════════ -->
<div class="view" id="view-imgreg-bench">
  <div class="page-header">
    <div>
      <div class="page-title">MLP Image Regression · <span>Benchmark</span></div>
      <div class="page-sub">Compara Goroutines vs Matrix vs Mini-batch · paralelo ou sequencial</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <div class="bench-mode-toggle">
        <button class="bench-mode-btn active" id="bench-btn-parallel">PARALELO</button>
        <button class="bench-mode-btn" id="bench-btn-seq">SEQUENCIAL</button>
      </div>
      <button class="btn btn-ghost" id="btn-bench-reset" style="padding:8px 16px;font-size:10px">RESET</button>
      <button class="btn btn-primary" id="btn-bench-run">
        <span class="spin" id="spin-bench" style="display:none"></span>
        RODAR BENCHMARK
      </button>
    </div>
  </div>

  <!-- Config compartilhada -->
  <div class="grid-3" style="margin-bottom:12px">
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">Imagem Alvo</div>
      <select class="imgreg-select" id="sel-bench-img">
        <option value="coracao">♥ Coração</option>
        <option value="smiley">☺ Smiley</option>
        <option value="radial">◎ Ondas Radiais</option>
        <option value="brasil">◈ Brasil</option>
      </select>
    </div>
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">Camadas × Neurônios · Épocas</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <select class="imgreg-select" id="sel-bench-layers" style="flex:1">
          <option value="2">2 cam.</option>
          <option value="3" selected>3 cam.</option>
          <option value="4">4 cam.</option>
        </select>
        <select class="imgreg-select" id="sel-bench-neurons" style="flex:1">
          <option value="16">16</option>
          <option value="32" selected>32</option>
          <option value="64">64</option>
        </select>
        <select class="imgreg-select" id="sel-bench-epocas" style="flex:1">
          <option value="200">200</option>
          <option value="500" selected>500</option>
          <option value="1000">1000</option>
        </select>
      </div>
    </div>
    <div class="card" style="padding:16px 20px">
      <div class="imgreg-select-label">LR · Batch Size · Workers</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <select class="imgreg-select" id="sel-bench-lr" style="flex:1">
          <option value="0.01" selected>α 0.01</option>
          <option value="0.02">α 0.02</option>
        </select>
        <select class="imgreg-select" id="sel-bench-batch" style="flex:1">
          <option value="32" selected>32px</option>
          <option value="64">64px</option>
        </select>
        <select class="imgreg-select" id="sel-bench-workers" style="flex:1">
          <option value="4" selected>4w</option>
          <option value="8">8w</option>
          <option value="0">CPU</option>
        </select>
      </div>
    </div>
  </div>

  <!-- 3 colunas de progresso -->
  <div class="bench-cols" id="bench-cols">
    <div class="card" style="padding:16px">
      <div class="bench-col-header">Goroutines</div>
      <div style="display:flex;justify-content:center;margin-bottom:8px">
        <div class="imgreg-grid" id="bench-grid-goroutines" style="opacity:0.4"></div>
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--on-surface);text-align:center" id="bench-stat-goroutines">aguardando...</div>
    </div>
    <div class="card" style="padding:16px">
      <div class="bench-col-header">Matrix</div>
      <div style="display:flex;justify-content:center;margin-bottom:8px">
        <div class="imgreg-grid" id="bench-grid-matrix" style="opacity:0.4"></div>
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--on-surface);text-align:center" id="bench-stat-matrix">aguardando...</div>
    </div>
    <div class="card" style="padding:16px">
      <div class="bench-col-header">Mini-batch</div>
      <div style="display:flex;justify-content:center;margin-bottom:8px">
        <div class="imgreg-grid" id="bench-grid-minibatch" style="opacity:0.4"></div>
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--on-surface);text-align:center" id="bench-stat-minibatch">aguardando...</div>
    </div>
  </div>

  <!-- Cards de resultado final -->
  <div class="bench-result-cards" id="bench-results" style="display:none">
    <div class="bench-result-card" id="bench-card-goroutines">
      <div class="bench-col-header">Goroutines</div>
      <div class="metric-label">Tempo Total</div>
      <div class="metric-val" id="bench-total-goroutines">—</div>
      <div class="metric-label" style="margin-top:8px">ms/época</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-msepoca-goroutines">—</div>
      <div class="metric-label" style="margin-top:8px">px/s</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-pxs-goroutines">—</div>
      <div class="metric-label" style="margin-top:8px">Loss Final</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--on-surface)" id="bench-loss-goroutines">—</div>
    </div>
    <div class="bench-result-card" id="bench-card-matrix">
      <div class="bench-col-header">Matrix</div>
      <div class="metric-label">Tempo Total</div>
      <div class="metric-val" id="bench-total-matrix">—</div>
      <div class="metric-label" style="margin-top:8px">ms/época</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-msepoca-matrix">—</div>
      <div class="metric-label" style="margin-top:8px">px/s</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-pxs-matrix">—</div>
      <div class="metric-label" style="margin-top:8px">Loss Final</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--on-surface)" id="bench-loss-matrix">—</div>
    </div>
    <div class="bench-result-card" id="bench-card-minibatch">
      <div class="bench-col-header">Mini-batch</div>
      <div class="metric-label">Tempo Total</div>
      <div class="metric-val" id="bench-total-minibatch">—</div>
      <div class="metric-label" style="margin-top:8px">ms/época</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-msepoca-minibatch">—</div>
      <div class="metric-label" style="margin-top:8px">px/s</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--primary)" id="bench-pxs-minibatch">—</div>
      <div class="metric-label" style="margin-top:8px">Loss Final</div>
      <div style="font-family:var(--font-mono);font-size:13px;color:var(--on-surface)" id="bench-loss-minibatch">—</div>
    </div>
  </div>

  <!-- Gráfico de barras comparativo -->
  <div class="card" style="margin-top:16px;display:none" id="bench-chart-card">
    <div class="card-title">Comparação — Tempo Total por Backend</div>
    <div class="chart-wrap"><canvas id="bench-chart"></canvas></div>
  </div>
</div>
```

- [ ] **Step 2: Inicializar os grids das 3 colunas**

Nos grids benchmark, cada `.imgreg-grid` precisa dos 256 pixels `div.imgreg-pixel`. No JS, adicione na seção de inicialização (junto com o `initImgregGrids` existente):

```js
function initBenchGrids() {
  ['bench-grid-goroutines','bench-grid-matrix','bench-grid-minibatch'].forEach(id => {
    const el = document.getElementById(id);
    el.innerHTML = '';
    for (let i = 0; i < 256; i++) {
      const px = document.createElement('div');
      px.className = 'imgreg-pixel';
      px.style.background = '#1c2026';
      el.appendChild(px);
    }
  });
}
initBenchGrids();
```

- [ ] **Step 3: Commit**

```bash
git add web/static/index.html
git commit -m "feat: add benchmark view HTML with 3-column layout and result cards"
```

---

## Task 10: Frontend — JS para os 3 backends otimizados

**Files:**
- Modify: `web/static/index.html` (seção `<script>`)

Cada backend tem a mesma lógica JS do Standard, com IDs e API path diferentes.

- [ ] **Step 1: Adicionar JS para `imgreg-goroutines`**

No `<script>`, após o bloco de JS do `imgreg` original, adicione:

```js
// ── IMGREG GOROUTINES ──
let igorES = null;
let igorLossHistory = [];
let igorInitialized = false;
let igorLayerSizes = [];

function getIgorCfg() {
  return {
    hiddenLayers:    parseInt($id('sel-igor-layers').value),
    neuronsPerLayer: parseInt($id('sel-igor-neurons').value),
    learningRate:    parseFloat($id('sel-igor-lr').value),
    imagem:          $id('sel-igor-img').value,
    maxEpocas:       parseInt($id('sel-igor-epocas').value),
  };
}

$id('btn-igor-init').addEventListener('click', async () => {
  if (igorES) { igorES.close(); igorES = null; }
  const cfg = getIgorCfg();
  igorLayerSizes = [2];
  for (let i = 0; i < cfg.hiddenLayers; i++) igorLayerSizes.push(cfg.neuronsPerLayer);
  igorLayerSizes.push(3);
  igorLossHistory = [];
  igorInitialized = true;
  // Carrega target
  const pixels = await fetch(`${API}/imgreg/target?img=${cfg.imagem}`).then(r => r.json());
  fillImgregGrid('igor-target-grid', pixels);
  // Inicializa output com ruído
  const noise = Array.from({length:256}, () => [Math.random()*0.3,Math.random()*0.3,Math.random()*0.3]);
  fillImgregGrid('igor-output-grid', noise);
  drawNetworkViz(igorLayerSizes, -1, 'igor-net-svg');
  addLogTo($id('igor-log'), `// rede inicializada: [${igorLayerSizes.join('→')}] · α=${cfg.learningRate}`, 'ok');
  $id('badge-igor').textContent = '';
});

$id('btn-igor-train').addEventListener('click', async () => {
  if (igorES) {
    igorES.close(); igorES = null;
    $id('btn-igor-train').textContent = 'TREINAR';
    $id('spin-igor').style.display = 'none';
    return;
  }
  if (!igorInitialized) {
    addLogTo($id('igor-log'), '// inicialize primeiro', 'err'); return;
  }
  const cfg = getIgorCfg();
  const log = $id('igor-log');
  await fetch(API + '/imgreg-goroutines/config', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(cfg)
  });
  $id('btn-igor-train').textContent = 'PAUSAR';
  $id('spin-igor').style.display = 'inline-block';
  addLogTo(log, `// iniciando treino goroutines · ${cfg.maxEpocas} épocas`, 'dim');

  igorES = new EventSource(API + '/imgreg-goroutines/train');
  igorES.onmessage = e => {
    const step = JSON.parse(e.data);
    if (step.done) {
      igorES.close(); igorES = null;
      fillImgregGrid('igor-output-grid', step.outputPixels);
      drawLogChart('igor-chart', step.lossHistorico, '#00ff00');
      igorLossHistory = step.lossHistorico;
      $id('igor-output-loss').textContent = `LOSS: ${step.loss.toFixed(6)}`;
      $id('igor-output-epoch').textContent = `época ${step.epoca}`;
      $id('btn-igor-train').textContent = 'TREINAR';
      $id('spin-igor').style.display = 'none';
      $id('badge-igor').textContent = 'ON';
      drawNetworkViz(igorLayerSizes, -1, 'igor-net-svg');
      addLogTo(log, `// concluído · loss=${step.loss.toFixed(6)} · ${step.convergiu?'CONVERGIU':'não convergiu'}`, step.convergiu?'ok':'warn');
      igorInitialized = false;
      return;
    }
    igorLossHistory.push(step.loss);
    fillImgregGrid('igor-output-grid', step.outputPixels);
    drawNetworkViz(igorLayerSizes, step.activeLayer, 'igor-net-svg');
    $id('igor-output-epoch').textContent = `época ${step.epoca}`;
    $id('igor-output-loss').textContent = `LOSS: ${step.loss.toFixed(6)}`;
    $id('igor-epoca').textContent = `${step.epoca}/${step.maxEpocas}`;
    $id('igor-loss').textContent = step.loss.toFixed(6);
    $id('igor-epochms').textContent = `${step.epochMs}ms`;
    const throughput = step.epochMs > 0 ? Math.round(256 / (step.epochMs / 1000)) : '—';
    $id('igor-throughput').textContent = `${throughput}`;
    $id('igor-progress').style.width = `${(step.epoca/step.maxEpocas*100).toFixed(1)}%`;
    if (igorLossHistory.length % 20 === 0) drawLogChart('igor-chart', igorLossHistory, '#00ff00');
  };
  igorES.onerror = () => {
    if (igorES) { igorES.close(); igorES = null; }
    $id('btn-igor-train').textContent = 'TREINAR';
    $id('spin-igor').style.display = 'none';
  };
});

$id('btn-igor-reset').addEventListener('click', async () => {
  if (igorES) { igorES.close(); igorES = null; }
  igorLossHistory = []; igorInitialized = false; igorLayerSizes = [];
  $id('badge-igor').textContent = 'OFF';
  $id('btn-igor-train').textContent = 'TREINAR';
  $id('spin-igor').style.display = 'none';
  $id('igor-log').innerHTML = '<div class="log-line dim">// resetado.</div>';
  try { await fetch(API + '/imgreg-goroutines/reset', { method: 'POST' }); } catch(_) {}
});
```

**IMPORTANTE — `drawNetworkViz` hardcoda `imgreg-net-svg`:** A função existente sempre renderiza no container `imgreg-net-svg`. Antes de adicionar os event listeners dos novos backends, modifique `drawNetworkViz` para aceitar um container ID opcional:

Encontre (por volta da linha 1815):
```js
function drawNetworkViz(layerSizes, activeLayer = -1) {
  const container = document.getElementById('imgreg-net-svg');
```

Substitua por:
```js
function drawNetworkViz(layerSizes, activeLayer = -1, containerId = 'imgreg-net-svg') {
  const container = document.getElementById(containerId);
```

Todas as chamadas existentes a `drawNetworkViz` continuam funcionando pois usam o default. As novas chamadas passam o containerId explicitamente:
```js
drawNetworkViz(igorLayerSizes, -1, 'igor-net-svg')
drawNetworkViz(imatLayerSizes, -1, 'imat-net-svg')
drawNetworkViz(imbLayerSizes,  -1, 'imb-net-svg')
```

- [ ] **Step 2: Adicionar JS idêntico para `imgreg-matrix`**

Copie o bloco do Step 1, substituindo:
- `igor` → `imat`, `IGOR` → `IMAT`
- `/imgreg-goroutines/` → `/imgreg-matrix/`
- título: `matrix`

- [ ] **Step 3: Adicionar JS para `imgreg-minibatch`**

Copie, substituindo:
- `igor` → `imb`, `IGOR` → `IMB`
- `/imgreg-goroutines/` → `/imgreg-minibatch/`
- Adicione `batchSize` e `numWorkers` na função `getImbCfg()`:
```js
function getImbCfg() {
  return {
    hiddenLayers:    parseInt($id('sel-imb-layers').value),
    neuronsPerLayer: parseInt($id('sel-imb-neurons').value),
    learningRate:    parseFloat($id('sel-imb-lr').value),
    imagem:          $id('sel-imb-img').value,
    maxEpocas:       parseInt($id('sel-imb-epocas').value),
    batchSize:       parseInt($id('sel-imb-batch').value),
    numWorkers:      parseInt($id('sel-imb-workers').value),
  };
}
```

- [ ] **Step 4: Verificar no browser**

Abra cada view, clique Inicializar, Treinar. Verifique:
- Grid target carrega corretamente
- Output atualiza durante o treino
- Stat bar mostra `ms/época` e `px/s`
- Gráfico de loss atualiza

- [ ] **Step 5: Commit**

```bash
git add web/static/index.html
git commit -m "feat: add JS controllers for goroutines, matrix, minibatch views"
```

---

## Task 11: Frontend — JS para o Benchmark

**Files:**
- Modify: `web/static/index.html` (seção `<script>`)

- [ ] **Step 1: Adicionar toggle paralelo/sequencial**

```js
// ── BENCHMARK MODE TOGGLE ──
let benchParallel = true;
$id('bench-btn-parallel').addEventListener('click', () => {
  benchParallel = true;
  $id('bench-btn-parallel').classList.add('active');
  $id('bench-btn-seq').classList.remove('active');
});
$id('bench-btn-seq').addEventListener('click', () => {
  benchParallel = false;
  $id('bench-btn-seq').classList.add('active');
  $id('bench-btn-parallel').classList.remove('active');
});
```

- [ ] **Step 2: Adicionar lógica principal do benchmark**

```js
// ── BENCHMARK RUN ──
let benchES = null;
const benchResults = {goroutines: null, matrix: null, minibatch: null};

function getBenchCfg() {
  return {
    hiddenLayers:    parseInt($id('sel-bench-layers').value),
    neuronsPerLayer: parseInt($id('sel-bench-neurons').value),
    learningRate:    parseFloat($id('sel-bench-lr').value),
    imagem:          $id('sel-bench-img').value,
    maxEpocas:       parseInt($id('sel-bench-epocas').value),
    batchSize:       parseInt($id('sel-bench-batch').value),
    numWorkers:      parseInt($id('sel-bench-workers').value),
    parallel:        benchParallel,
  };
}

$id('btn-bench-run').addEventListener('click', async () => {
  if (benchES) {
    benchES.close(); benchES = null;
    $id('btn-bench-run').textContent = 'RODAR BENCHMARK';
    $id('spin-bench').style.display = 'none';
    return;
  }

  // Reset visual
  ['goroutines','matrix','minibatch'].forEach(b => {
    $id(`bench-stat-${b}`).textContent = 'aguardando...';
    $id(`bench-grid-${b}`).style.opacity = '0.4';
    benchResults[b] = null;
  });
  $id('bench-results').style.display = 'none';
  $id('bench-chart-card').style.display = 'none';
  initBenchGrids();

  const cfg = getBenchCfg();
  await fetch(API + '/imgreg-bench/config', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(cfg)
  });

  $id('btn-bench-run').textContent = 'PARAR';
  $id('spin-bench').style.display = 'inline-block';

  benchES = new EventSource(API + '/imgreg-bench/train');
  benchES.onmessage = e => {
    const bs = JSON.parse(e.data); // BenchStep {backend, step}
    const {backend, step} = bs;
    const grid = $id(`bench-grid-${backend}`);
    const stat = $id(`bench-stat-${backend}`);

    if (step.done) {
      grid.style.opacity = '1';
      benchResults[backend] = step;
      fillImgregGrid(`bench-grid-${backend}`, step.outputPixels);
      stat.textContent = `✓ ${step.elapsedMs}ms · loss ${step.loss.toFixed(5)}`;
      // Verifica se todos terminaram
      if (benchResults.goroutines && benchResults.matrix && benchResults.minibatch) {
        benchES.close(); benchES = null;
        $id('btn-bench-run').textContent = 'RODAR BENCHMARK';
        $id('spin-bench').style.display = 'none';
        showBenchResults();
      }
      return;
    }

    fillImgregGrid(`bench-grid-${backend}`, step.outputPixels);
    grid.style.opacity = '1';
    const throughput = step.epochMs > 0 ? Math.round(256 / (step.epochMs / 1000)) : 0;
    stat.textContent = `ép ${step.epoca}/${step.maxEpocas} · loss ${step.loss.toFixed(5)} · ${step.epochMs}ms/ép · ${throughput}px/s`;
  };
  benchES.onerror = () => {
    if (benchES) { benchES.close(); benchES = null; }
    $id('btn-bench-run').textContent = 'RODAR BENCHMARK';
    $id('spin-bench').style.display = 'none';
  };
});

function showBenchResults() {
  const backends = ['goroutines','matrix','minibatch'];
  // Encontra o mais rápido (menor elapsedMs)
  let fastest = backends.reduce((a, b) =>
    benchResults[a].elapsedMs < benchResults[b].elapsedMs ? a : b
  );

  backends.forEach(b => {
    const r = benchResults[b];
    const msPerEpoca = (r.elapsedMs / r.maxEpocas).toFixed(1);
    const throughput = r.epochMs > 0 ? Math.round(256 / (r.epochMs / 1000)) : 0;
    $id(`bench-total-${b}`).textContent = `${r.elapsedMs}ms`;
    $id(`bench-msepoca-${b}`).textContent = `${msPerEpoca}ms`;
    $id(`bench-pxs-${b}`).textContent = `${throughput}`;
    $id(`bench-loss-${b}`).textContent = r.loss.toFixed(6);
    $id(`bench-card-${b}`).classList.toggle('winner', b === fastest);
  });

  $id('bench-results').style.display = 'grid';

  // Gráfico de barras
  $id('bench-chart-card').style.display = 'block';
  const ctx = document.getElementById('bench-chart').getContext('2d');
  if (window.benchChartInst) window.benchChartInst.destroy();
  window.benchChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Goroutines', 'Matrix', 'Mini-batch'],
      datasets: [{
        label: 'Tempo Total (ms)',
        data: backends.map(b => benchResults[b].elapsedMs),
        backgroundColor: ['#ff6ec7','#00fbfb','#00ff00'],
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { ticks: { color: '#b9ccaf', font: {family:'JetBrains Mono',size:10} } },
        x: { ticks: { color: '#b9ccaf', font: {family:'JetBrains Mono',size:10} } }
      }
    }
  });
}

$id('btn-bench-reset').addEventListener('click', async () => {
  if (benchES) { benchES.close(); benchES = null; }
  $id('btn-bench-run').textContent = 'RODAR BENCHMARK';
  $id('spin-bench').style.display = 'none';
  $id('bench-results').style.display = 'none';
  $id('bench-chart-card').style.display = 'none';
  ['goroutines','matrix','minibatch'].forEach(b => {
    $id(`bench-stat-${b}`).textContent = 'aguardando...';
    $id(`bench-grid-${b}`).style.opacity = '0.4';
    benchResults[b] = null;
  });
  initBenchGrids();
  try { await fetch(API + '/imgreg-bench/reset', { method: 'POST' }); } catch(_) {}
});
```

- [ ] **Step 3: Verificar no browser**

Rodar benchmark em modo paralelo e sequencial. Verificar:
- Grids animam nas 3 colunas
- Ao terminar, cards de resultado aparecem com o vencedor destacado
- Gráfico de barras renderiza corretamente

- [ ] **Step 4: Commit**

```bash
git add web/static/index.html
git commit -m "feat: add benchmark JS (parallel/sequential SSE, result cards, bar chart)"
```

---

## Task 12: Smoke test e verificação final

**Files:** nenhum arquivo novo

- [ ] **Step 1: Build completo**

```bash
cd web/server && go build .
```

Expected: sem erros.

- [ ] **Step 2: Rodar servidor e testar cada backend individualmente**

```bash
cd web/server && go run .
```

Abra `http://localhost:8080`. Para cada view (Standard, Goroutines, Matrix, Mini-batch):
1. Selecione imagem `smiley`, 2 camadas, 16 neurônios, α=0.01, 200 épocas
2. Clique INICIALIZAR → grid target deve aparecer
3. Clique TREINAR → progress bar deve mover, output grid deve atualizar a cada 5 épocas
4. Aguardar conclusão → badge deve mudar para `ON`

- [ ] **Step 3: Testar benchmark em modo paralelo**

Na view Benchmark:
1. Config: smiley, 2 camadas, 16 neurônios, 200 épocas, batch 32, 4 workers, PARALELO
2. Clique RODAR BENCHMARK
3. Verificar que as 3 colunas animam simultaneamente
4. Ao terminar: cards com `tempo total`, `ms/época`, `px/s`, `loss final` devem aparecer
5. Gráfico de barras deve renderizar

- [ ] **Step 4: Testar benchmark em modo sequencial**

Repita o Step 3 com toggle em SEQUENCIAL. As colunas devem animar uma por vez.

- [ ] **Step 5: Commit final**

```bash
git add -A
git commit -m "feat: complete imgreg optimized backends (goroutines, matrix, minibatch, benchmark)"
```
