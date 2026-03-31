package cnn

// =============================================================================
// Operações de Convolução e Pooling — CNN (Aula 07)
//
// Implementação das 4 etapas do slide:
//   Etapa 1: Convolução — extração de features usando filtros/kernels
//   Etapa 2: Pooling — redução de dimensionalidade
//   Etapa 3: Flattening — conversão para vetor 1D
//   Etapa 4: Rede Densa — MLP para classificação
//
// Cada operação tem forward (propagação direta) e backward (backpropagation).
// =============================================================================

import "math"

// =============================================================================
// Convolução 2D — Forward
//
// Do slide: "O filtro percorre a imagem fazendo produto escalar"
//
// Para cada filtro f (canal de saída):
//   Para cada posição (oh, ow) na saída:
//     output[f][oh][ow] = bias[f] + Σ_c Σ_kh Σ_kw input[c][oh+kh][ow+kw] * filter[f][c][kh][kw]
//
// Depois aplica ReLU: max(0, x) — ativação padrão para CNNs (do slide)
//
// input:   [inChannels][inH][inW]
// filters: [outChannels][inChannels][kH][kW]
// output:  [outChannels][outH][outW]   onde outH = inH - kH + 1
// =============================================================================

// Conv2DForward aplica convolução + ReLU.
// Retorna a saída e o resultado pré-ReLU (para backward).
func Conv2DForward(
	input [][][]float64,
	filters [][][][]float64,
	biases []float64,
) (output, preRelu [][][]float64) {
	outCh := len(filters)
	inCh := len(input)
	inH := len(input[0])
	inW := len(input[0][0])
	kH := len(filters[0][0])
	kW := len(filters[0][0][0])
	outH := inH - kH + 1
	outW := inW - kW + 1

	output = make3D(outCh, outH, outW)
	preRelu = make3D(outCh, outH, outW)

	// Sim, são 6 loops aninhados mesmo, O(n^6), it's big brain time!
	// Para cada filtro f (canal de saída):
	for f := range outCh {
		// Para cada posição (oh, ow) na saída, h->Height, w->Width:
		for oh := range outH {
			for ow := range outW {
				sum := biases[f]
				// Para cada canal de entrada c:
				for c := range inCh {
					// Para cada posição (kh, kw) no filtro:
					for kh := range kH {
						for kw := range kW {
							// Produto escalar
							sum += input[c][oh+kh][ow+kw] * filters[f][c][kh][kw]
						}
					}
				}
				preRelu[f][oh][ow] = sum
				// ReLU: f(x) = max(0, x) — ativação padrão para CNNs
				if sum > 0 {
					output[f][oh][ow] = sum
				}
			}
		}
	}
	return
}

// =============================================================================
// Convolução 2D — Backward
//
// Backpropagation pela camada convolucional:
//
// 1. Gradiente dos biases:
//    dBias[f] = Σ_oh Σ_ow dOutput[f][oh][ow]
//
// 2. Gradiente dos filtros (para atualização de pesos):
//    dFilter[f][c][kh][kw] = Σ_oh Σ_ow dOutput[f][oh][ow] * input[c][oh+kh][ow+kw]
//
// 3. Gradiente da entrada (para propagar para camada anterior):
//    dInput[c][ih][iw] = Σ_f Σ_kh Σ_kw dOutput[f][ih-kh][iw-kw] * filter[f][c][kh][kw]
//    (equivale a convolução com filtro rotacionado 180°)
//
// ReLU backward: zero out gradient onde preRelu <= 0
// =============================================================================

func Conv2DBackward(
	input, preRelu, dOutput [][][]float64,
	filters [][][][]float64,
) (dInput [][][]float64, dFilters [][][][]float64, dBiases []float64) {
	outCh := len(filters)
	inCh := len(input)
	inH := len(input[0])
	inW := len(input[0][0])
	kH := len(filters[0][0])
	kW := len(filters[0][0][0])
	outH := len(dOutput[0])
	outW := len(dOutput[0][0])

	// Aplicar ReLU backward: zerar gradiente onde preRelu <= 0
	dAct := make3D(outCh, outH, outW)
	for f := range outCh {
		for oh := range outH {
			for ow := range outW {
				if preRelu[f][oh][ow] > 0 {
					dAct[f][oh][ow] = dOutput[f][oh][ow]
				}
			}
		}
	}

	dInput = make3D(inCh, inH, inW)
	dFilters = make4D(outCh, inCh, kH, kW)
	dBiases = make([]float64, outCh)

	for f := range outCh {
		for oh := range outH {
			for ow := range outW {
				d := dAct[f][oh][ow]
				// 1. Gradiente do bias
				dBiases[f] += d
				for c := range inCh {
					for kh := range kH {
						for kw := range kW {
							// 2. Gradiente do filtro
							dFilters[f][c][kh][kw] += d * input[c][oh+kh][ow+kw]
							// 3. Gradiente da entrada
							dInput[c][oh+kh][ow+kw] += d * filters[f][c][kh][kw]
						}
					}
				}
			}
		}
	}
	return
}

// =============================================================================
// Max Pooling 2D — Forward
//
// Do slide: "Reduz a dimensionalidade usando o valor máximo de cada região"
//
// Para cada região poolSize×poolSize:
//   output = max dos valores na região
//   Salva o índice do max para usar no backward
// =============================================================================

// PoolIndex armazena a posição do valor máximo no pool (para backward)
type PoolIndex struct {
	H, W int
}

func MaxPool2DForward(input [][][]float64, poolSize int) (output [][][]float64, indices [][]PoolIndex) {
	ch := len(input)
	inH := len(input[0])
	inW := len(input[0][0])
	outH := inH / poolSize
	outW := inW / poolSize

	output = make3D(ch, outH, outW)
	// indices[channel][oh*outW+ow] = posição do max
	indices = make([][]PoolIndex, ch)

	for c := range ch {
		indices[c] = make([]PoolIndex, outH*outW)
		for oh := range outH {
			for ow := range outW {
				maxVal := -math.MaxFloat64
				var maxH, maxW int
				for ph := range poolSize {
					for pw := range poolSize {
						ih := oh*poolSize + ph
						iw := ow*poolSize + pw
						if input[c][ih][iw] > maxVal {
							maxVal = input[c][ih][iw]
							maxH = ih
							maxW = iw
						}
					}
				}
				output[c][oh][ow] = maxVal
				indices[c][oh*outW+ow] = PoolIndex{maxH, maxW}
			}
		}
	}
	return
}

// =============================================================================
// Max Pooling 2D — Backward
//
// Gradiente flui APENAS para a posição do max (as outras recebem zero).
// =============================================================================

func MaxPool2DBackward(dOutput [][][]float64, indices [][]PoolIndex, inH, inW int) [][][]float64 {
	ch := len(dOutput)
	outH := len(dOutput[0])
	outW := len(dOutput[0][0])

	dInput := make3D(ch, inH, inW)
	for c := range ch {
		for oh := range outH {
			for ow := range outW {
				idx := indices[c][oh*outW+ow]
				dInput[c][idx.H][idx.W] += dOutput[c][oh][ow]
			}
		}
	}
	return dInput
}

// =============================================================================
// Flatten / Unflatten
//
// Do slide: "Converter a matriz 2D em um vetor 1D para alimentar a rede densa"
// =============================================================================

func Flatten(input [][][]float64) []float64 {
	ch := len(input)
	h := len(input[0])
	w := len(input[0][0])
	flat := make([]float64, ch*h*w)
	idx := 0
	for c := range ch {
		for i := range h {
			for j := range w {
				flat[idx] = input[c][i][j]
				idx++
			}
		}
	}
	return flat
}

func Unflatten(flat []float64, ch, h, w int) [][][]float64 {
	out := make3D(ch, h, w)
	idx := 0
	for c := range ch {
		for i := range h {
			for j := range w {
				out[c][i][j] = flat[idx]
				idx++
			}
		}
	}
	return out
}

// =============================================================================
// Dense (Fully Connected) Layer — Forward & Backward
//
// Mesma lógica do MLP, mas integrada no pipeline da CNN.
//
// Forward: y = W·x + b, depois ReLU (oculta) ou Softmax (saída)
// Backward: dW = dY · x^T, dx = W^T · dY, db = dY
// =============================================================================

// DenseForward calcula y = W·x + b (sem ativação — ativação aplicada separadamente)
func DenseForward(x []float64, W [][]float64, b []float64) []float64 {
	out := len(b)
	y := make([]float64, out)
	for j := range out {
		y[j] = b[j]
		for i := range x {
			y[j] += x[i] * W[i][j]
		}
	}
	return y
}

// DenseBackward calcula gradientes da camada densa.
// dY = gradiente vindo da camada seguinte
// Retorna: dX (para propagar), dW, dB (para atualizar pesos)
func DenseBackward(x, dY []float64, W [][]float64) (dX []float64, dW [][]float64, dB []float64) {
	inSize := len(x)
	outSize := len(dY)

	dX = make([]float64, inSize)
	dW = make([][]float64, inSize)
	dB = make([]float64, outSize)

	for i := range inSize {
		dW[i] = make([]float64, outSize)
		for j := range outSize {
			dW[i][j] = dY[j] * x[i]
			dX[i] += dY[j] * W[i][j]
		}
	}
	copy(dB, dY)
	return
}

// =============================================================================
// Ativações
// =============================================================================

// ReLU aplica max(0, x) in-place e retorna máscara para backward
func ReLU(x []float64) (out []float64, mask []bool) {
	out = make([]float64, len(x))
	mask = make([]bool, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
			mask[i] = true
		}
	}
	return
}

// ReLUBackward aplica a máscara ReLU ao gradiente
func ReLUBackward(dY []float64, mask []bool) []float64 {
	dX := make([]float64, len(dY))
	for i := range dY {
		if mask[i] {
			dX[i] = dY[i]
		}
	}
	return dX
}

// Softmax calcula a distribuição de probabilidade (numericamente estável)
// Do slide: "Função de ativação para camada de saída em classificação"
func Softmax(x []float64) []float64 {
	n := len(x)
	out := make([]float64, n)
	// Subtrair max para estabilidade numérica
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i, v := range x {
		out[i] = math.Exp(v - maxVal)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// CrossEntropyLoss calcula -log(p[target])
func CrossEntropyLoss(probs []float64, target int) float64 {
	p := probs[target]
	if p < 1e-12 {
		p = 1e-12
	}
	return -math.Log(p)
}

// SoftmaxCrossEntropyBackward calcula o gradiente combinado: dZ_i = probs_i - 1(i==target)
// Essa simplificação é elegante — o gradiente da softmax+cross-entropy é simplesmente probs - one_hot
func SoftmaxCrossEntropyBackward(probs []float64, target int) []float64 {
	dZ := make([]float64, len(probs))
	copy(dZ, probs)
	dZ[target] -= 1.0
	return dZ
}

// =============================================================================
// Helpers para alocação de tensores
// =============================================================================

func make3D(d, h, w int) [][][]float64 {
	t := make([][][]float64, d)
	for i := range d {
		t[i] = make([][]float64, h)
		for j := range h {
			t[i][j] = make([]float64, w)
		}
	}
	return t
}

func make4D(a, b, c, d int) [][][][]float64 {
	t := make([][][][]float64, a)
	for i := range a {
		t[i] = make([][][]float64, b)
		for j := range b {
			t[i][j] = make([][]float64, c)
			for k := range c {
				t[i][j][k] = make([]float64, d)
			}
		}
	}
	return t
}
