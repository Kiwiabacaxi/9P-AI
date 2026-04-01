package cnn

// =============================================================================
// CNN — Rede Neural Convolucional para classificação EMNIST Letters (Aula 07)
//
// Arquitetura (do slide):
//   Input: 28×28×1 (grayscale)
//   → Conv1: 8 filtros 3×3, ReLU            → 26×26×8
//   → MaxPool 2×2                            → 13×13×8
//   → Conv2: 16 filtros 3×3, ReLU           → 11×11×16
//   → MaxPool 2×2                            → 5×5×16
//   → Flatten                                → 400
//   → Dense1: 400→64, ReLU
//   → Dense2: 64→26, Softmax + Cross-Entropy
//
// 4 etapas do slide:
//   Etapa 1: Convolução (feature extraction)
//   Etapa 2: Pooling (redução de dimensionalidade)
//   Etapa 3: Flattening (converter para 1D)
//   Etapa 4: Rede Densa (classificação MLP)
//
// Dataset: EMNIST Letters (derivado do NIST SD19)
//   - 26 classes (A-Z), 28×28 grayscale
//   - ~88k treino, ~14k teste
// =============================================================================

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"time"
)

// =============================================================================
// Configuração e tipos
// =============================================================================

// Config permite customizar hiperparâmetros via frontend
type Config struct {
	Alfa       float64 `json:"alfa"`       // taxa de aprendizado (default 0.001)
	MaxEpocas  int     `json:"maxEpocas"`  // máximo de épocas (default 10)
	BatchSize  int     `json:"batchSize"`  // tamanho do mini-batch (default 32)
	TrainLimit int     `json:"trainLimit"` // limitar samples de treino (0 = all)
}

func DefaultConfig() Config {
	return Config{
		Alfa:       0.001,
		MaxEpocas:  10,
		BatchSize:  32,
		TrainLimit: 10000,
	}
}

// CnnStep é enviado via SSE durante o treinamento
type CnnStep struct {
	Epoca      int     `json:"epoca"`
	Batch      int     `json:"batch"`
	TotalBatch int     `json:"totalBatch"`
	Loss       float64 `json:"loss"`       // loss médio da época até agora
	Acuracia   float64 `json:"acuracia"`   // acurácia parcial
}

// CnnResult é o resultado final do treinamento
type CnnResult struct {
	Epocas        int       `json:"epocas"`
	LossFinal     float64   `json:"lossFinal"`
	LossHistorico []float64 `json:"lossHistorico"` // loss médio por época
	Acuracia      float64   `json:"acuracia"`       // acurácia no treino
	AcuraciaTest  float64   `json:"acuraciaTest"`   // acurácia no teste
	TempoMs       int64     `json:"tempoMs"`
}

type CnnCandidate struct {
	Letra string  `json:"letra"`
	Score float64 `json:"score"`
	Idx   int     `json:"idx"`
}

type ClassifyResp struct {
	LetraIdx int            `json:"letraIdx"`
	Letra    string         `json:"letra"`
	Scores   []float64      `json:"scores"` // probabilidades softmax [26]
	Top5     []CnnCandidate `json:"top5"`
}

// =============================================================================
// Estrutura da CNN
//
// Duas camadas convolucionais + duas camadas densas.
// Pesos armazenados como slices dinâmicos.
// =============================================================================

type CNN struct {
	// Conv1: 8 filtros de [1][3][3] (1 canal de entrada)
	Conv1F [][][][]float64 // [8][1][3][3]
	Conv1B []float64       // [8]
	// Conv2: 16 filtros de [8][3][3] (8 canais de entrada do conv1)
	Conv2F [][][][]float64 // [16][8][3][3]
	Conv2B []float64       // [16]
	// Dense1: 400 → 64
	W1 [][]float64 // [400][64]
	B1 []float64   // [64]
	// Dense2: 64 → 26
	W2 [][]float64 // [64][26]
	B2 []float64   // [26]
}

// forwardCache armazena todos os valores intermediários para o backward pass
type forwardCache struct {
	input      [][][]float64 // [1][28][28]
	conv1Pre   [][][]float64 // pré-ReLU do conv1
	conv1Out   [][][]float64 // [8][26][26]
	pool1Out   [][][]float64 // [8][13][13]
	pool1Idx   [][]PoolIndex
	conv2Pre   [][][]float64
	conv2Out   [][][]float64 // [16][11][11]
	pool2Out   [][][]float64 // [16][5][5]
	pool2Idx   [][]PoolIndex
	flat       []float64 // [400]
	dense1Raw  []float64 // pré-ReLU
	dense1Out  []float64 // [64]
	dense1Mask []bool    // máscara ReLU
	dense2Raw  []float64 // pré-softmax [26]
	probs      []float64 // softmax [26]
}

// =============================================================================
// Inicialização — He initialization (para ReLU)
//
// Do slide: "Para redes profundas, a inicialização dos pesos é crucial"
// He init: w ~ N(0, sqrt(2/fan_in)) — ideal para ReLU
// =============================================================================

func Inicializar() *CNN {
	rng := rand.New(rand.NewSource(42))
	net := &CNN{}

	// Conv1: 8 filtros [1][3][3], fan_in = 1*3*3 = 9
	net.Conv1F = make([][][][]float64, 8)
	net.Conv1B = make([]float64, 8)
	for f := range 8 {
		net.Conv1F[f] = make([][][]float64, 1)
		for c := range 1 {
			net.Conv1F[f][c] = make([][]float64, 3)
			for kh := range 3 {
				net.Conv1F[f][c][kh] = make([]float64, 3)
				for kw := range 3 {
					net.Conv1F[f][c][kh][kw] = rng.NormFloat64() * math.Sqrt(2.0/9.0)
				}
			}
		}
	}

	// Conv2: 16 filtros [8][3][3], fan_in = 8*3*3 = 72
	net.Conv2F = make([][][][]float64, 16)
	net.Conv2B = make([]float64, 16)
	for f := range 16 {
		net.Conv2F[f] = make([][][]float64, 8)
		for c := range 8 {
			net.Conv2F[f][c] = make([][]float64, 3)
			for kh := range 3 {
				net.Conv2F[f][c][kh] = make([]float64, 3)
				for kw := range 3 {
					net.Conv2F[f][c][kh][kw] = rng.NormFloat64() * math.Sqrt(2.0/72.0)
				}
			}
		}
	}

	// Dense1: 400 → 64, fan_in = 400
	net.W1 = make([][]float64, 400)
	net.B1 = make([]float64, 64)
	for i := range 400 {
		net.W1[i] = make([]float64, 64)
		for j := range 64 {
			net.W1[i][j] = rng.NormFloat64() * math.Sqrt(2.0/400.0)
		}
	}

	// Dense2: 64 → 26, fan_in = 64 (Xavier para softmax)
	net.W2 = make([][]float64, 64)
	net.B2 = make([]float64, NumClasses)
	for i := range 64 {
		net.W2[i] = make([]float64, NumClasses)
		for j := range NumClasses {
			net.W2[i][j] = rng.NormFloat64() * math.Sqrt(1.0/64.0)
		}
	}

	return net
}

// =============================================================================
// Forward pass completo
//
// Segue as 4 etapas do slide:
//   1. Convolução + ReLU (feature extraction)
//   2. Pooling (redução de dimensionalidade)
//   3. Flatten (converter para 1D)
//   4. Rede Densa + Softmax (classificação)
// =============================================================================

func (net *CNN) Forward(input [][][]float64) (probs []float64, cache forwardCache) {
	cache.input = input

	// Etapa 1a: Conv1 (28×28×1 → 26×26×8) + ReLU
	cache.conv1Out, cache.conv1Pre = Conv2DForward(input, net.Conv1F, net.Conv1B)

	// Etapa 2a: MaxPool (26×26×8 → 13×13×8)
	cache.pool1Out, cache.pool1Idx = MaxPool2DForward(cache.conv1Out, 2)

	// Etapa 1b: Conv2 (13×13×8 → 11×11×16) + ReLU
	cache.conv2Out, cache.conv2Pre = Conv2DForward(cache.pool1Out, net.Conv2F, net.Conv2B)

	// Etapa 2b: MaxPool (11×11×16 → 5×5×16)
	cache.pool2Out, cache.pool2Idx = MaxPool2DForward(cache.conv2Out, 2)

	// Etapa 3: Flatten (5×5×16 → 400)
	cache.flat = Flatten(cache.pool2Out)

	// Etapa 4a: Dense1 (400 → 64) + ReLU
	cache.dense1Raw = DenseForward(cache.flat, net.W1, net.B1)
	cache.dense1Out, cache.dense1Mask = ReLU(cache.dense1Raw)

	// Etapa 4b: Dense2 (64 → 26) + Softmax
	cache.dense2Raw = DenseForward(cache.dense1Out, net.W2, net.B2)
	cache.probs = Softmax(cache.dense2Raw)

	probs = cache.probs
	return
}

// =============================================================================
// Backward pass completo + atualização de pesos
//
// Backpropagation pela CNN inteira, de trás pra frente:
//   Dense2 → Dense1 → Flatten → Pool2 → Conv2 → Pool1 → Conv1
//
// Usa a regra da cadeia em cada camada para propagar o gradiente.
// =============================================================================

func (net *CNN) Backward(cache forwardCache, target int, alfa float64) {
	// Gradiente da Softmax + Cross-Entropy (simplificação elegante: probs - one_hot)
	dLogits := SoftmaxCrossEntropyBackward(cache.probs, target)

	// Backward Dense2 (64 → 26)
	dDense1Out, dW2, dB2 := DenseBackward(cache.dense1Out, dLogits, net.W2)

	// Backward ReLU da Dense1
	dDense1Raw := ReLUBackward(dDense1Out, cache.dense1Mask)

	// Backward Dense1 (400 → 64)
	dFlat, dW1, dB1 := DenseBackward(cache.flat, dDense1Raw, net.W1)

	// Backward Flatten → Unflatten para 3D (16×5×5)
	dPool2Out := Unflatten(dFlat, 16, 5, 5)

	// Backward MaxPool2 (5×5×16 → 11×11×16)
	dConv2Out := MaxPool2DBackward(dPool2Out, cache.pool2Idx, 11, 11)

	// Backward Conv2 (11×11×16 → 13×13×8)
	dPool1Out, dConv2F, dConv2B := Conv2DBackward(cache.pool1Out, cache.conv2Pre, dConv2Out, net.Conv2F)

	// Backward MaxPool1 (13×13×8 → 26×26×8)
	dConv1Out := MaxPool2DBackward(dPool1Out, cache.pool1Idx, 26, 26)

	// Backward Conv1 (26×26×8 → 28×28×1) — não precisamos de dInput
	_, dConv1F, dConv1B := Conv2DBackward(cache.input, cache.conv1Pre, dConv1Out, net.Conv1F)

	// Atualizar pesos (SGD: w -= alfa * gradient)
	// Conv1
	for f := range len(net.Conv1F) {
		net.Conv1B[f] -= alfa * dConv1B[f]
		for c := range len(net.Conv1F[f]) {
			for kh := range len(net.Conv1F[f][c]) {
				for kw := range len(net.Conv1F[f][c][kh]) {
					net.Conv1F[f][c][kh][kw] -= alfa * dConv1F[f][c][kh][kw]
				}
			}
		}
	}
	// Conv2
	for f := range len(net.Conv2F) {
		net.Conv2B[f] -= alfa * dConv2B[f]
		for c := range len(net.Conv2F[f]) {
			for kh := range len(net.Conv2F[f][c]) {
				for kw := range len(net.Conv2F[f][c][kh]) {
					net.Conv2F[f][c][kh][kw] -= alfa * dConv2F[f][c][kh][kw]
				}
			}
		}
	}
	// Dense1
	for i := range len(net.W1) {
		for j := range len(net.W1[i]) {
			net.W1[i][j] -= alfa * dW1[i][j]
		}
	}
	for j := range len(net.B1) {
		net.B1[j] -= alfa * dB1[j]
	}
	// Dense2
	for i := range len(net.W2) {
		for j := range len(net.W2[i]) {
			net.W2[i][j] -= alfa * dW2[i][j]
		}
	}
	for j := range len(net.B2) {
		net.B2[j] -= alfa * dB2[j]
	}
}

// =============================================================================
// Treinamento principal
//
// Algoritmo:
//   Para cada época:
//     Embaralhar dados de treino
//     Para cada mini-batch:
//       Para cada amostra no batch:
//         1. Converter imagem flat → tensor [1][28][28]
//         2. Forward pass → probabilidades softmax
//         3. Calcular loss (cross-entropy)
//         4. Backward pass → atualizar pesos
//       Reportar progresso via SSE
//     Avaliar acurácia no teste
//
// Streaming: envia CnnStep a cada 50 batches para não sobrecarregar.
// =============================================================================

func Treinar(ctx context.Context, cfg Config, data *EMNISTData, progressCh chan<- CnnStep) (*CNN, CnnResult) {
	if cfg.Alfa <= 0 {
		cfg.Alfa = 0.001
	}
	if cfg.MaxEpocas <= 0 {
		cfg.MaxEpocas = 10
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 32
	}

	start := time.Now()
	rng := rand.New(rand.NewSource(42))
	net := Inicializar()

	// Limitar dados de treino se configurado
	trainImages := data.TrainImages
	trainLabels := data.TrainLabels
	if cfg.TrainLimit > 0 && cfg.TrainLimit < len(trainImages) {
		trainImages = trainImages[:cfg.TrainLimit]
		trainLabels = trainLabels[:cfg.TrainLimit]
	}

	nTrain := len(trainImages)
	var res CnnResult

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		// Verificar cancelamento
		select {
		case <-ctx.Done():
			res.Epocas = epoca - 1
			res.TempoMs = time.Since(start).Milliseconds()
			return net, res
		default:
		}

		// Embaralhar dados
		ShuffleData(rng, trainImages, trainLabels)

		epochLoss := 0.0
		correct := 0
		totalBatches := (nTrain + cfg.BatchSize - 1) / cfg.BatchSize

		for batch := 0; batch < totalBatches; batch++ {
			// Verificar cancelamento
			select {
			case <-ctx.Done():
				res.Epocas = epoca
				res.TempoMs = time.Since(start).Milliseconds()
				return net, res
			default:
			}

			bStart := batch * cfg.BatchSize
			bEnd := bStart + cfg.BatchSize
			if bEnd > nTrain {
				bEnd = nTrain
			}

			// Processar cada amostra do batch
			for i := bStart; i < bEnd; i++ {
				input := ImageToTensor(trainImages[i])
				probs, cache := net.Forward(input)

				label := trainLabels[i]
				epochLoss += CrossEntropyLoss(probs, label)

				// Verificar predição
				pred := argmax(probs)
				if pred == label {
					correct++
				}

				// Backward + update
				net.Backward(cache, label, cfg.Alfa)
			}

			// Enviar progresso a cada 50 batches
			if progressCh != nil && (batch%50 == 0 || batch == totalBatches-1) {
				processed := bEnd
				step := CnnStep{
					Epoca:      epoca,
					Batch:      batch + 1,
					TotalBatch: totalBatches,
					Loss:       epochLoss / float64(processed),
					Acuracia:   float64(correct) / float64(processed) * 100,
				}
				select {
				case progressCh <- step:
				default:
				}
			}
		}

		avgLoss := epochLoss / float64(nTrain)
		res.LossHistorico = append(res.LossHistorico, avgLoss)
	}

	// Resultado final
	res.Epocas = cfg.MaxEpocas
	res.LossFinal = res.LossHistorico[len(res.LossHistorico)-1]
	res.Acuracia = avaliarAcuracia(net, trainImages, trainLabels, 2000)
	res.AcuraciaTest = avaliarAcuracia(net, data.TestImages, data.TestLabels, 2000)
	res.TempoMs = time.Since(start).Milliseconds()

	return net, res
}

// avaliarAcuracia calcula a acurácia em um subset dos dados
func avaliarAcuracia(net *CNN, images [][]float64, labels []int, maxSamples int) float64 {
	n := len(images)
	if maxSamples > 0 && maxSamples < n {
		n = maxSamples
	}
	correct := 0
	for i := range n {
		input := ImageToTensor(images[i])
		probs, _ := net.Forward(input)
		if argmax(probs) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(n) * 100
}

// =============================================================================
// Classificação de uma imagem
// =============================================================================

func Classificar(net *CNN, pixels []float64) ClassifyResp {
	input := ImageToTensor(pixels)
	probs, _ := net.Forward(input)

	best := argmax(probs)

	// Top-5 por maior score
	type scoreIdx struct {
		score float64
		idx   int
	}
	items := make([]scoreIdx, NumClasses)
	for i := range NumClasses {
		items[i] = scoreIdx{probs[i], i}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].score > items[b].score
	})

	top5 := make([]CnnCandidate, 5)
	for i := range 5 {
		top5[i] = CnnCandidate{
			Letra: LetterNames[items[i].idx],
			Score: items[i].score,
			Idx:   items[i].idx,
		}
	}

	return ClassifyResp{
		LetraIdx: best,
		Letra:    LetterNames[best],
		Scores:   probs,
		Top5:     top5,
	}
}

func argmax(x []float64) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// =============================================================================
// Visualização das etapas da CNN
//
// Faz um forward pass e retorna TODAS as ativações intermediárias
// para visualizar o pipeline: Input → Conv1 → Pool1 → Conv2 → Pool2 → resultado
// =============================================================================

// VisualizeResp contém todas as ativações intermediárias para visualização
type VisualizeResp struct {
	Input     []float64      `json:"input"`     // [784] imagem flat
	Conv1Maps [][][]float64  `json:"conv1Maps"` // [8][26][26] feature maps após Conv1+ReLU
	Pool1Maps [][][]float64  `json:"pool1Maps"` // [8][13][13] após MaxPool
	Conv2Maps [][][]float64  `json:"conv2Maps"` // [16][11][11] feature maps após Conv2+ReLU
	Pool2Maps [][][]float64  `json:"pool2Maps"` // [16][5][5] após MaxPool
	Filters1  [][][][]float64 `json:"filters1"` // [8][1][3][3] kernels do Conv1
	Filters2  [][][][]float64 `json:"filters2"` // [16][8][3][3] kernels do Conv2
	Probs     []float64      `json:"probs"`     // [26] softmax final
	LetraIdx  int            `json:"letraIdx"`
	Letra     string         `json:"letra"`
	Top5      []CnnCandidate `json:"top5"`
}

// Visualizar faz forward pass e extrai todas as ativações para visualização.
func Visualizar(net *CNN, pixels []float64) VisualizeResp {
	input := ImageToTensor(pixels)
	probs, cache := net.Forward(input)

	best := argmax(probs)

	// Top-5
	type scoreIdx struct {
		score float64
		idx   int
	}
	items := make([]scoreIdx, NumClasses)
	for i := range NumClasses {
		items[i] = scoreIdx{probs[i], i}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].score > items[b].score
	})
	top5 := make([]CnnCandidate, 5)
	for i := range 5 {
		top5[i] = CnnCandidate{
			Letra: LetterNames[items[i].idx],
			Score: items[i].score,
			Idx:   items[i].idx,
		}
	}

	return VisualizeResp{
		Input:     pixels,
		Conv1Maps: cache.conv1Out,
		Pool1Maps: cache.pool1Out,
		Conv2Maps: cache.conv2Out,
		Pool2Maps: cache.pool2Out,
		Filters1:  net.Conv1F,
		Filters2:  net.Conv2F,
		Probs:     probs,
		LetraIdx:  best,
		Letra:     LetterNames[best],
		Top5:      top5,
	}
}
