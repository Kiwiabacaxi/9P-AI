package imgreg

import (
	"context"
	"math"
	"math/rand"
	"runtime"
)

// =============================================================================
// IMAGE REGRESSION com MLP — Desafio 3
// A rede aprende a mapear coordenadas (x,y) → (R,G,B),
// reconstruindo uma imagem pixel a pixel.
// Demonstra o Teorema da Aproximação Universal:
// um MLP com camadas suficientes pode aproximar qualquer função contínua.
// =============================================================================

// Config — parâmetros configuráveis pelo usuário
type Config struct {
	HiddenLayers    int     `json:"hiddenLayers"`    // número de camadas ocultas: 2, 3, 4, 5
	NeuronsPerLayer int     `json:"neuronsPerLayer"` // neurônios por camada: 16, 32, 64, 128
	LearningRate    float64 `json:"learningRate"`    // taxa de aprendizado: 0.001..0.05
	Imagem          string  `json:"imagem"`          // "coracao", "smiley", "radial", "brasil"
	MaxEpocas       int     `json:"maxEpocas"`       // épocas máximas (default 2000)
}

// Net — a rede MLP com arquitetura variável
// W[l][i][j] = peso da camada l, do neurônio i (entrada) para o neurônio j (saída)
// B[l][j]    = bias do neurônio j na camada l
type Net struct {
	// Arquitetura armazenada para rebuild
	LayerSizes []int `json:"layerSizes"` // ex: [2, 32, 32, 3]

	// Pesos e biases por camada de transição (len = len(LayerSizes)-1)
	W [][][]float64 `json:"w"` // W[camada][de][para]
	B [][]float64   `json:"b"` // B[camada][neurônio]
}

// Step — mensagem enviada via SSE durante e ao fim do treino.
// Quando Done=true, os campos de resultado final estão preenchidos.
// Um único tipo de mensagem evita race condition entre canal e goroutine.
type Step struct {
	// Campos de progresso (enviados a cada N épocas)
	Epoca        int          `json:"epoca"`
	MaxEpocas    int          `json:"maxEpocas"`
	Loss         float64      `json:"loss"`
	OutputPixels [][3]float64 `json:"outputPixels"`
	ActiveLayer  int          `json:"activeLayer"`

	// Campos de conclusão (apenas quando Done=true)
	Done          bool      `json:"done"`
	Convergiu     bool      `json:"convergiu"`
	LossHistorico []float64 `json:"lossHistorico"`
}

// =============================================================================
// FUNÇÕES DE ATIVAÇÃO
// =============================================================================

// relu — Rectified Linear Unit: f(x) = max(0, x)
// Cria não-linearidade sem "saturar" como tanh/sigmoid nas camadas ocultas.
// Derivada: 1 se x > 0, 0 caso contrário ("morte do gradiente" em ReLU negativo)
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDeriv — derivada de ReLU em relação à PRÉ-ativação z
// f'(z) = 1 se z > 0, 0 caso contrário
func reluDeriv(z float64) float64 {
	if z > 0 {
		return 1
	}
	return 0
}

// sigmoid — função logística: f(x) = 1 / (1 + e^-x)
// Comprime saída para [0,1] — ideal para RGB normalizado
// Derivada: f(x) * (1 - f(x)) — calculada a partir do valor pós-ativação
func sigmoid(x float64) float64 {
	// Clamp para evitar overflow numérico em e^-x com x muito negativo
	if x < -500 {
		return 0
	}
	if x > 500 {
		return 1
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDeriv — derivada de sigmoid em relação ao valor pós-ativação y = sigmoid(z)
// f'(z) = y * (1 - y)
func sigmoidDeriv(y float64) float64 {
	return y * (1 - y)
}

// =============================================================================
// INICIALIZAÇÃO DA REDE
// =============================================================================

// inicializar — cria uma nova rede com arquitetura [2, N, N, ..., 3]
// e inicializa pesos com He Initialization: W ~ N(0, sqrt(2/fan_in))
// He initialization é ideal para ReLU: mantém variância dos gradientes ao longo das camadas
func inicializar(cfg Config, rng *rand.Rand) Net {
	// Monta o vetor de tamanhos das camadas
	// Entrada: 2 (coordenadas x, y normalizadas)
	// Ocultas: cfg.HiddenLayers camadas de cfg.NeuronsPerLayer neurônios cada
	// Saída: 3 (canais R, G, B normalizados para [0,1])
	sizes := make([]int, 0, cfg.HiddenLayers+2)
	sizes = append(sizes, 2) // camada de entrada
	for i := 0; i < cfg.HiddenLayers; i++ {
		sizes = append(sizes, cfg.NeuronsPerLayer)
	}
	sizes = append(sizes, 3) // camada de saída

	nLayers := len(sizes) - 1 // número de transições (matrizes de peso)

	W := make([][][]float64, nLayers)
	B := make([][]float64, nLayers)

	for l := 0; l < nLayers; l++ {
		fanIn := sizes[l]    // neurônios na camada de entrada desta transição
		fanOut := sizes[l+1] // neurônios na camada de saída desta transição

		// Escala He: sqrt(2 / fan_in) — otimizado para ReLU
		// Garante que a variância dos gradientes se mantenha ao propagar para trás
		scale := math.Sqrt(2.0 / float64(fanIn))

		W[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			W[l][i] = make([]float64, fanOut)
			for j := 0; j < fanOut; j++ {
				// Amostrar da distribuição normal e escalar
				W[l][i][j] = rng.NormFloat64() * scale
			}
		}

		// Biases inicializados com zero (prática padrão com He init)
		B[l] = make([]float64, fanOut)
	}

	return Net{LayerSizes: sizes, W: W, B: B}
}

// =============================================================================
// FORWARD PASS
// =============================================================================

// imgregForward — propaga uma entrada (x, y) pela rede, camada a camada
// Retorna:
//   - activations: valor pós-ativação de cada neurônio em cada camada
//   - preActivations: valor PRÉ-ativação (z = W·a + b) — necessário para backprop
//
// Arquitetura:
//   - Camadas ocultas: ReLU(z)  — não-linearidade eficiente
//   - Camada de saída: sigmoid(z) — comprime para [0,1] representando RGB
func imgregForward(net Net, x, y float64) (activations [][]float64, preActivations [][]float64) {
	nLayers := len(net.LayerSizes)

	activations = make([][]float64, nLayers)
	preActivations = make([][]float64, nLayers)

	// Camada 0 = entrada: a[0] = [x, y]
	activations[0] = []float64{x, y}
	preActivations[0] = []float64{x, y} // sem pré-ativação na entrada

	// Propaga camada por camada (l vai de 0 até nLayers-2 = última transição)
	for l := 0; l < len(net.W); l++ {
		fanIn := len(net.W[l])
		fanOut := len(net.W[l][0])

		z := make([]float64, fanOut)
		a := make([]float64, fanOut)

		// Calcula pré-ativação: z[j] = bias[j] + Σ_i(a_anterior[i] * W[l][i][j])
		for j := 0; j < fanOut; j++ {
			z[j] = net.B[l][j] // começa com o bias
			for i := 0; i < fanIn; i++ {
				z[j] += activations[l][i] * net.W[l][i][j]
			}

			// Aplica ativação:
			// - Camadas ocultas (não é a última): ReLU
			// - Camada de saída (última): Sigmoid → saída em [0,1] para RGB
			isOutputLayer := (l == len(net.W)-1)
			if isOutputLayer {
				a[j] = sigmoid(z[j])
			} else {
				a[j] = relu(z[j])
			}
		}

		preActivations[l+1] = z
		activations[l+1] = a
	}

	return activations, preActivations
}

// =============================================================================
// BACKWARD PASS (Backpropagation)
// =============================================================================

// imgregBackward — calcula os gradientes da loss (MSE) em relação a todos os pesos
// usando a regra da cadeia (chain rule), propagando o erro da saída para a entrada.
//
// Loss MSE por pixel: L = 0.5 * Σ_k (target[k] - output[k])²
//
// Retorna:
//   - gradW[l][i][j] = ∂L/∂W[l][i][j]
//   - gradB[l][j]    = ∂L/∂B[l][j]
func imgregBackward(net Net, activations [][]float64, preActivations [][]float64, target [3]float64) (gradW [][][]float64, gradB [][]float64) {
	nTransitions := len(net.W)

	gradW = make([][][]float64, nTransitions)
	gradB = make([][]float64, nTransitions)

	for l := 0; l < nTransitions; l++ {
		fanIn := len(net.W[l])
		fanOut := len(net.W[l][0])
		gradW[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			gradW[l][i] = make([]float64, fanOut)
		}
		gradB[l] = make([]float64, fanOut)
	}

	// PASSO 1: Calcula δ (delta) na camada de saída
	// δ_k = (target[k] - output[k]) * sigmoid'(output[k])
	// Onde sigmoid'(y) = y * (1 - y)
	// O sinal negativo vem do gradiente de MSE: ∂L/∂output = -(target - output)
	nLayers := len(activations)
	outputLayer := activations[nLayers-1]
	deltas := make([][]float64, nLayers)

	// Delta da camada de saída
	deltaOutput := make([]float64, len(outputLayer))
	for k := 0; k < len(outputLayer); k++ {
		erro := target[k] - outputLayer[k]
		// δ_k = erro * f'(z_k) onde f = sigmoid e f'(y) = y*(1-y)
		deltaOutput[k] = erro * sigmoidDeriv(outputLayer[k])
	}
	deltas[nLayers-1] = deltaOutput

	// PASSO 2: Propaga δ para trás pelas camadas ocultas (backpropagation)
	// δ_j = (Σ_k δ_k * W[l][j][k]) * relu'(z_j)
	// Onde relu'(z) = 1 se z > 0, 0 caso contrário
	for l := nTransitions - 1; l >= 1; l-- {
		fanIn := len(net.W[l-1][0]) // tamanho da camada l (que recebe os deltas)
		fanOut := len(net.W[l])     // tamanho da camada l+1 (que tem os deltas calculados)

		// fanIn aqui é o número de neurônios na camada l (oculta)
		deltaHidden := make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			// Acumula contribuição de todos os δ da camada seguinte
			var soma float64
			for k := 0; k < len(deltas[l+1]); k++ {
				soma += deltas[l+1][k] * net.W[l][j][k]
			}
			// Multiplica pela derivada de ReLU na pré-ativação z_j
			deltaHidden[j] = soma * reluDeriv(preActivations[l][j])
		}
		_ = fanIn
		deltas[l] = deltaHidden
	}

	// PASSO 3: Calcula gradientes dos pesos usando os deltas
	// ∂L/∂W[l][i][j] = δ_j * a_i (ativação da camada anterior)
	// ∂L/∂B[l][j]    = δ_j
	for l := 0; l < nTransitions; l++ {
		fanIn := len(net.W[l])
		fanOut := len(net.W[l][0])
		for i := 0; i < fanIn; i++ {
			for j := 0; j < fanOut; j++ {
				gradW[l][i][j] = deltas[l+1][j] * activations[l][i]
			}
		}
		for j := 0; j < fanOut; j++ {
			gradB[l][j] = deltas[l+1][j]
		}
	}

	return gradW, gradB
}

// atualizarPesos — aplica SGD (Stochastic Gradient Descent):
// W[l][i][j] -= α * gradW[l][i][j]
// B[l][j]    -= α * gradB[l][j]
// O sinal negativo segue a direção do gradiente descendente (minimizar loss)
func atualizarPesos(net Net, gradW [][][]float64, gradB [][]float64, lr float64) Net {
	for l := range net.W {
		for i := range net.W[l] {
			for j := range net.W[l][i] {
				// Desce o gradiente: move na direção oposta ao gradiente da loss
				net.W[l][i][j] += lr * gradW[l][i][j]
			}
		}
		for j := range net.B[l] {
			net.B[l][j] += lr * gradB[l][j]
		}
	}
	return net
}

// =============================================================================
// GERAÇÃO DE IMAGENS-ALVO (16×16 pixels, proceduralmente)
// =============================================================================

// GetTarget — retorna os 256 pixels RGB [0,1] da imagem escolhida
// Os pixels são armazenados em ordem row-major: pixel[y*16 + x]
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
// Usa a equação do coração: (x²+y²-1)³ - x²y³ ≤ 0
func imgCoracao() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			// Normaliza coordenadas para [-1, 1] — mesmas que o treino usa
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0) // inverte Y (tela vs matemática)

			// Escala para [-1.3, 1.3] só para a equação do coração funcionar bem
			cx, cy := x*1.3, y*1.3
			x2, y2 := cx*cx, cy*cy
			// Equação do coração: (x²+y²-1)³ ≤ x²y³
			f := (x2+y2-1)*(x2+y2-1)*(x2+y2-1) - x2*cy*y2

			var r, g, b float64
			if f <= 0 {
				// Dentro do coração: vermelho com gradiente de intensidade
				intensity := 1.0 - math.Sqrt(x2+y2)*0.3
				intensity = math.Max(0.6, math.Min(1.0, intensity))
				r = intensity
				g = 0.05
				b = 0.1
			} else {
				// Fora do coração: fundo escuro azulado
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
			// Normaliza para [-1, 1]
			x := (float64(px)/15.0)*2.0 - 1.0
			y := -((float64(py)/15.0)*2.0 - 1.0)

			dist := math.Sqrt(x*x + y*y)

			var r, g, b float64
			// Fundo azul escuro
			r, g, b = 0.05, 0.08, 0.25

			// Rosto amarelo (círculo externo r < 0.85)
			if dist < 0.85 {
				r, g, b = 0.95, 0.85, 0.05
			}

			// Olho esquerdo (círculo pequeno)
			olhoEsqX, olhoEsqY := -0.28, 0.28
			if math.Sqrt((x-olhoEsqX)*(x-olhoEsqX)+(y-olhoEsqY)*(y-olhoEsqY)) < 0.14 {
				r, g, b = 0.1, 0.07, 0.05
			}

			// Olho direito
			olhoDirX, olhoDirY := 0.28, 0.28
			if math.Sqrt((x-olhoDirX)*(x-olhoDirX)+(y-olhoDirY)*(y-olhoDirY)) < 0.14 {
				r, g, b = 0.1, 0.07, 0.05
			}

			// Sorriso: arco de círculo na parte inferior
			// Usa distância a um ponto central de círculo maior
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
// Demonstra que a rede consegue aprender funções periódicas complexas
func imgRadial() [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0

			dist := math.Sqrt(x*x + y*y)
			angle := math.Atan2(y, x)

			// Ondas radiais com modulação angular
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

			// Fundo verde
			r, g, b = 0.0, 0.5, 0.15

			// Losango amarelo: |x| + |y| < 0.75
			if math.Abs(x)*0.9+math.Abs(y) < 0.65 {
				r, g, b = 0.95, 0.80, 0.0
			}

			// Círculo azul: dist < 0.35
			dist := math.Sqrt(x*x + y*y)
			if dist < 0.35 {
				r, g, b = 0.0, 0.2, 0.7
			}

			// Faixa branca diagonal (representando "Ordem e Progresso" simplificado)
			if dist < 0.35 && math.Abs(y) < 0.06 {
				r, g, b = 0.95, 0.95, 0.95
			}

			pixels[py*16+px] = [3]float64{r, g, b}
		}
	}
	return pixels
}

// =============================================================================
// LOOP DE TREINAMENTO
// =============================================================================

// predict — aplica a rede a todos os 256 pixels e retorna as predições RGB
func predict(net Net) [][3]float64 {
	pixels := make([][3]float64, 256)
	for py := 0; py < 16; py++ {
		for px := 0; px < 16; px++ {
			// Normaliza coordenadas para [-1, 1]
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0

			acts, _ := imgregForward(net, x, y)
			output := acts[len(acts)-1]
			pixels[py*16+px] = [3]float64{output[0], output[1], output[2]}
		}
	}
	return pixels
}

// Treinar — loop de treinamento principal com streaming SSE
// Cada época = apresentar os 256 pixels em ordem aleatória (SGD com shuffle)
// Envia Step pelo canal a cada epochStep épocas para atualizar o frontend
// ctx permite cancelamento externo (ex: reset enquanto treina)
func Treinar(ctx context.Context, cfg Config, progressCh chan<- Step) Net {
	rng := rand.New(rand.NewSource(42))

	// Limita configurações para valores válidos
	if cfg.MaxEpocas <= 0 {
		cfg.MaxEpocas = 2000
	}
	if cfg.HiddenLayers < 1 {
		cfg.HiddenLayers = 2
	}
	if cfg.NeuronsPerLayer < 4 {
		cfg.NeuronsPerLayer = 16
	}

	// Inicializa a rede com He initialization
	net := inicializar(cfg, rng)

	// Carrega a imagem-alvo (256 pixels RGB)
	target := GetTarget(cfg.Imagem)

	// Cria índices dos 256 pixels para shuffle a cada época
	indices := make([]int, 256)
	for i := range indices {
		indices[i] = i
	}

	lossHistorico := make([]float64, 0, cfg.MaxEpocas)

	// Frequência de envio de steps via SSE:
	// enviar a cada N épocas para não saturar o canal
	const epochStep = 5

	for epoca := 1; epoca <= cfg.MaxEpocas; epoca++ {
		// Verifica cancelamento (ex: reset do frontend)
		select {
		case <-ctx.Done():
			close(progressCh)
			return net
		default:
		}
		// Yield para o runtime (importante no WASM para não bloquear o event loop do Worker)
		runtime.Gosched()

		// SHUFFLE dos índices — SGD estocástico sem ordem fixa
		// Evita que a rede "memorize" a ordem de apresentação dos pixels
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		var lossTotal float64
		activeLayer := ((epoca / epochStep) % len(net.W)) // alterna a cada step enviado

		// Para cada pixel na ordem embaralhada:
		for _, idx := range indices {
			py := idx / 16
			px := idx % 16

			// Normaliza coordenadas para [-1, 1]
			// A rede aprende o mapeamento contínuo (x,y) → (R,G,B)
			x := (float64(px)/15.0)*2.0 - 1.0
			y := (float64(py)/15.0)*2.0 - 1.0

			t := target[idx] // alvo RGB para este pixel

			// FORWARD PASS — propaga (x,y) pela rede
			acts, preActs := imgregForward(net, x, y)
			output := acts[len(acts)-1]

			// Calcula MSE deste pixel: 0.5 * Σ(target - output)²
			for k := 0; k < 3; k++ {
				d := t[k] - output[k]
				lossTotal += 0.5 * d * d
			}

			// BACKWARD PASS — calcula gradientes da loss
			var tArr [3]float64
			tArr[0], tArr[1], tArr[2] = t[0], t[1], t[2]
			gradW, gradB := imgregBackward(net, acts, preActs, tArr)

			// UPDATE DOS PESOS — SGD: W -= α * ∂L/∂W
			net = atualizarPesos(net, gradW, gradB, cfg.LearningRate)
		}

		// Loss média desta época (MSE normalizado pelo número de pixels e canais)
		lossMedia := lossTotal / float64(256*3)
		lossHistorico = append(lossHistorico, lossMedia)

		// Envia step pelo canal SSE a cada epochStep épocas
		if epoca%epochStep == 0 || epoca == 1 || epoca == cfg.MaxEpocas {
			pixels := predict(net)
			step := Step{
				Epoca:        epoca,
				MaxEpocas:    cfg.MaxEpocas,
				Loss:         lossMedia,
				OutputPixels: pixels,
				ActiveLayer:  activeLayer,
			}
			select {
			case progressCh <- step:
			default: // descarta se o canal estiver cheio
			}
		}
	}

	// Envia mensagem final com Done=true — usa select para não bloquear se o cliente desconectou
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
	}:
	case <-ctx.Done():
	}

	close(progressCh)
	return net
}
