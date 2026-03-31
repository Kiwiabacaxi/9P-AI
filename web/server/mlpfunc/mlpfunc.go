package mlpfunc

import (
	"math"
	"math/rand"
)

// =============================================================================
// MLP Funções — Aproximação de funções com backpropagation (Aula 06)
//
// Baseado no código Python do slide da Aula 06 (MLPfuncaoBase.py).
// A rede aprende a aproximar uma função matemática (ex: sin(x)*sin(2x))
// usando 50 pontos igualmente espaçados no intervalo [-1, 1].
//
// Arquitetura: 1 entrada (x) → N camadas ocultas (tanh/sigmoid/relu) → 1 saída
//
// Diferente dos MLPs anteriores que classificam padrões, este faz REGRESSÃO:
// a saída é um valor contínuo, não uma classe.
//
// Algoritmo (slide Aula 06):
//  1. Gerar dataset: 50 pontos (x_i, t_i) onde t_i = f(x_i)
//  2. Inicializar pesos aleatórios
//  3. Para cada ciclo:
//     a. Para cada padrão (x, t):
//        - Forward: propagar x pela rede → obter saída y
//        - Calcular erro: E += 0.5 * (t - y)²
//        - Backward: calcular deltas e atualizar pesos (regra delta generalizada)
//     b. Se erro total <= erro alvo → convergiu
// =============================================================================

const (
	nIn     = 1  // uma entrada: o valor x
	nOut    = 1  // uma saída: o valor y aproximado
	nPontos = 50 // 50 pontos igualmente espaçados (do slide)
)

// Config permite customizar hiperparâmetros via frontend
type Config struct {
	Funcao       string  `json:"funcao"`
	NHid         int     `json:"nHid"`                   // fallback: single hidden layer
	HiddenLayers []int   `json:"hiddenLayers,omitempty"` // neuron counts per hidden layer
	Alfa         float64 `json:"alfa"`                   // taxa de aprendizado (learning rate)
	MaxCiclo     int     `json:"maxCiclo"`               // máximo de épocas
	Ativacao     string  `json:"ativacao"`               // função de ativação: tanh, sigmoid, relu
}

// DefaultConfig retorna a configuração padrão (valores do slide)
func DefaultConfig() Config {
	return Config{
		Funcao:       "sin(x)*sin(2x)",
		HiddenLayers: []int{200},
		Alfa:         0.005,
		MaxCiclo:     100000,
		Ativacao:     "tanh",
	}
}

// resolveHiddenLayers retorna os tamanhos das camadas ocultas,
// usando HiddenLayers se disponível, senão NHid como fallback.
func resolveHiddenLayers(cfg Config) []int {
	if len(cfg.HiddenLayers) > 0 {
		return cfg.HiddenLayers
	}
	if cfg.NHid > 0 {
		return []int{cfg.NHid}
	}
	return []int{200}
}

// MLP armazena os pesos da rede com suporte a N camadas ocultas.
//
// Estrutura de camadas (exemplo com 2 camadas ocultas de 100 neurônios):
//   sizes = [1, 100, 100, 1]    → entrada, oculta1, oculta2, saída
//   layers = 3                   → 3 transições de pesos (W[0], W[1], W[2])
//
// W[l][i][j] = peso da conexão do neurônio i na camada l para o neurônio j na camada l+1
// B[l][j]    = bias do neurônio j na camada l+1
type MLP struct {
	layers int           // número de transições de peso (len(sizes) - 1)
	sizes  []int         // tamanho de cada camada: [nIn, h0, h1, ..., nOut]
	W      [][][]float64 // W[l][i][j] — pesos entre camada l e l+1
	B      [][]float64   // B[l][j]    — bias da camada l+1
}

// FuncPoint representa um ponto da função (x, y_real, y_predito)
type FuncPoint struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`     // valor real da função
	YPred float64 `json:"yPred"` // valor predito pela rede
}

// FuncStep é enviado via SSE a cada 100 ciclos para atualizar o frontend
type FuncStep struct {
	Ciclo       int         `json:"ciclo"`
	ErroTotal   float64     `json:"erroTotal"`
	Pontos      []FuncPoint `json:"pontos"`
	ActiveLayer int         `json:"activeLayer"` // para animação do NetworkViz
}

// FuncResult é o resultado final do treinamento
type FuncResult struct {
	Convergiu     bool        `json:"convergiu"`
	Ciclos        int         `json:"ciclos"`
	ErroFinal     float64     `json:"erroFinal"`
	ErroHistorico []float64   `json:"erroHistorico"`
	Pontos        []FuncPoint `json:"pontos"`
	Funcao        string      `json:"funcao"`
}

// =============================================================================
// Funções matemáticas disponíveis para aproximação
// =============================================================================

func Funcao(nome string, x float64) float64 {
	switch nome {
	case "sin(x)*sin(2x)":
		return math.Sin(x) * math.Sin(2*x)
	case "sin(x)":
		return math.Sin(x)
	case "x^2":
		return x * x
	case "x^3":
		return x * x * x
	default:
		return math.Sin(x) * math.Sin(2*x)
	}
}

func FuncoesDisponiveis() []string {
	return []string{"sin(x)*sin(2x)", "sin(x)", "x^2", "x^3"}
}

// =============================================================================
// Funções de ativação e suas derivadas
//
// Do slide: "Função de ativação contínua e diferenciável"
// - tanh: f(x) = tanh(x),        f'(x) = (1+y)(1-y)  onde y = tanh(x)
// - sigmoid: f(x) = 1/(1+e^-x),  f'(x) = y(1-y)      onde y = sigmoid(x)
// - relu: f(x) = max(0, x),       f'(x) = 1 se x>0, 0 caso contrário
// =============================================================================

func ativacao(nome string, x float64) float64 {
	switch nome {
	case "sigmoid":
		return 1.0 / (1.0 + math.Exp(-x))
	case "relu":
		if x > 0 {
			return x
		}
		return 0
	default: // "tanh"
		return math.Tanh(x)
	}
}

// ativacaoDeriv calcula a derivada da função de ativação.
// NOTA: recebe y (saída da ativação), não x (entrada), para eficiência.
func ativacaoDeriv(nome string, y float64) float64 {
	switch nome {
	case "sigmoid":
		return y * (1 - y) // f'(x) = f(x) * (1 - f(x))
	case "relu":
		if y > 0 {
			return 1
		}
		return 0
	default: // "tanh"
		return (1 + y) * (1 - y) // f'(x) = 1 - tanh²(x) = (1+y)(1-y)
	}
}

// =============================================================================
// Geração do dataset
//
// Gera N pontos igualmente espaçados em [xmin, xmax].
// xs[i] = valor de x, ts[i] = f(x) = valor target (real)
// =============================================================================

func gerarDataset(funcao string, xmin, xmax float64, n int) ([]float64, []float64) {
	xs := make([]float64, n)
	ts := make([]float64, n)
	for i := 0; i < n; i++ {
		xs[i] = xmin + float64(i)*(xmax-xmin)/float64(n-1)
		ts[i] = Funcao(funcao, xs[i])
	}
	return xs, ts
}

// =============================================================================
// Inicialização de pesos
//
// Pesos inicializados aleatoriamente com ranges diferentes por camada:
//   - Entrada → primeira oculta: [-1.0, +1.0]   (range maior para capturar variação)
//   - Oculta → oculta:           [-0.5, +0.5]   (range médio)
//   - Última oculta → saída:     [-0.2, +0.2]   (range menor para estabilidade)
//
// Seed fixa (42) para reprodutibilidade.
// =============================================================================

func inicializar(hiddenLayers []int) MLP {
	rng := rand.New(rand.NewSource(42))

	// Construir vetor de tamanhos: [1, h0, h1, ..., 1]
	sizes := make([]int, 0, len(hiddenLayers)+2)
	sizes = append(sizes, nIn)
	sizes = append(sizes, hiddenLayers...)
	sizes = append(sizes, nOut)

	nTransitions := len(sizes) - 1

	m := MLP{
		layers: nTransitions,
		sizes:  sizes,
		W:      make([][][]float64, nTransitions),
		B:      make([][]float64, nTransitions),
	}

	for l := 0; l < nTransitions; l++ {
		fanIn := sizes[l]
		fanOut := sizes[l+1]

		// Range de inicialização depende da posição da camada
		var halfRange float64
		if l == 0 {
			halfRange = 1.0 // entrada → oculta: [-1, 1]
		} else if l == nTransitions-1 {
			halfRange = 0.2 // oculta → saída: [-0.2, 0.2]
		} else {
			halfRange = 0.5 // oculta → oculta: [-0.5, 0.5]
		}

		m.W[l] = make([][]float64, fanIn)
		for i := 0; i < fanIn; i++ {
			m.W[l][i] = make([]float64, fanOut)
			for j := 0; j < fanOut; j++ {
				m.W[l][i][j] = rng.Float64()*2*halfRange - halfRange
			}
		}

		m.B[l] = make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			m.B[l][j] = rng.Float64()*2*halfRange - halfRange
		}
	}

	return m
}

// =============================================================================
// Forward pass (propagação direta)
//
// Do slide: "Propagar o sinal de entrada camada por camada"
//
// Para cada camada l:
//   net_j = bias_j + Σ(a_i * W[l][i][j])   — combinação linear
//   a_j   = f(net_j)                         — aplicar ativação
//
// Retorna as ativações de TODAS as camadas (necessárias para o backward).
//   a[0] = entrada  (single value)
//   a[1] = primeira camada oculta
//   ...
//   a[N] = saída
// =============================================================================

func forward(m MLP, x float64, ativ string) (a [][]float64) {
	a = make([][]float64, m.layers+1)

	// Camada de entrada: apenas o valor x
	a[0] = []float64{x}

	// Propagar camada por camada
	for l := 0; l < m.layers; l++ {
		fanOut := m.sizes[l+1]
		a[l+1] = make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			sum := m.B[l][j] // bias
			for i := 0; i < m.sizes[l]; i++ {
				sum += a[l][i] * m.W[l][i][j] // soma ponderada
			}
			a[l+1][j] = ativacao(ativ, sum) // aplicar função de ativação
		}
	}

	return
}

// =============================================================================
// Backward pass + atualização de pesos (backpropagation)
//
// Do slide: "Regra Delta Generalizada" (Rumelhart, Hinton, Williams 1986)
//
// 1. Calcular delta da camada de saída:
//    δ_k = (t_k - y_k) * f'(y_k)
//
// 2. Propagar deltas para camadas ocultas (de trás pra frente):
//    δ_j = (Σ δ_k * W[j][k]) * f'(z_j)
//
// 3. Atualizar pesos (regra delta):
//    W[i][j] += α * δ_j * a_i     (α = taxa de aprendizado)
//    B[j]    += α * δ_j
// =============================================================================

func backwardAndUpdate(m *MLP, a [][]float64, t, alfa float64, ativ string) {
	// 1. Calcular deltas para cada camada (da saída para a primeira oculta)
	deltas := make([][]float64, m.layers)

	// Delta da camada de saída: δ_k = (target - saída) * f'(saída)
	outIdx := m.layers
	outSize := m.sizes[outIdx]
	deltas[m.layers-1] = make([]float64, outSize)
	for j := 0; j < outSize; j++ {
		deriv := ativacaoDeriv(ativ, a[outIdx][j])
		deltas[m.layers-1][j] = (t - a[outIdx][j]) * deriv
	}

	// 2. Deltas das camadas ocultas (backpropagation do erro)
	for l := m.layers - 2; l >= 0; l-- {
		curSize := m.sizes[l+1]
		nextSize := m.sizes[l+2]
		deltas[l] = make([]float64, curSize)
		for i := 0; i < curSize; i++ {
			sum := 0.0
			for j := 0; j < nextSize; j++ {
				sum += deltas[l+1][j] * m.W[l+1][i][j] // erro propagado
			}
			deriv := ativacaoDeriv(ativ, a[l+1][i])
			deltas[l][i] = sum * deriv
		}
	}

	// 3. Atualizar pesos e biases (regra delta: ΔW = α * δ * a)
	for l := 0; l < m.layers; l++ {
		fanIn := m.sizes[l]
		fanOut := m.sizes[l+1]
		for i := 0; i < fanIn; i++ {
			for j := 0; j < fanOut; j++ {
				m.W[l][i][j] += alfa * deltas[l][j] * a[l][i]
			}
		}
		for j := 0; j < fanOut; j++ {
			m.B[l][j] += alfa * deltas[l][j]
		}
	}
}

// =============================================================================
// Treinamento principal
//
// Loop de treinamento (do slide):
//   Para cada ciclo (época):
//     Para cada padrão (x_i, t_i):
//       1. Forward: calcular saída y_i
//       2. Erro: E += 0.5 * (t_i - y_i)²
//       3. Backward: atualizar pesos
//     Se erro total <= erro_alvo: CONVERGIU
//
// Envia progresso via SSE a cada 100 ciclos para o frontend.
// =============================================================================

func Treinar(progressCh chan<- FuncStep, cfg Config) FuncResult {
	hiddenLayers := resolveHiddenLayers(cfg)
	if cfg.Alfa <= 0 {
		cfg.Alfa = 0.005
	}
	if cfg.MaxCiclo <= 0 {
		cfg.MaxCiclo = 100000
	}
	if cfg.Ativacao == "" {
		cfg.Ativacao = "tanh"
	}
	if cfg.Funcao == "" {
		cfg.Funcao = "sin(x)*sin(2x)"
	}

	xs, ts := gerarDataset(cfg.Funcao, -1, 1, nPontos)
	m := inicializar(hiddenLayers)
	ativ := cfg.Ativacao

	var res FuncResult
	res.Funcao = cfg.Funcao

	erroAlvo := 0.02 // erro quadrático total aceitável

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		erroTotal := 0.0

		// Apresentar todos os padrões à rede
		for padrao := 0; padrao < nPontos; padrao++ {
			a := forward(m, xs[padrao], ativ)         // propagação direta
			y := a[m.layers][0]                        // saída da rede
			d := ts[padrao] - y                        // diferença (target - saída)
			erroTotal += 0.5 * d * d                   // erro quadrático: E = 0.5*(t-y)²
			backwardAndUpdate(&m, a, ts[padrao], cfg.Alfa, ativ) // backpropagation
		}

		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		// Enviar progresso ao frontend via SSE a cada 100 ciclos
		if progressCh != nil && ciclo%100 == 0 {
			pontos := predizer(m, xs, ts, ativ)
			step := FuncStep{
				Ciclo:       ciclo,
				ErroTotal:   erroTotal,
				Pontos:      pontos,
				ActiveLayer: (ciclo / 100) % m.layers,
			}
			select {
			case progressCh <- step:
			default:
			}
		}

		// Critério de parada: erro <= erro alvo
		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.Pontos = predizer(m, xs, ts, ativ)
			return res
		}
	}

	// Não convergiu dentro do limite de ciclos
	res.Convergiu = false
	res.Ciclos = cfg.MaxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Pontos = predizer(m, xs, ts, ativ)
	return res
}

// predizer gera os pontos (x, y_real, y_predito) para visualização no gráfico.
func predizer(m MLP, xs, ts []float64, ativ string) []FuncPoint {
	pontos := make([]FuncPoint, len(xs))
	for i := range xs {
		a := forward(m, xs[i], ativ)
		y := a[m.layers][0]
		pontos[i] = FuncPoint{X: xs[i], Y: ts[i], YPred: y}
	}
	return pontos
}
