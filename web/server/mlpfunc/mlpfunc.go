package mlpfunc

import (
	"math"
	"math/rand"
)

// =============================================================================
// MLP Funcoes — Aproximacao de funcoes com backpropagation
//
// Baseado no codigo Python do slide da Aula 06 (MLPfuncaoBase.py).
// A rede aprende a aproximar uma funcao matematica (ex: sin(x)*sin(2x))
// usando 50 pontos igualmente espacados no intervalo [-1, 1].
//
// Arquitetura: 1 entrada (x) -> N camadas ocultas -> 1 saida
//
// Diferente dos MLPs anteriores que classificam padroes, este faz REGRESSAO:
// a saida eh um valor continuo, nao uma classe.
// =============================================================================

const (
	nIn     = 1  // uma entrada: o valor x
	nOut    = 1  // uma saida: o valor y aproximado
	nPontos = 50 // 50 pontos igualmente espacados
)

// Config permite customizar hiperparametros via frontend
type Config struct {
	Funcao       string  `json:"funcao"`
	NHid         int     `json:"nHid"`                   // fallback: single hidden layer
	HiddenLayers []int   `json:"hiddenLayers,omitempty"` // neuron counts per hidden layer
	Alfa         float64 `json:"alfa"`
	MaxCiclo     int     `json:"maxCiclo"`
	Ativacao     string  `json:"ativacao"`
}

// DefaultConfig retorna a configuracao padrao (valores do slide)
func DefaultConfig() Config {
	return Config{
		Funcao:       "sin(x)*sin(2x)",
		HiddenLayers: []int{200},
		Alfa:         0.005,
		MaxCiclo:     100000,
		Ativacao:     "tanh",
	}
}

// resolveHiddenLayers returns the hidden layer sizes from config,
// falling back to NHid if HiddenLayers is empty.
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
// W[l][i][j] = peso da conexao do neuronio i na camada l para o neuronio j na camada l+1
// B[l][j]    = bias do neuronio j na camada l+1
//
// Camada 0: entrada(1) -> oculta[0]
// Camada 1: oculta[0]  -> oculta[1]   (se >1 camada oculta)
// ...
// Camada N: oculta[N-1] -> saida(1)
type MLP struct {
	layers int           // numero de transicoes (len(hiddenLayers) + 1)
	sizes  []int         // tamanho de cada camada: [1, h0, h1, ..., 1]
	W      [][][]float64 // W[l][i][j]
	B      [][]float64   // B[l][j]
}

// FuncPoint representa um ponto da funcao
type FuncPoint struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	YPred float64 `json:"yPred"`
}

// FuncStep eh enviado via SSE a cada N ciclos
type FuncStep struct {
	Ciclo     int         `json:"ciclo"`
	ErroTotal float64     `json:"erroTotal"`
	Pontos    []FuncPoint `json:"pontos"`
}

// FuncResult eh o resultado final do treinamento
type FuncResult struct {
	Convergiu     bool        `json:"convergiu"`
	Ciclos        int         `json:"ciclos"`
	ErroFinal     float64     `json:"erroFinal"`
	ErroHistorico []float64   `json:"erroHistorico"`
	Pontos        []FuncPoint `json:"pontos"`
	Funcao        string      `json:"funcao"`
}

// ----- Funcoes matematicas disponiveis -----

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

// ----- Funcoes de ativacao -----

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

func ativacaoDeriv(nome string, y float64) float64 {
	switch nome {
	case "sigmoid":
		return y * (1 - y)
	case "relu":
		if y > 0 {
			return 1
		}
		return 0
	default: // "tanh"
		return (1 + y) * (1 - y)
	}
}

// ----- Geracao do dataset -----

func gerarDataset(funcao string, xmin, xmax float64, n int) ([]float64, []float64) {
	xs := make([]float64, n)
	ts := make([]float64, n)
	for i := 0; i < n; i++ {
		xs[i] = xmin + float64(i)*(xmax-xmin)/float64(n-1)
		ts[i] = Funcao(funcao, xs[i])
	}
	return xs, ts
}

// ----- Inicializacao de pesos -----

func inicializar(hiddenLayers []int) MLP {
	rng := rand.New(rand.NewSource(42))

	// Build sizes: [1, h0, h1, ..., 1]
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

		// Determine random range based on layer position
		var halfRange float64
		if l == 0 {
			// input -> first hidden: [-1, 1]
			halfRange = 1.0
		} else if l == nTransitions-1 {
			// last hidden -> output: [-0.2, 0.2]
			halfRange = 0.2
		} else {
			// middle hidden -> hidden: [-0.5, 0.5]
			halfRange = 0.5
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

// ----- Forward pass -----
// Returns activations for all layers (including input and output).
// a[0] = input (single value wrapped in slice)
// a[1] = first hidden layer activations
// ...
// a[N] = output layer activations

func forward(m MLP, x float64, ativ string) (a [][]float64) {
	a = make([][]float64, m.layers+1)

	// Input layer
	a[0] = []float64{x}

	// Propagate through each layer transition
	for l := 0; l < m.layers; l++ {
		fanOut := m.sizes[l+1]
		a[l+1] = make([]float64, fanOut)
		for j := 0; j < fanOut; j++ {
			sum := m.B[l][j]
			for i := 0; i < m.sizes[l]; i++ {
				sum += a[l][i] * m.W[l][i][j]
			}
			a[l+1][j] = ativacao(ativ, sum)
		}
	}

	return
}

// ----- Backward pass + atualizacao de pesos -----

func backwardAndUpdate(m *MLP, a [][]float64, t, alfa float64, ativ string) {
	// Compute deltas for each layer (from output to first hidden)
	deltas := make([][]float64, m.layers)

	// Output layer delta
	outIdx := m.layers
	outSize := m.sizes[outIdx]
	deltas[m.layers-1] = make([]float64, outSize)
	for j := 0; j < outSize; j++ {
		deriv := ativacaoDeriv(ativ, a[outIdx][j])
		deltas[m.layers-1][j] = (t - a[outIdx][j]) * deriv
	}

	// Hidden layer deltas (backpropagate)
	for l := m.layers - 2; l >= 0; l-- {
		curSize := m.sizes[l+1]
		nextSize := m.sizes[l+2]
		deltas[l] = make([]float64, curSize)
		for i := 0; i < curSize; i++ {
			sum := 0.0
			for j := 0; j < nextSize; j++ {
				sum += deltas[l+1][j] * m.W[l+1][i][j]
			}
			deriv := ativacaoDeriv(ativ, a[l+1][i])
			deltas[l][i] = sum * deriv
		}
	}

	// Update weights and biases for each layer
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

// ----- Treinamento -----

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

	erroAlvo := 0.02

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		erroTotal := 0.0

		for padrao := 0; padrao < nPontos; padrao++ {
			a := forward(m, xs[padrao], ativ)
			y := a[m.layers][0]
			d := ts[padrao] - y
			erroTotal += 0.5 * d * d
			backwardAndUpdate(&m, a, ts[padrao], cfg.Alfa, ativ)
		}

		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		if progressCh != nil && ciclo%100 == 0 {
			pontos := predizer(m, xs, ts, ativ)
			step := FuncStep{
				Ciclo:     ciclo,
				ErroTotal: erroTotal,
				Pontos:    pontos,
			}
			select {
			case progressCh <- step:
			default:
			}
		}

		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.Pontos = predizer(m, xs, ts, ativ)
			return res
		}
	}

	res.Convergiu = false
	res.Ciclos = cfg.MaxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Pontos = predizer(m, xs, ts, ativ)
	return res
}

func predizer(m MLP, xs, ts []float64, ativ string) []FuncPoint {
	pontos := make([]FuncPoint, len(xs))
	for i := range xs {
		a := forward(m, xs[i], ativ)
		y := a[m.layers][0]
		pontos[i] = FuncPoint{X: xs[i], Y: ts[i], YPred: y}
	}
	return pontos
}
