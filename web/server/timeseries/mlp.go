package timeseries

// =============================================================================
// MLP Série Temporal — Previsão de preço de ações (Aula 08)
//
// Do slide: "Modelar uma MLP que aprenda a dinâmica de uma ação brasileira
// e faça a predição para o dia seguinte."
//
// Arquitetura: windowSize entradas → hiddenSize ocultos → 1 saída
// Entrada: preços normalizados dos últimos N dias (janela deslizante)
// Saída: preço normalizado predito para o dia seguinte
//
// A rede aprende padrões temporais no preço (tendência, momentum)
// e tenta extrapolar para o próximo dia.
// =============================================================================

import (
	"math"
	"math/rand"
	"time"
)

// Config permite customizar hiperparâmetros via frontend
type Config struct {
	Ticker       string  `json:"ticker"`       // ex: "COGN3.SA"
	WindowSize   int     `json:"windowSize"`   // dias de entrada (default 5)
	HiddenSize   int     `json:"hiddenSize"`   // neurônios ocultos (default 32)
	Alfa         float64 `json:"alfa"`         // learning rate (default 0.01)
	MaxCiclo     int     `json:"maxCiclo"`     // épocas (default 5000)
	Ativacao     string  `json:"ativacao"`     // tanh/sigmoid/relu
	ValidDays    int     `json:"validDays"`    // dias de validação (default 7)
	ForecastDays int     `json:"forecastDays"` // dias de previsão futura (default 7)
	ValidPct     float64 `json:"validPct"`     // % dos dados para validação (0 = usar validDays fixo)
}

func DefaultConfig() Config {
	return Config{
		Ticker:       "COGN3.SA",
		WindowSize:   5,
		HiddenSize:   32,
		Alfa:         0.01,
		MaxCiclo:     5000,
		Ativacao:     "tanh",
		ValidDays:    7,
		ForecastDays: 7,
		ValidPct:     0,
	}
}

// ForecastPoint é um ponto de previsão futura com intervalo de confiança
type ForecastPoint struct {
	Dia      int     `json:"dia"`      // D+1, D+2, ...
	Predito  float64 `json:"predito"`  // preço predito
	Upper    float64 `json:"upper"`    // limite superior (confiança)
	Lower    float64 `json:"lower"`    // limite inferior (confiança)
}

// TimeSeriesStep é enviado via SSE durante treinamento
type TimeSeriesStep struct {
	Ciclo     int     `json:"ciclo"`
	MseTreino float64 `json:"mseTreino"` // MSE no treino
	MseValid  float64 `json:"mseValid"`  // MSE na validação
}

// TimeSeriesResult é o resultado final
type TimeSeriesResult struct {
	Ciclos         int               `json:"ciclos"`
	MseFinal       float64           `json:"mseFinal"`
	RmseFinal      float64           `json:"rmseFinal"`
	MaeFinal       float64           `json:"maeFinal"`
	MseHistorico   []float64         `json:"mseHistorico"`
	Pontos         []TimeSeriesPoint `json:"pontos"`         // todos (treino+valid)
	PontosValid    []TimeSeriesPoint `json:"pontosValid"`    // só validação
	PredicaoAmanha float64           `json:"predicaoAmanha"` // preço predito para amanhã
	Forecast       []ForecastPoint   `json:"forecast"`       // previsão multi-dia futura
	Ticker         string            `json:"ticker"`
	TempoMs        int64             `json:"tempoMs"`
}

// =============================================================================
// MLP para série temporal — mesma estrutura do mlpfunc
// =============================================================================

// TimeSeriesMLP armazena os pesos da rede
type TimeSeriesMLP struct {
	InSize  int           // windowSize
	HidSize int           // neurônios ocultos
	W1      [][]float64   // [inSize][hidSize] — pesos entrada→oculta
	B1      []float64     // [hidSize]
	W2      [][]float64   // [hidSize][1] — pesos oculta→saída
	B2      []float64     // [1]
	Ativ    string        // função de ativação
}

// ativacao aplica a função de ativação
func ativacao(nome string, x float64) float64 {
	switch nome {
	case "sigmoid":
		return 1.0 / (1.0 + math.Exp(-x))
	case "relu":
		if x > 0 { return x }
		return 0
	default: // tanh
		return math.Tanh(x)
	}
}

// ativacaoDeriv calcula a derivada (recebe y = saída da ativação)
func ativacaoDeriv(nome string, y float64) float64 {
	switch nome {
	case "sigmoid":
		return y * (1 - y)
	case "relu":
		if y > 0 { return 1 }
		return 0
	default: // tanh
		return (1 + y) * (1 - y)
	}
}

// Inicializar cria a rede com pesos aleatórios (He/Xavier init)
func Inicializar(inSize, hidSize int, ativ string) *TimeSeriesMLP {
	rng := rand.New(rand.NewSource(42))
	m := &TimeSeriesMLP{
		InSize: inSize, HidSize: hidSize, Ativ: ativ,
		W1: make([][]float64, inSize),
		B1: make([]float64, hidSize),
		W2: make([][]float64, hidSize),
		B2: make([]float64, 1),
	}

	// Entrada → oculta
	scale1 := math.Sqrt(2.0 / float64(inSize))
	for i := range inSize {
		m.W1[i] = make([]float64, hidSize)
		for j := range hidSize {
			m.W1[i][j] = rng.NormFloat64() * scale1
		}
	}
	for j := range hidSize {
		m.B1[j] = rng.NormFloat64() * 0.01
	}

	// Oculta → saída
	scale2 := math.Sqrt(1.0 / float64(hidSize))
	for j := range hidSize {
		m.W2[j] = make([]float64, 1)
		m.W2[j][0] = rng.NormFloat64() * scale2
	}
	m.B2[0] = 0

	return m
}

// Forward propaga entrada pela rede, retorna saída + ativações ocultas
func (m *TimeSeriesMLP) Forward(x []float64) (output float64, hidden []float64) {
	hidden = make([]float64, m.HidSize)

	// Camada oculta: h = ativação(W1·x + B1)
	for j := range m.HidSize {
		sum := m.B1[j]
		for i := range m.InSize {
			sum += x[i] * m.W1[i][j]
		}
		hidden[j] = ativacao(m.Ativ, sum)
	}

	// Camada de saída: y = W2·h + B2 (linear — regressão)
	output = m.B2[0]
	for j := range m.HidSize {
		output += hidden[j] * m.W2[j][0]
	}
	return
}

// BackwardAndUpdate calcula gradientes e atualiza pesos
func (m *TimeSeriesMLP) BackwardAndUpdate(x []float64, hidden []float64, target, output, alfa float64) {
	// Delta saída (linear — derivada = 1)
	deltaOut := target - output

	// Delta oculta: δ_j = deltaOut * W2[j] * f'(h_j)
	deltaHid := make([]float64, m.HidSize)
	for j := range m.HidSize {
		deltaHid[j] = deltaOut * m.W2[j][0] * ativacaoDeriv(m.Ativ, hidden[j])
	}

	// Atualizar W2 (oculta → saída): ΔW = α * δ * h
	for j := range m.HidSize {
		m.W2[j][0] += alfa * deltaOut * hidden[j]
	}
	m.B2[0] += alfa * deltaOut

	// Atualizar W1 (entrada → oculta): ΔW = α * δ * x
	for i := range m.InSize {
		for j := range m.HidSize {
			m.W1[i][j] += alfa * deltaHid[j] * x[i]
		}
	}
	for j := range m.HidSize {
		m.B1[j] += alfa * deltaHid[j]
	}
}

// =============================================================================
// Treinamento
//
// Do slide: "Use dados de 6 meses para treinar, 7 dias para validar"
//
// Para cada ciclo:
//   Para cada par (janela, target) no treino:
//     Forward → calcular erro → Backward → atualizar pesos
//   Calcular MSE treino e validação
//   Enviar progresso via SSE
// =============================================================================

func Treinar(cfg Config, data NormalizedData, progressCh chan<- TimeSeriesStep) (*TimeSeriesMLP, TimeSeriesResult) {
	if cfg.WindowSize <= 0 { cfg.WindowSize = 5 }
	if cfg.HiddenSize <= 0 { cfg.HiddenSize = 32 }
	if cfg.Alfa <= 0 { cfg.Alfa = 0.01 }
	if cfg.MaxCiclo <= 0 { cfg.MaxCiclo = 5000 }
	if cfg.Ativacao == "" { cfg.Ativacao = "tanh" }

	start := time.Now()
	m := Inicializar(cfg.WindowSize, cfg.HiddenSize, cfg.Ativacao)
	nTrain := len(data.TrainX)

	var res TimeSeriesResult
	res.Ticker = cfg.Ticker

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		mseTreino := 0.0

		// Treinar em todos os pares
		for i := range nTrain {
			output, hidden := m.Forward(data.TrainX[i])
			d := data.TrainY[i] - output
			mseTreino += d * d
			m.BackwardAndUpdate(data.TrainX[i], hidden, data.TrainY[i], output, cfg.Alfa)
		}
		mseTreino /= float64(nTrain)
		res.MseHistorico = append(res.MseHistorico, mseTreino)

		// Calcular MSE validação a cada 100 ciclos
		if progressCh != nil && ciclo%100 == 0 {
			mseValid := 0.0
			for i := range len(data.ValidX) {
				output, _ := m.Forward(data.ValidX[i])
				d := data.ValidY[i] - output
				mseValid += d * d
			}
			if len(data.ValidX) > 0 {
				mseValid /= float64(len(data.ValidX))
			}

			step := TimeSeriesStep{
				Ciclo:     ciclo,
				MseTreino: mseTreino,
				MseValid:  mseValid,
			}
			select {
			case progressCh <- step:
			default:
			}
		}
	}

	// Resultado final — gerar predições para todos os dados
	res.Ciclos = cfg.MaxCiclo
	res.MseFinal = res.MseHistorico[len(res.MseHistorico)-1]
	res.RmseFinal = math.Sqrt(res.MseFinal)
	res.TempoMs = time.Since(start).Milliseconds()

	// Gerar pontos preditos (treino + validação)
	allX := append(data.TrainX, data.ValidX...)
	allY := append(data.TrainY, data.ValidY...)
	trainLen := len(data.TrainX)

	var reaisValid, preditosValid []float64

	for i := range len(allX) {
		output, _ := m.Forward(allX[i])
		precoReal := Desnormalizar(allY[i], data.MinPrice, data.MaxPrice)
		precoPred := Desnormalizar(output, data.MinPrice, data.MaxPrice)

		pt := TimeSeriesPoint{
			Data:    data.Dates[i],
			Preco:   precoReal,
			Predito: precoPred,
		}
		res.Pontos = append(res.Pontos, pt)

		if i >= trainLen {
			res.PontosValid = append(res.PontosValid, pt)
			reaisValid = append(reaisValid, precoReal)
			preditosValid = append(preditosValid, precoPred)
		}
	}

	// Métricas na validação
	res.MseFinal, res.RmseFinal, res.MaeFinal = CalcularMetricas(reaisValid, preditosValid)

	// Previsão futura multi-dia com intervalo de confiança
	forecastDays := cfg.ForecastDays
	if forecastDays <= 0 { forecastDays = 7 }

	closes := data.AllClose
	rangeP := data.MaxPrice - data.MinPrice
	if rangeP < 0.0001 { rangeP = 1 }

	if len(closes) >= cfg.WindowSize {
		// Construir janela inicial com os últimos preços normalizados
		window := make([]float64, cfg.WindowSize)
		for j := range cfg.WindowSize {
			window[j] = (closes[len(closes)-cfg.WindowSize+j] - data.MinPrice) / rangeP
		}

		// RMSE da validação como base do intervalo de confiança
		confidence := res.RmseFinal
		if confidence < 0.01 { confidence = 0.01 }

		for d := 1; d <= forecastDays; d++ {
			predNorm, _ := m.Forward(window)
			predPrice := Desnormalizar(predNorm, data.MinPrice, data.MaxPrice)

			// Intervalo cresce com a raiz do dia (incerteza acumula)
			spread := confidence * math.Sqrt(float64(d))

			fp := ForecastPoint{
				Dia:     d,
				Predito: predPrice,
				Upper:   predPrice + spread,
				Lower:   predPrice - spread,
			}
			res.Forecast = append(res.Forecast, fp)

			if d == 1 {
				res.PredicaoAmanha = predPrice
			}

			// Deslizar janela: remover o mais antigo, adicionar predição
			window = append(window[1:], predNorm)
		}
	}

	return m, res
}

// Predizer faz forward pass para uma entrada e retorna preço desnormalizado
func Predizer(m *TimeSeriesMLP, input []float64, minP, maxP float64) float64 {
	output, _ := m.Forward(input)
	return Desnormalizar(output, minP, maxP)
}
