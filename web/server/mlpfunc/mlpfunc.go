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
// Arquitetura: 1 entrada (x) -> N neuronios ocultos (tanh) -> 1 saida (tanh)
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
	Funcao   string  `json:"funcao"`
	NHid     int     `json:"nHid"`
	Alfa     float64 `json:"alfa"`
	MaxCiclo int     `json:"maxCiclo"`
}

// DefaultConfig retorna a configuracao padrao (valores do slide)
func DefaultConfig() Config {
	return Config{
		Funcao:   "sin(x)*sin(2x)",
		NHid:     200,
		Alfa:     0.005,
		MaxCiclo: 100000,
	}
}

// MLP armazena os pesos da rede com tamanho dinamico.
type MLP struct {
	nHid int
	V    []float64 // [nIn * nHid] = [nHid] (nIn=1)
	V0   []float64 // [nHid]
	W    []float64 // [nHid * nOut] = [nHid] (nOut=1)
	W0   []float64 // [nOut] = [1]
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

func inicializar(nHid int) MLP {
	rng := rand.New(rand.NewSource(42))
	m := MLP{
		nHid: nHid,
		V:    make([]float64, nHid),
		V0:   make([]float64, nHid),
		W:    make([]float64, nHid),
		W0:   make([]float64, 1),
	}

	// V: entrada -> oculta (aleatorio = 1)
	for j := 0; j < nHid; j++ {
		m.V[j] = rng.Float64()*2 - 1 // [-1, 1]
	}
	// Bias oculta V0
	for j := 0; j < nHid; j++ {
		m.V0[j] = rng.Float64()*2 - 1
	}
	// W: oculta -> saida (aleatorio = 0.2)
	for j := 0; j < nHid; j++ {
		m.W[j] = rng.Float64()*0.4 - 0.2 // [-0.2, 0.2]
	}
	// Bias saida W0
	m.W0[0] = rng.Float64()*0.4 - 0.2

	return m
}

// ----- Forward pass -----

func forward(m MLP, x float64) (zj []float64, y float64) {
	zj = make([]float64, m.nHid)
	// Camada oculta
	for j := 0; j < m.nHid; j++ {
		zin := m.V0[j] + x*m.V[j]
		zj[j] = math.Tanh(zin)
	}
	// Camada de saida
	yin := m.W0[0]
	for j := 0; j < m.nHid; j++ {
		yin += zj[j] * m.W[j]
	}
	y = math.Tanh(yin)
	return
}

// ----- Backward pass + atualizacao de pesos -----

func backwardAndUpdate(m *MLP, zj []float64, y, x, t, alfa float64) {
	tanhDerivY := (1 + y) * (1 - y)
	deltaK := (t - y) * tanhDerivY

	deltaJ := make([]float64, m.nHid)
	for j := 0; j < m.nHid; j++ {
		deltaInJ := deltaK * m.W[j]
		tanhDerivZ := (1 + zj[j]) * (1 - zj[j])
		deltaJ[j] = deltaInJ * tanhDerivZ
	}

	// Atualiza W (oculta -> saida)
	for j := 0; j < m.nHid; j++ {
		m.W[j] += alfa * deltaK * zj[j]
	}
	m.W0[0] += alfa * deltaK

	// Atualiza V (entrada -> oculta)
	for j := 0; j < m.nHid; j++ {
		m.V[j] += alfa * deltaJ[j] * x
	}
	for j := 0; j < m.nHid; j++ {
		m.V0[j] += alfa * deltaJ[j]
	}
}

// ----- Treinamento -----

func Treinar(progressCh chan<- FuncStep, cfg Config) FuncResult {
	if cfg.NHid <= 0 {
		cfg = DefaultConfig()
		cfg.Funcao = cfg.Funcao
	}

	xs, ts := gerarDataset(cfg.Funcao, -1, 1, nPontos)
	m := inicializar(cfg.NHid)

	var res FuncResult
	res.Funcao = cfg.Funcao

	erroAlvo := 0.02

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		erroTotal := 0.0

		for padrao := 0; padrao < nPontos; padrao++ {
			zj, y := forward(m, xs[padrao])
			d := ts[padrao] - y
			erroTotal += 0.5 * d * d
			backwardAndUpdate(&m, zj, y, xs[padrao], ts[padrao], cfg.Alfa)
		}

		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		if progressCh != nil && ciclo%100 == 0 {
			pontos := predizer(m, xs, ts)
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
			res.Pontos = predizer(m, xs, ts)
			return res
		}
	}

	res.Convergiu = false
	res.Ciclos = cfg.MaxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Pontos = predizer(m, xs, ts)
	return res
}

func predizer(m MLP, xs, ts []float64) []FuncPoint {
	pontos := make([]FuncPoint, len(xs))
	for i := range xs {
		_, y := forward(m, xs[i])
		pontos[i] = FuncPoint{X: xs[i], Y: ts[i], YPred: y}
	}
	return pontos
}
