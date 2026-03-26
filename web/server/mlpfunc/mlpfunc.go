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
// Arquitetura: 1 entrada (x) -> 200 neuronios ocultos (tanh) -> 1 saida (tanh)
//
// Diferente dos MLPs anteriores que classificam padroes, este faz REGRESSAO:
// a saida eh um valor continuo, nao uma classe. O erro eh medido como
// 0.5 * sum((target - saida)^2) sobre todos os pontos.
// =============================================================================

// Hiperparametros do slide
const (
	NIn      = 1   // uma entrada: o valor x
	NHid     = 200 // 200 neuronios ocultos (slide usa neur = 200)
	NOut     = 1   // uma saida: o valor y aproximado
	alfa     = 0.005
	maxCiclo = 100000
	erroAlvo = 0.02
	nPontos  = 50 // 50 pontos igualmente espacados
)

// MLP armazena os pesos da rede.
// V: pesos entrada->oculta, V0: bias oculta
// W: pesos oculta->saida, W0: bias saida
type MLP struct {
	V  [NIn][NHid]float64  `json:"v"`
	V0 [NHid]float64       `json:"v0"`
	W  [NHid][NOut]float64 `json:"w"`
	W0 [NOut]float64       `json:"w0"`
}

// FuncPoint representa um ponto da funcao: x -> y (original) e yPred (predito pela rede)
type FuncPoint struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	YPred float64 `json:"yPred"`
}

// FuncStep eh enviado via SSE a cada N ciclos para atualizar o grafico
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

// PredictReq permite prever um ponto arbitrario
type PredictReq struct {
	X float64 `json:"x"`
}

// PredictResp retorna a predicao
type PredictResp struct {
	X     float64 `json:"x"`
	YPred float64 `json:"yPred"`
}

// ----- Funcoes matematicas disponiveis -----

// Funcao avalia a funcao alvo dado um nome
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

// FuncoesDisponiveis retorna a lista de funcoes que a rede pode aproximar
func FuncoesDisponiveis() []string {
	return []string{"sin(x)*sin(2x)", "sin(x)", "x^2", "x^3"}
}

// ----- Geracao do dataset -----

// gerarDataset cria N pontos igualmente espacados no intervalo [xmin, xmax]
// e calcula o valor da funcao para cada ponto.
// Corresponde ao trecho do slide:
//   x_orig = np.linspace(xmin, xmax, npontos)
//   t_orig = (np.sin(x)) * (np.sin(2*x))
func gerarDataset(funcao string, xmin, xmax float64, n int) ([]float64, []float64) {
	xs := make([]float64, n)
	ts := make([]float64, n)
	for i := 0; i < n; i++ {
		// linspace: pontos igualmente espacados
		xs[i] = xmin + float64(i)*(xmax-xmin)/float64(n-1)
		ts[i] = Funcao(funcao, xs[i])
	}
	return xs, ts
}

// ----- Inicializacao de pesos -----

// inicializar cria a rede com pesos aleatorios.
// V em [-1, 1] (aleatorio=1 no slide), W em [-0.2, 0.2] (aleatorio=0.2 no slide)
func inicializar() MLP {
	rng := rand.New(rand.NewSource(42))
	var m MLP

	// Pesos V: entrada -> oculta (aleatorio = 1)
	for i := 0; i < NIn; i++ {
		for j := 0; j < NHid; j++ {
			m.V[i][j] = rng.Float64()*2 - 1 // [-1, 1]
		}
	}
	// Bias oculta V0
	for j := 0; j < NHid; j++ {
		m.V0[j] = rng.Float64()*2 - 1 // [-1, 1]
	}
	// Pesos W: oculta -> saida (aleatorio = 0.2)
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOut; k++ {
			m.W[j][k] = rng.Float64()*0.4 - 0.2 // [-0.2, 0.2]
		}
	}
	// Bias saida W0
	for k := 0; k < NOut; k++ {
		m.W0[k] = rng.Float64()*0.4 - 0.2 // [-0.2, 0.2]
	}
	return m
}

// ----- Forward pass -----

// forward calcula a saida da rede para uma entrada x.
// Corresponde ao trecho do slide:
//   zin_j = np.dot(x[padrao,:], v[:,j]) + v0[0][j]
//   z_j = np.tanh(zin_j)
//   yin = np.dot(z_j, w) + w0
//   y = np.tanh(yin)
func forward(m MLP, x float64) (zj [NHid]float64, y float64) {
	// Camada oculta: zin_j = v0_j + x * v[0][j]
	for j := 0; j < NHid; j++ {
		zin := m.V0[j] + x*m.V[0][j]
		zj[j] = math.Tanh(zin)
	}
	// Camada de saida: yin = w0 + sum(z_j * w[j][0])
	yin := m.W0[0]
	for j := 0; j < NHid; j++ {
		yin += zj[j] * m.W[j][0]
	}
	y = math.Tanh(yin)
	return
}

// ----- Backward pass + atualizacao de pesos -----

// backwardAndUpdate calcula os deltas e atualiza os pesos in-place.
// Corresponde ao trecho do slide:
//   deltinha_k = (t_transp - y_transp) * (1 + y_transp) * (1 - y_transp)
//   deltaw = alfa * (np.dot(deltinha_k, z_j))
//   deltaw0 = alfa * deltinha_k
//   deltinhain_j = np.dot(np.transpose(deltinha_k), np.transpose(w))
//   deltinha_j = deltinhain_j * (1 + z_j) * (1 - z_j)
//   deltav = alfa * np.dot(np.transpose(deltinha_j), x_linhaTransp)
//   deltav0 = alfa * deltinha_j
func backwardAndUpdate(m *MLP, zj [NHid]float64, y, x, t float64) {
	// Derivada da tanh: f'(y) = (1+y)*(1-y)
	tanhDerivY := (1 + y) * (1 - y)

	// Delta da camada de saida
	deltaK := (t - y) * tanhDerivY

	// Propaga delta para camada oculta
	var deltaJ [NHid]float64
	for j := 0; j < NHid; j++ {
		deltaInJ := deltaK * m.W[j][0]
		tanhDerivZ := (1 + zj[j]) * (1 - zj[j])
		deltaJ[j] = deltaInJ * tanhDerivZ
	}

	// Atualiza pesos W (oculta -> saida)
	for j := 0; j < NHid; j++ {
		m.W[j][0] += alfa * deltaK * zj[j]
	}
	m.W0[0] += alfa * deltaK

	// Atualiza pesos V (entrada -> oculta)
	for j := 0; j < NHid; j++ {
		m.V[0][j] += alfa * deltaJ[j] * x
	}
	for j := 0; j < NHid; j++ {
		m.V0[j] += alfa * deltaJ[j]
	}
}

// ----- Treinamento -----

// Treinar executa o loop de treinamento e envia progresso via canal.
// A cada 100 ciclos, envia um FuncStep com os pontos preditos para visualizacao.
func Treinar(progressCh chan<- FuncStep, funcao string) FuncResult {
	xs, ts := gerarDataset(funcao, -1, 1, nPontos)
	m := inicializar()

	var res FuncResult
	res.Funcao = funcao

	// Loop principal — identico ao "while errotolerado < errototal" do slide
	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		erroTotal := 0.0

		// Para cada padrao (ponto da funcao)
		for padrao := 0; padrao < nPontos; padrao++ {
			// Forward
			zj, y := forward(m, xs[padrao])

			// Erro deste padrao: 0.5 * (t - y)^2
			d := ts[padrao] - y
			erroTotal += 0.5 * d * d

			// Backward + atualiza pesos
			backwardAndUpdate(&m, zj, y, xs[padrao], ts[padrao])
		}

		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		// Envia progresso a cada 100 ciclos para nao sobrecarregar o SSE
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

		// Criterio de parada
		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.Pontos = predizer(m, xs, ts)
			return res
		}
	}

	res.Convergiu = false
	res.Ciclos = maxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Pontos = predizer(m, xs, ts)
	return res
}

// predizer calcula a saida da rede para todos os pontos
func predizer(m MLP, xs, ts []float64) []FuncPoint {
	pontos := make([]FuncPoint, len(xs))
	for i := range xs {
		_, y := forward(m, xs[i])
		pontos[i] = FuncPoint{X: xs[i], Y: ts[i], YPred: y}
	}
	return pontos
}

// Predizer calcula a saida para um x arbitrario
func Predizer(m MLP, x float64) float64 {
	_, y := forward(m, x)
	return y
}
