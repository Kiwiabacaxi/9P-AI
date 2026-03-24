package letras

import "math"

// =============================================================================
// MLP Letras — Reconhecimento A–Z (35 entradas, 15 ocultos, 26 saídas)
// Lógica extraída de Multilayer Perceptron (MLP) Letras/main.go
// =============================================================================

const (
	NHid     = 15
	alfa     = 0.01
	maxCiclo = 50000
	erroAlvo = 0.5
)

type LtrMLP struct {
	V  [NIn][NHid]float64
	V0 [NHid]float64
	W  [NHid][NOut]float64
	W0 [NOut]float64
}

type LtrForward struct {
	Z [NHid]float64
	Y [NOut]float64
}

type LtrStep struct {
	Ciclo     int     `json:"ciclo"`
	LetraIdx  int     `json:"letraIdx"`
	Letra     string  `json:"letra"`
	ErroTotal float64 `json:"erroTotal"`
}

type LtrResult struct {
	Convergiu     bool      `json:"convergiu"`
	Ciclos        int       `json:"ciclos"`
	ErroFinal     float64   `json:"erroFinal"`
	ErroHistorico []float64 `json:"erroHistorico"`
	Steps         []LtrStep `json:"steps"`
	Acertos       int       `json:"acertos"`
	Total         int       `json:"total"`
	Acuracia      float64   `json:"acuracia"`
}

type ClassifyReq struct {
	Grade [NIn]float64 `json:"grade"`
}

type ClassifyResp struct {
	LetraIdx int              `json:"letraIdx"`
	Letra    string           `json:"letra"`
	Scores   [NOut]float64    `json:"scores"`
	Top5     []LtrCandidate   `json:"top5"`
}

type LtrCandidate struct {
	Letra string  `json:"letra"`
	Score float64 `json:"score"`
	Idx   int     `json:"idx"`
}

func tanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func ltrForward(m LtrMLP, x [NIn]float64) LtrForward {
	var f LtrForward
	var zin [NHid]float64
	for j := 0; j < NHid; j++ {
		zin[j] = m.V0[j]
		for i := 0; i < NIn; i++ {
			zin[j] += x[i] * m.V[i][j]
		}
		f.Z[j] = math.Tanh(zin[j])
	}
	for k := 0; k < NOut; k++ {
		yin := m.W0[k]
		for j := 0; j < NHid; j++ {
			yin += f.Z[j] * m.W[j][k]
		}
		f.Y[k] = math.Tanh(yin)
	}
	return f
}

func backwardAndUpdate(m LtrMLP, f LtrForward, target [NOut]float64, x [NIn]float64) LtrMLP {
	var deltaK [NOut]float64
	for k := 0; k < NOut; k++ {
		deltaK[k] = (target[k] - f.Y[k]) * tanhDeriv(f.Y[k])
	}
	var deltaJ [NHid]float64
	for j := 0; j < NHid; j++ {
		var s float64
		for k := 0; k < NOut; k++ {
			s += deltaK[k] * m.W[j][k]
		}
		deltaJ[j] = s * tanhDeriv(f.Z[j])
	}
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOut; k++ {
			m.W[j][k] += alfa * deltaK[k] * f.Z[j]
		}
	}
	for k := 0; k < NOut; k++ {
		m.W0[k] += alfa * deltaK[k]
	}
	for i := 0; i < NIn; i++ {
		for j := 0; j < NHid; j++ {
			m.V[i][j] += alfa * deltaJ[j] * x[i]
		}
	}
	for j := 0; j < NHid; j++ {
		m.V0[j] += alfa * deltaJ[j]
	}
	return m
}

func calcErro(y, t [NOut]float64) float64 {
	var e float64
	for k := 0; k < NOut; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

func ltrTarget(idx int) [NOut]float64 {
	var t [NOut]float64
	for i := range t {
		t[i] = -1.0
	}
	t[idx] = 1.0
	return t
}

// Classificar classifica uma entrada e retorna a resposta com top5.
func Classificar(m LtrMLP, x [NIn]float64) ClassifyResp {
	f := ltrForward(m, x)
	best := 0
	for k := 1; k < NOut; k++ {
		if f.Y[k] > f.Y[best] {
			best = k
		}
	}

	// top-5
	scores := f.Y
	var top5 []LtrCandidate
	for i := 0; i < 5; i++ {
		b := 0
		for k := 1; k < NOut; k++ {
			if scores[k] > scores[b] {
				b = k
			}
		}
		top5 = append(top5, LtrCandidate{Letra: Nomes[b], Score: scores[b], Idx: b})
		scores[b] = -999.0
	}

	return ClassifyResp{
		LetraIdx: best,
		Letra:    Nomes[best],
		Scores:   f.Y,
		Top5:     top5,
	}
}

// Treinar treina a rede e envia steps pelo canal progressCh para SSE.
// Retorna o resultado final e o modelo treinado.
func Treinar(progressCh chan<- LtrStep) (LtrResult, LtrMLP) {
	dataset := Dataset()

	// inicializar pesos aleatórios usando seed fixa para reprodutibilidade
	m := Inicializar(NHid)

	var res LtrResult
	var erroHistorico []float64
	const maxSteps = 200

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		erroTotal := 0.0
		for letraIdx := 0; letraIdx < NOut; letraIdx++ {
			x := dataset[letraIdx]
			target := ltrTarget(letraIdx)
			f := ltrForward(m, x)
			m = backwardAndUpdate(m, f, target, x)
			erroTotal += calcErro(f.Y, target)

			if len(res.Steps) < maxSteps {
				step := LtrStep{
					Ciclo:     ciclo,
					LetraIdx:  letraIdx,
					Letra:     Nomes[letraIdx],
					ErroTotal: erroTotal,
				}
				res.Steps = append(res.Steps, step)
				if progressCh != nil {
					select {
					case progressCh <- step:
					default:
					}
				}
			}
		}
		erroHistorico = append(erroHistorico, erroTotal)

		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.ErroHistorico = erroHistorico
			break
		}
	}

	if !res.Convergiu {
		res.Ciclos = maxCiclo
		res.ErroFinal = erroHistorico[len(erroHistorico)-1]
		res.ErroHistorico = erroHistorico
	}

	// acurácia
	for i := 0; i < NOut; i++ {
		r := Classificar(m, dataset[i])
		if r.LetraIdx == i {
			res.Acertos++
		}
	}
	res.Total = NOut
	res.Acuracia = float64(res.Acertos) / float64(NOut) * 100.0

	return res, m
}
