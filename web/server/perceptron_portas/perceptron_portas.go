package perceptronportas

import "math/rand"

// =============================================================================
// Perceptron — Portas Lógicas (Trab 02 PT2)
// =============================================================================

const alfa = 0.01
const maxCiclos = 1000

type PercPortasStep struct {
	Ciclo    int     `json:"ciclo"`
	Amostra  int     `json:"amostra"`
	X1       int     `json:"x1"`
	X2       int     `json:"x2"`
	Target   int     `json:"target"`
	YLiq     float64 `json:"yLiq"`
	Y        int     `json:"y"`
	TeveErro bool    `json:"teveErro"`
	W1       float64 `json:"w1"`
	W2       float64 `json:"w2"`
	Bias     float64 `json:"bias"`
}

type PercPortasTest struct {
	X1       int     `json:"x1"`
	X2       int     `json:"x2"`
	Target   int     `json:"target"`
	Predicao int     `json:"predicao"`
	YLiq     float64 `json:"yLiq"`
	Acertou  bool    `json:"acertou"`
}

type PercPortasResult struct {
	Porta     string           `json:"porta"`
	Convergiu bool             `json:"convergiu"`
	Ciclos    int              `json:"ciclos"`
	W1        float64          `json:"w1"`
	W2        float64          `json:"w2"`
	Bias      float64          `json:"bias"`
	Acertos   int              `json:"acertos"`
	Acuracia  float64          `json:"acuracia"`
	Steps     []PercPortasStep `json:"steps"`
	Testes    []PercPortasTest `json:"testes"`
}

func bipolar(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

// Treinar executa o algoritmo Perceptron para uma porta lógica.
func Treinar(p PortaLogica) PercPortasResult {
	rng := rand.New(rand.NewSource(42))
	w1 := rng.Float64() - 0.5
	w2 := rng.Float64() - 0.5
	bias := rng.Float64() - 0.5

	var steps []PercPortasStep
	convergiu := false
	ciclo := 0

	for ciclo < maxCiclos {
		ciclo++
		errou := false

		for i := 0; i < 4; i++ {
			x1 := p.Inputs[i][0]
			x2 := p.Inputs[i][1]
			yLiq := w1*float64(x1) + w2*float64(x2) + bias
			y := bipolar(yLiq)
			teveErro := false

			if y != p.Targets[i] {
				errou = true
				teveErro = true
				delta := alfa * float64(p.Targets[i]-y)
				w1 += delta * float64(x1)
				w2 += delta * float64(x2)
				bias += delta
			}

			steps = append(steps, PercPortasStep{
				Ciclo: ciclo, Amostra: i + 1,
				X1: x1, X2: x2, Target: p.Targets[i],
				YLiq: yLiq, Y: y, TeveErro: teveErro,
				W1: w1, W2: w2, Bias: bias,
			})
		}

		if !errou {
			convergiu = true
			break
		}
	}

	var testes []PercPortasTest
	acertos := 0
	for i := 0; i < 4; i++ {
		x1, x2 := p.Inputs[i][0], p.Inputs[i][1]
		yLiq := w1*float64(x1) + w2*float64(x2) + bias
		pred := bipolar(yLiq)
		ok := pred == p.Targets[i]
		if ok {
			acertos++
		}
		testes = append(testes, PercPortasTest{
			X1: x1, X2: x2, Target: p.Targets[i],
			Predicao: pred, YLiq: yLiq, Acertou: ok,
		})
	}

	return PercPortasResult{
		Porta: p.Nome, Convergiu: convergiu, Ciclos: ciclo,
		W1: w1, W2: w2, Bias: bias,
		Acertos: acertos, Acuracia: float64(acertos) / 4.0 * 100.0,
		Steps: steps, Testes: testes,
	}
}
