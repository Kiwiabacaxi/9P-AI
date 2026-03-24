package perceptronletras

import "math/rand"

// =============================================================================
// Perceptron — Letras A/B (Trab 02 PT1)
// =============================================================================

const alfa = 0.01
const maxCiclos = 10000

type PercLetrasStep struct {
	Ciclo    int     `json:"ciclo"`
	Amostra  string  `json:"amostra"`
	Target   int     `json:"target"`
	YLiq     float64 `json:"yLiq"`
	Y        int     `json:"y"`
	Delta    float64 `json:"delta"`
	NovoBias float64 `json:"novoBias"`
	TeveErro bool    `json:"teveErro"`
}

type PercLetrasTest struct {
	Letra    string  `json:"letra"`
	Target   int     `json:"target"`
	Predicao int     `json:"predicao"`
	YLiq     float64 `json:"yLiq"`
	Acertou  bool    `json:"acertou"`
}

type PercLetrasResult struct {
	Convergiu bool             `json:"convergiu"`
	Ciclos    int              `json:"ciclos"`
	Bias      float64          `json:"bias"`
	Acertos   int              `json:"acertos"`
	Acuracia  float64          `json:"acuracia"`
	Steps     []PercLetrasStep `json:"steps"`
	Testes    []PercLetrasTest `json:"testes"`
}

type DatasetResp struct {
	Letra string      `json:"letra"`
	Grade [NIn]float64 `json:"grade"`
}

func bipolar(yLiq float64) int {
	if yLiq >= 0 {
		return 1
	}
	return -1
}

// Treinar executa o algoritmo Perceptron para classificação das letras A e B.
func Treinar() PercLetrasResult {
	rng := rand.New(rand.NewSource(42))

	amostras := [2][NIn]float64{LetraA(), LetraB()}
	targets := [2]int{-1, 1}
	nomes := [2]string{"A", "B"}

	var pesos [NIn]float64
	for i := range pesos {
		pesos[i] = rng.Float64() - 0.5
	}
	bias := rng.Float64() - 0.5

	var steps []PercLetrasStep
	ciclo := 0

	for ciclo < maxCiclos {
		ciclo++
		condErro := false

		for s := 0; s < 2; s++ {
			yLiq := bias
			for i := 0; i < NIn; i++ {
				yLiq += amostras[s][i] * pesos[i]
			}
			y := bipolar(yLiq)
			teveErro := false
			var delta float64

			if y != targets[s] {
				condErro = true
				teveErro = true
				delta = alfa * float64(targets[s]-y)
				for i := 0; i < NIn; i++ {
					pesos[i] += delta * amostras[s][i]
				}
				bias += delta
			}

			steps = append(steps, PercLetrasStep{
				Ciclo:    ciclo,
				Amostra:  nomes[s],
				Target:   targets[s],
				YLiq:     yLiq,
				Y:        y,
				Delta:    delta,
				NovoBias: bias,
				TeveErro: teveErro,
			})
		}

		if !condErro {
			break
		}
	}

	var testes []PercLetrasTest
	acertos := 0
	for s := 0; s < 2; s++ {
		yLiq := bias
		for i := 0; i < NIn; i++ {
			yLiq += amostras[s][i] * pesos[i]
		}
		pred := bipolar(yLiq)
		ok := pred == targets[s]
		if ok {
			acertos++
		}
		testes = append(testes, PercLetrasTest{
			Letra:    nomes[s],
			Target:   targets[s],
			Predicao: pred,
			YLiq:     yLiq,
			Acertou:  ok,
		})
	}

	return PercLetrasResult{
		Convergiu: ciclo < maxCiclos,
		Ciclos:    ciclo,
		Bias:      bias,
		Acertos:   acertos,
		Acuracia:  float64(acertos) / 2.0 * 100.0,
		Steps:     steps,
		Testes:    testes,
	}
}
