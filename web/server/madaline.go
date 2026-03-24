package main

import "math/rand"

// =============================================================================
// MADALINE — Letras A–M (Trab 03)
// =============================================================================

const madAlfa = 0.01
const madMaxCiclos = 10000

type ADALINEUnit struct {
	Pesos [madNIn]float64
	Bias  float64
}

type MadNet struct {
	Unidades [madNLetras]ADALINEUnit
}

type MadStep struct {
	Ciclo    int                  `json:"ciclo"`
	LetraIdx int                  `json:"letraIdx"`
	Letra    string               `json:"letra"`
	YIn      [madNLetras]float64  `json:"yIn"`
	Y        [madNLetras]int      `json:"y"`
	Target   [madNLetras]int      `json:"target"`
	Erros    [madNLetras]bool     `json:"erros"`
}

type MadResult struct {
	Convergiu bool      `json:"convergiu"`
	Ciclos    int       `json:"ciclos"`
	Steps     []MadStep `json:"steps"`
	Acertos   int       `json:"acertos"`
	Total     int       `json:"total"`
	Acuracia  float64   `json:"acuracia"`
}

type MadClassifyReq struct {
	Grade [madNIn]float64 `json:"grade"`
}

type MadCandidate struct {
	Letra string  `json:"letra"`
	Idx   int     `json:"idx"`
	Score float64 `json:"score"`
}

type MadClassifyResp struct {
	LetraIdx int            `json:"letraIdx"`
	Letra    string         `json:"letra"`
	Scores   [madNLetras]float64 `json:"scores"`
	Top5     []MadCandidate `json:"top5"`
}

func madBipolar(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}

func madCalcYIn(u ADALINEUnit, entrada [madNIn]int) float64 {
	soma := u.Bias
	for i := 0; i < madNIn; i++ {
		soma += u.Pesos[i] * float64(entrada[i])
	}
	return soma
}

func madCalcYInFloat(u ADALINEUnit, entrada [madNIn]float64) float64 {
	soma := u.Bias
	for i := 0; i < madNIn; i++ {
		soma += u.Pesos[i] * entrada[i]
	}
	return soma
}

func madTarget(idx int) [madNLetras]int {
	var t [madNLetras]int
	for i := range t {
		t[i] = -1
	}
	t[idx] = 1
	return t
}

func madTreinar(progressCh chan<- MadStep) (MadResult, MadNet) {
	rng := rand.New(rand.NewSource(42))
	var rede MadNet
	for j := 0; j < madNLetras; j++ {
		for i := 0; i < madNIn; i++ {
			rede.Unidades[j].Pesos[i] = rng.Float64() - 0.5
		}
		rede.Unidades[j].Bias = rng.Float64() - 0.5
	}

	dataset := madDataset()
	var steps []MadStep
	const maxSteps = 300
	convergiu := false
	ciclosReais := madMaxCiclos

	for ciclo := 1; ciclo <= madMaxCiclos; ciclo++ {
		erroNoCiclo := false

		for letraIdx := 0; letraIdx < madNLetras; letraIdx++ {
			entrada := dataset[letraIdx]
			target := madTarget(letraIdx)

			var yIn [madNLetras]float64
			var y [madNLetras]int
			var erros [madNLetras]bool

			for j := 0; j < madNLetras; j++ {
				yIn[j] = madCalcYIn(rede.Unidades[j], entrada)
				y[j] = madBipolar(yIn[j])

				if y[j] != target[j] {
					delta := madAlfa * float64(target[j]-y[j])
					for i := 0; i < madNIn; i++ {
						rede.Unidades[j].Pesos[i] += delta * float64(entrada[i])
					}
					rede.Unidades[j].Bias += delta
					erros[j] = true
					erroNoCiclo = true
				}
			}

			hasErro := false
			for _, e := range erros {
				if e {
					hasErro = true
					break
				}
			}
			if hasErro && len(steps) < maxSteps {
				step := MadStep{
					Ciclo: ciclo, LetraIdx: letraIdx,
					Letra: madNomes[letraIdx],
					YIn: yIn, Y: y, Target: target, Erros: erros,
				}
				steps = append(steps, step)
				if progressCh != nil {
					select {
					case progressCh <- step:
					default:
					}
				}
			}
		}

		if !erroNoCiclo {
			convergiu = true
			ciclosReais = ciclo
			break
		}
	}

	// acurácia
	acertos := 0
	for i := 0; i < madNLetras; i++ {
		pred, _ := madClassificarInt(rede, dataset[i])
		if pred == i {
			acertos++
		}
	}

	res := MadResult{
		Convergiu: convergiu,
		Ciclos:    ciclosReais,
		Steps:     steps,
		Acertos:   acertos,
		Total:     madNLetras,
		Acuracia:  float64(acertos) / float64(madNLetras) * 100.0,
	}
	return res, rede
}

func madClassificarInt(rede MadNet, entrada [madNIn]int) (int, [madNLetras]float64) {
	var yIns [madNLetras]float64
	for j := 0; j < madNLetras; j++ {
		yIns[j] = madCalcYIn(rede.Unidades[j], entrada)
	}
	best := 0
	for j := 1; j < madNLetras; j++ {
		if yIns[j] > yIns[best] {
			best = j
		}
	}
	return best, yIns
}

func madClassificar(rede MadNet, grade [madNIn]float64) MadClassifyResp {
	var yIns [madNLetras]float64
	for j := 0; j < madNLetras; j++ {
		yIns[j] = madCalcYInFloat(rede.Unidades[j], grade)
	}
	best := 0
	for j := 1; j < madNLetras; j++ {
		if yIns[j] > yIns[best] {
			best = j
		}
	}

	scores := yIns
	var top5 []MadCandidate
	for i := 0; i < 5; i++ {
		b := 0
		for j := 1; j < madNLetras; j++ {
			if scores[j] > scores[b] {
				b = j
			}
		}
		top5 = append(top5, MadCandidate{Letra: madNomes[b], Idx: b, Score: scores[b]})
		scores[b] = -999.0
	}

	return MadClassifyResp{
		LetraIdx: best,
		Letra:    madNomes[best],
		Scores:   yIns,
		Top5:     top5,
	}
}
