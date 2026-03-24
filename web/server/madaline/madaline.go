package madaline

import "math/rand"

// =============================================================================
// MADALINE — Letras A–M (Trab 03)
// =============================================================================

const alfa = 0.01
const maxCiclos = 10000

type ADALINEUnit struct {
	Pesos [NIn]float64
	Bias  float64
}

type MadNet struct {
	Unidades [NLetras]ADALINEUnit
}

type MadStep struct {
	Ciclo    int                `json:"ciclo"`
	LetraIdx int                `json:"letraIdx"`
	Letra    string             `json:"letra"`
	YIn      [NLetras]float64   `json:"yIn"`
	Y        [NLetras]int       `json:"y"`
	Target   [NLetras]int       `json:"target"`
	Erros    [NLetras]bool      `json:"erros"`
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
	Grade [NIn]float64 `json:"grade"`
}

type MadCandidate struct {
	Letra string  `json:"letra"`
	Idx   int     `json:"idx"`
	Score float64 `json:"score"`
}

type MadClassifyResp struct {
	LetraIdx int              `json:"letraIdx"`
	Letra    string           `json:"letra"`
	Scores   [NLetras]float64 `json:"scores"`
	Top5     []MadCandidate   `json:"top5"`
}

func bipolar(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}

func calcYIn(u ADALINEUnit, entrada [NIn]int) float64 {
	soma := u.Bias
	for i := 0; i < NIn; i++ {
		soma += u.Pesos[i] * float64(entrada[i])
	}
	return soma
}

func calcYInFloat(u ADALINEUnit, entrada [NIn]float64) float64 {
	soma := u.Bias
	for i := 0; i < NIn; i++ {
		soma += u.Pesos[i] * entrada[i]
	}
	return soma
}

func target(idx int) [NLetras]int {
	var t [NLetras]int
	for i := range t {
		t[i] = -1
	}
	t[idx] = 1
	return t
}

// Treinar treina a rede MADALINE e envia steps pelo canal progressCh para SSE.
func Treinar(progressCh chan<- MadStep) (MadResult, MadNet) {
	rng := rand.New(rand.NewSource(42))
	var rede MadNet
	for j := 0; j < NLetras; j++ {
		for i := 0; i < NIn; i++ {
			rede.Unidades[j].Pesos[i] = rng.Float64() - 0.5
		}
		rede.Unidades[j].Bias = rng.Float64() - 0.5
	}

	dataset := Dataset()
	var steps []MadStep
	const maxSteps = 300
	convergiu := false
	ciclosReais := maxCiclos

	for ciclo := 1; ciclo <= maxCiclos; ciclo++ {
		erroNoCiclo := false

		for letraIdx := 0; letraIdx < NLetras; letraIdx++ {
			entrada := dataset[letraIdx]
			tgt := target(letraIdx)

			var yIn [NLetras]float64
			var y [NLetras]int
			var erros [NLetras]bool

			for j := 0; j < NLetras; j++ {
				yIn[j] = calcYIn(rede.Unidades[j], entrada)
				y[j] = bipolar(yIn[j])

				if y[j] != tgt[j] {
					delta := alfa * float64(tgt[j]-y[j])
					for i := 0; i < NIn; i++ {
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
					Letra: Nomes[letraIdx],
					YIn: yIn, Y: y, Target: tgt, Erros: erros,
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
	for i := 0; i < NLetras; i++ {
		pred, _ := classificarInt(rede, dataset[i])
		if pred == i {
			acertos++
		}
	}

	res := MadResult{
		Convergiu: convergiu,
		Ciclos:    ciclosReais,
		Steps:     steps,
		Acertos:   acertos,
		Total:     NLetras,
		Acuracia:  float64(acertos) / float64(NLetras) * 100.0,
	}
	return res, rede
}

func classificarInt(rede MadNet, entrada [NIn]int) (int, [NLetras]float64) {
	var yIns [NLetras]float64
	for j := 0; j < NLetras; j++ {
		yIns[j] = calcYIn(rede.Unidades[j], entrada)
	}
	best := 0
	for j := 1; j < NLetras; j++ {
		if yIns[j] > yIns[best] {
			best = j
		}
	}
	return best, yIns
}

// Classificar classifica uma entrada float64 e retorna a resposta com top5.
func Classificar(rede MadNet, grade [NIn]float64) MadClassifyResp {
	var yIns [NLetras]float64
	for j := 0; j < NLetras; j++ {
		yIns[j] = calcYInFloat(rede.Unidades[j], grade)
	}
	best := 0
	for j := 1; j < NLetras; j++ {
		if yIns[j] > yIns[best] {
			best = j
		}
	}

	scores := yIns
	var top5 []MadCandidate
	for i := 0; i < 5; i++ {
		b := 0
		for j := 1; j < NLetras; j++ {
			if scores[j] > scores[b] {
				b = j
			}
		}
		top5 = append(top5, MadCandidate{Letra: Nomes[b], Idx: b, Score: scores[b]})
		scores[b] = -999.0
	}

	return MadClassifyResp{
		LetraIdx: best,
		Letra:    Nomes[best],
		Scores:   yIns,
		Top5:     top5,
	}
}
