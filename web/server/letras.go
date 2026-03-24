package main

import "math"

// =============================================================================
// MLP Letras — Reconhecimento A–Z (35 entradas, 15 ocultos, 26 saídas)
// Lógica extraída de Multilayer Perceptron (MLP) Letras/main.go
// =============================================================================

const (
	ltrNLinhas  = 7
	ltrNColunas = 5
	ltrNIn      = 35
	ltrNHid     = 15
	ltrNOut     = 26
	ltrAlfa     = 0.01
	ltrMaxCiclo = 50000
	ltrErroAlvo = 0.5
)

var ltrNomes = [ltrNOut]string{
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
	"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
}

type LtrMLP struct {
	V  [ltrNIn][ltrNHid]float64
	V0 [ltrNHid]float64
	W  [ltrNHid][ltrNOut]float64
	W0 [ltrNOut]float64
}

type LtrForward struct {
	Z [ltrNHid]float64
	Y [ltrNOut]float64
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

type LtrClassifyReq struct {
	Grade [ltrNIn]float64 `json:"grade"`
}

type LtrClassifyResp struct {
	LetraIdx int       `json:"letraIdx"`
	Letra    string    `json:"letra"`
	Scores   [ltrNOut]float64 `json:"scores"`
	Top5     []LtrCandidate   `json:"top5"`
}

type LtrCandidate struct {
	Letra string  `json:"letra"`
	Score float64 `json:"score"`
	Idx   int     `json:"idx"`
}

func ltrTanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func ltrForward(m LtrMLP, x [ltrNIn]float64) LtrForward {
	var f LtrForward
	var zin [ltrNHid]float64
	for j := 0; j < ltrNHid; j++ {
		zin[j] = m.V0[j]
		for i := 0; i < ltrNIn; i++ {
			zin[j] += x[i] * m.V[i][j]
		}
		f.Z[j] = math.Tanh(zin[j])
	}
	for k := 0; k < ltrNOut; k++ {
		yin := m.W0[k]
		for j := 0; j < ltrNHid; j++ {
			yin += f.Z[j] * m.W[j][k]
		}
		f.Y[k] = math.Tanh(yin)
	}
	return f
}

func ltrBackwardAndUpdate(m LtrMLP, f LtrForward, target [ltrNOut]float64, x [ltrNIn]float64) LtrMLP {
	var deltaK [ltrNOut]float64
	for k := 0; k < ltrNOut; k++ {
		deltaK[k] = (target[k] - f.Y[k]) * ltrTanhDeriv(f.Y[k])
	}
	var deltaJ [ltrNHid]float64
	for j := 0; j < ltrNHid; j++ {
		var s float64
		for k := 0; k < ltrNOut; k++ {
			s += deltaK[k] * m.W[j][k]
		}
		deltaJ[j] = s * ltrTanhDeriv(f.Z[j])
	}
	for j := 0; j < ltrNHid; j++ {
		for k := 0; k < ltrNOut; k++ {
			m.W[j][k] += ltrAlfa * deltaK[k] * f.Z[j]
		}
	}
	for k := 0; k < ltrNOut; k++ {
		m.W0[k] += ltrAlfa * deltaK[k]
	}
	for i := 0; i < ltrNIn; i++ {
		for j := 0; j < ltrNHid; j++ {
			m.V[i][j] += ltrAlfa * deltaJ[j] * x[i]
		}
	}
	for j := 0; j < ltrNHid; j++ {
		m.V0[j] += ltrAlfa * deltaJ[j]
	}
	return m
}

func ltrCalcErro(y, t [ltrNOut]float64) float64 {
	var e float64
	for k := 0; k < ltrNOut; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

func ltrTarget(idx int) [ltrNOut]float64 {
	var t [ltrNOut]float64
	for i := range t {
		t[i] = -1.0
	}
	t[idx] = 1.0
	return t
}

func ltrClassificar(m LtrMLP, x [ltrNIn]float64) LtrClassifyResp {
	f := ltrForward(m, x)
	best := 0
	for k := 1; k < ltrNOut; k++ {
		if f.Y[k] > f.Y[best] {
			best = k
		}
	}

	// top-5
	scores := f.Y
	var top5 []LtrCandidate
	for i := 0; i < 5; i++ {
		b := 0
		for k := 1; k < ltrNOut; k++ {
			if scores[k] > scores[b] {
				b = k
			}
		}
		top5 = append(top5, LtrCandidate{Letra: ltrNomes[b], Score: scores[b], Idx: b})
		scores[b] = -999.0
	}

	return LtrClassifyResp{
		LetraIdx: best,
		Letra:    ltrNomes[best],
		Scores:   f.Y,
		Top5:     top5,
	}
}

// ltrTreinar treina a rede e envia steps pelo canal progressCh para SSE.
// Retorna o resultado final e o modelo treinado.
func ltrTreinar(progressCh chan<- LtrStep) (LtrResult, LtrMLP) {
	dataset := ltrDataset()

	// inicializar pesos aleatórios usando seed fixa para reprodutibilidade
	m := ltrInicializar()

	var res LtrResult
	var erroHistorico []float64
	const maxSteps = 200

	for ciclo := 1; ciclo <= ltrMaxCiclo; ciclo++ {
		erroTotal := 0.0
		for letraIdx := 0; letraIdx < ltrNOut; letraIdx++ {
			x := dataset[letraIdx]
			target := ltrTarget(letraIdx)
			f := ltrForward(m, x)
			m = ltrBackwardAndUpdate(m, f, target, x)
			erroTotal += ltrCalcErro(f.Y, target)

			if len(res.Steps) < maxSteps {
				step := LtrStep{
					Ciclo:     ciclo,
					LetraIdx:  letraIdx,
					Letra:     ltrNomes[letraIdx],
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

		if erroTotal <= ltrErroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.ErroHistorico = erroHistorico
			break
		}
	}

	if !res.Convergiu {
		res.Ciclos = ltrMaxCiclo
		res.ErroFinal = erroHistorico[len(erroHistorico)-1]
		res.ErroHistorico = erroHistorico
	}

	// acurácia
	for i := 0; i < ltrNOut; i++ {
		r := ltrClassificar(m, dataset[i])
		if r.LetraIdx == i {
			res.Acertos++
		}
	}
	res.Total = ltrNOut
	res.Acuracia = float64(res.Acertos) / float64(ltrNOut) * 100.0

	return res, m
}
