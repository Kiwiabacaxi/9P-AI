package main

// =============================================================================
// Regra de Hebb — Trab 01
// =============================================================================

type HebbStep struct {
	Amostra int     `json:"amostra"`
	X1      int     `json:"x1"`
	X2      int     `json:"x2"`
	Target  int     `json:"target"`
	W1      float64 `json:"w1"`
	W2      float64 `json:"w2"`
	Bias    float64 `json:"bias"`
}

type HebbTest struct {
	X1       int     `json:"x1"`
	X2       int     `json:"x2"`
	Target   int     `json:"target"`
	YIn      float64 `json:"yIn"`
	Predicao int     `json:"predicao"`
	Acertou  bool    `json:"acertou"`
}

type HebbResult struct {
	Porta    string     `json:"porta"`
	W1       float64    `json:"w1"`
	W2       float64    `json:"w2"`
	Bias     float64    `json:"bias"`
	Acertos  int        `json:"acertos"`
	Acuracia float64    `json:"acuracia"`
	Steps    []HebbStep `json:"steps"`
	Testes   []HebbTest `json:"testes"`
}

func hebbBipolar(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

func hebbTreinar(p PortaLogica) HebbResult {
	w1, w2, bias := 0.0, 0.0, 0.0
	var steps []HebbStep

	for i := 0; i < 4; i++ {
		x1 := p.Inputs[i][0]
		x2 := p.Inputs[i][1]
		y := p.Targets[i]

		w1 += float64(x1 * y)
		w2 += float64(x2 * y)
		bias += float64(y)

		steps = append(steps, HebbStep{
			Amostra: i + 1,
			X1: x1, X2: x2, Target: y,
			W1: w1, W2: w2, Bias: bias,
		})
	}

	// Teste final
	var testes []HebbTest
	acertos := 0
	for i := 0; i < 4; i++ {
		x1, x2 := p.Inputs[i][0], p.Inputs[i][1]
		yIn := float64(x1)*w1 + float64(x2)*w2 + bias
		pred := hebbBipolar(yIn)
		ok := pred == p.Targets[i]
		if ok {
			acertos++
		}
		testes = append(testes, HebbTest{
			X1: x1, X2: x2, Target: p.Targets[i],
			YIn: yIn, Predicao: pred, Acertou: ok,
		})
	}

	return HebbResult{
		Porta:    p.Nome,
		W1:       w1,
		W2:       w2,
		Bias:     bias,
		Acertos:  acertos,
		Acuracia: float64(acertos) / 4.0 * 100.0,
		Steps:    steps,
		Testes:   testes,
	}
}
