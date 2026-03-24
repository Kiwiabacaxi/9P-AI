package mlp

import "math"

// =============================================================================
// MLP — Desafio (3 entradas, 2 ocultos, 3 saídas)
// Lógica extraída de Desafio Multilayer Perceptron (MLP)/main.go
// =============================================================================

const (
	NIn      = 3
	NHid     = 2
	NOut     = 3
	alfa     = 0.01
	maxCiclo = 50000
	erroAlvo = 0.001
)

type MLP struct {
	V  [NIn][NHid]float64
	V0 [NHid]float64
	W  [NHid][NOut]float64
	W0 [NOut]float64
}

type MLPForward struct {
	Zin [NHid]float64
	Z   [NHid]float64
	Yin [NOut]float64
	Y   [NOut]float64
}

type MLPBackward struct {
	DeltaK   [NOut]float64
	DeltaInJ [NHid]float64
	DeltaJ   [NHid]float64
	DeltaW   [NHid][NOut]float64
	DeltaW0  [NOut]float64
	DeltaV   [NIn][NHid]float64
	DeltaV0  [NHid]float64
}

type MLPStep struct {
	Ciclo     int              `json:"ciclo"`
	Padrao    int              `json:"padrao"`
	X         [NIn]float64     `json:"x"`
	Target    [NOut]float64    `json:"target"`
	Fwd       MLPForward       `json:"fwd"`
	Bwd       MLPBackward      `json:"bwd"`
	ErroTotal float64          `json:"erroTotal"`
}

type MLPResult struct {
	Convergiu     bool      `json:"convergiu"`
	Ciclos        int       `json:"ciclos"`
	ErroFinal     float64   `json:"erroFinal"`
	ErroHistorico []float64 `json:"erroHistorico"`
	Steps         []MLPStep `json:"steps"`
	Rede          MLP       `json:"rede"`
}

func inicializar() MLP {
	var m MLP
	m.V[0][0] = 0.12; m.V[0][1] = -0.03
	m.V[1][0] = -0.04; m.V[1][1] = 0.15
	m.V[2][0] = 0.31; m.V[2][1] = -0.41
	m.V0[0] = -0.09; m.V0[1] = 0.18
	m.W[0][0] = -0.05; m.W[0][1] = -0.34; m.W[0][2] = 0.21
	m.W[1][0] = 0.19; m.W[1][1] = -0.09; m.W[1][2] = 0.26
	m.W0[0] = 0.18; m.W0[1] = -0.27; m.W0[2] = -0.12
	return m
}

func tanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func forward(m MLP, x [NIn]float64) MLPForward {
	var f MLPForward
	for j := 0; j < NHid; j++ {
		f.Zin[j] = m.V0[j]
		for i := 0; i < NIn; i++ {
			f.Zin[j] += x[i] * m.V[i][j]
		}
		f.Z[j] = math.Tanh(f.Zin[j])
	}
	for k := 0; k < NOut; k++ {
		f.Yin[k] = m.W0[k]
		for j := 0; j < NHid; j++ {
			f.Yin[k] += f.Z[j] * m.W[j][k]
		}
		f.Y[k] = math.Tanh(f.Yin[k])
	}
	return f
}

func backward(m MLP, f MLPForward, target [NOut]float64, x [NIn]float64) MLPBackward {
	var b MLPBackward
	for k := 0; k < NOut; k++ {
		b.DeltaK[k] = (target[k] - f.Y[k]) * tanhDeriv(f.Y[k])
	}
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOut; k++ {
			b.DeltaInJ[j] += b.DeltaK[k] * m.W[j][k]
		}
		b.DeltaJ[j] = b.DeltaInJ[j] * tanhDeriv(f.Z[j])
	}
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOut; k++ {
			b.DeltaW[j][k] = alfa * b.DeltaK[k] * f.Z[j]
		}
	}
	for k := 0; k < NOut; k++ {
		b.DeltaW0[k] = alfa * b.DeltaK[k]
	}
	for i := 0; i < NIn; i++ {
		for j := 0; j < NHid; j++ {
			b.DeltaV[i][j] = alfa * b.DeltaJ[j] * x[i]
		}
	}
	for j := 0; j < NHid; j++ {
		b.DeltaV0[j] = alfa * b.DeltaJ[j]
	}
	return b
}

func atualizarPesos(m MLP, b MLPBackward) MLP {
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOut; k++ {
			m.W[j][k] += b.DeltaW[j][k]
		}
	}
	for k := 0; k < NOut; k++ {
		m.W0[k] += b.DeltaW0[k]
	}
	for i := 0; i < NIn; i++ {
		for j := 0; j < NHid; j++ {
			m.V[i][j] += b.DeltaV[i][j]
		}
	}
	for j := 0; j < NHid; j++ {
		m.V0[j] += b.DeltaV0[j]
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

// Treinar executa o treinamento do MLP desafio e retorna o resultado.
func Treinar() MLPResult {
	padroes := [3][NIn]float64{
		{1, 0.5, -1},
		{1, 0.5, 1},
		{1, -0.5, -1},
	}
	targets := [3][NOut]float64{
		{1, -1, -1},
		{-1, 1, -1},
		{-1, -1, 1},
	}

	m := inicializar()
	var res MLPResult
	const maxSteps = 300

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		erroTotal := 0.0
		for p := 0; p < 3; p++ {
			fwd := forward(m, padroes[p])
			bwd := backward(m, fwd, targets[p], padroes[p])
			m = atualizarPesos(m, bwd)
			erroTotal += calcErro(fwd.Y, targets[p])

			if len(res.Steps) < maxSteps {
				res.Steps = append(res.Steps, MLPStep{
					Ciclo:     ciclo,
					Padrao:    p + 1,
					X:         padroes[p],
					Target:    targets[p],
					Fwd:       fwd,
					Bwd:       bwd,
					ErroTotal: erroTotal,
				})
			}
		}
		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.Rede = m
			return res
		}
	}

	res.Convergiu = false
	res.Ciclos = maxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Rede = m
	return res
}

// Classificar classifica uma entrada e retorna o índice da classe vencedora.
func Classificar(m MLP, x [NIn]float64) int {
	fwd := forward(m, x)
	best := 0
	for k := 1; k < NOut; k++ {
		if fwd.Y[k] > fwd.Y[best] {
			best = k
		}
	}
	return best
}
