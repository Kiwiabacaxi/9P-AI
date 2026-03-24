package main

import "math"

// =============================================================================
// MLP — Desafio (3 entradas, 2 ocultos, 3 saídas)
// Lógica extraída de Desafio Multilayer Perceptron (MLP)/main.go
// =============================================================================

const (
	mlpNIn      = 3
	mlpNHid     = 2
	mlpNOut     = 3
	mlpAlfa     = 0.01
	mlpMaxCiclo = 50000
	mlpErroAlvo = 0.001
)

type MLP struct {
	V  [mlpNIn][mlpNHid]float64
	V0 [mlpNHid]float64
	W  [mlpNHid][mlpNOut]float64
	W0 [mlpNOut]float64
}

type MLPForward struct {
	Zin [mlpNHid]float64
	Z   [mlpNHid]float64
	Yin [mlpNOut]float64
	Y   [mlpNOut]float64
}

type MLPBackward struct {
	DeltaK   [mlpNOut]float64
	DeltaInJ [mlpNHid]float64
	DeltaJ   [mlpNHid]float64
	DeltaW   [mlpNHid][mlpNOut]float64
	DeltaW0  [mlpNOut]float64
	DeltaV   [mlpNIn][mlpNHid]float64
	DeltaV0  [mlpNHid]float64
}

type MLPStep struct {
	Ciclo     int            `json:"ciclo"`
	Padrao    int            `json:"padrao"`
	X         [mlpNIn]float64  `json:"x"`
	Target    [mlpNOut]float64 `json:"target"`
	Fwd       MLPForward     `json:"fwd"`
	Bwd       MLPBackward    `json:"bwd"`
	ErroTotal float64        `json:"erroTotal"`
}

type MLPResult struct {
	Convergiu     bool      `json:"convergiu"`
	Ciclos        int       `json:"ciclos"`
	ErroFinal     float64   `json:"erroFinal"`
	ErroHistorico []float64 `json:"erroHistorico"`
	Steps         []MLPStep `json:"steps"`
	Rede          MLP       `json:"rede"`
}

func mlpInicializar() MLP {
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

func mlpForward(m MLP, x [mlpNIn]float64) MLPForward {
	var f MLPForward
	for j := 0; j < mlpNHid; j++ {
		f.Zin[j] = m.V0[j]
		for i := 0; i < mlpNIn; i++ {
			f.Zin[j] += x[i] * m.V[i][j]
		}
		f.Z[j] = math.Tanh(f.Zin[j])
	}
	for k := 0; k < mlpNOut; k++ {
		f.Yin[k] = m.W0[k]
		for j := 0; j < mlpNHid; j++ {
			f.Yin[k] += f.Z[j] * m.W[j][k]
		}
		f.Y[k] = math.Tanh(f.Yin[k])
	}
	return f
}

func mlpBackward(m MLP, f MLPForward, target [mlpNOut]float64, x [mlpNIn]float64) MLPBackward {
	var b MLPBackward
	for k := 0; k < mlpNOut; k++ {
		b.DeltaK[k] = (target[k] - f.Y[k]) * tanhDeriv(f.Y[k])
	}
	for j := 0; j < mlpNHid; j++ {
		for k := 0; k < mlpNOut; k++ {
			b.DeltaInJ[j] += b.DeltaK[k] * m.W[j][k]
		}
		b.DeltaJ[j] = b.DeltaInJ[j] * tanhDeriv(f.Z[j])
	}
	for j := 0; j < mlpNHid; j++ {
		for k := 0; k < mlpNOut; k++ {
			b.DeltaW[j][k] = mlpAlfa * b.DeltaK[k] * f.Z[j]
		}
	}
	for k := 0; k < mlpNOut; k++ {
		b.DeltaW0[k] = mlpAlfa * b.DeltaK[k]
	}
	for i := 0; i < mlpNIn; i++ {
		for j := 0; j < mlpNHid; j++ {
			b.DeltaV[i][j] = mlpAlfa * b.DeltaJ[j] * x[i]
		}
	}
	for j := 0; j < mlpNHid; j++ {
		b.DeltaV0[j] = mlpAlfa * b.DeltaJ[j]
	}
	return b
}

func mlpAtualizarPesos(m MLP, b MLPBackward) MLP {
	for j := 0; j < mlpNHid; j++ {
		for k := 0; k < mlpNOut; k++ {
			m.W[j][k] += b.DeltaW[j][k]
		}
	}
	for k := 0; k < mlpNOut; k++ {
		m.W0[k] += b.DeltaW0[k]
	}
	for i := 0; i < mlpNIn; i++ {
		for j := 0; j < mlpNHid; j++ {
			m.V[i][j] += b.DeltaV[i][j]
		}
	}
	for j := 0; j < mlpNHid; j++ {
		m.V0[j] += b.DeltaV0[j]
	}
	return m
}

func mlpCalcErro(y, t [mlpNOut]float64) float64 {
	var e float64
	for k := 0; k < mlpNOut; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

func mlpTreinar() MLPResult {
	padroes := [3][mlpNIn]float64{
		{1, 0.5, -1},
		{1, 0.5, 1},
		{1, -0.5, -1},
	}
	targets := [3][mlpNOut]float64{
		{1, -1, -1},
		{-1, 1, -1},
		{-1, -1, 1},
	}

	m := mlpInicializar()
	var res MLPResult
	const maxSteps = 300

	for ciclo := 1; ciclo <= mlpMaxCiclo; ciclo++ {
		erroTotal := 0.0
		for p := 0; p < 3; p++ {
			fwd := mlpForward(m, padroes[p])
			bwd := mlpBackward(m, fwd, targets[p], padroes[p])
			m = mlpAtualizarPesos(m, bwd)
			erroTotal += mlpCalcErro(fwd.Y, targets[p])

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

		if erroTotal <= mlpErroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			res.Rede = m
			return res
		}
	}

	res.Convergiu = false
	res.Ciclos = mlpMaxCiclo
	res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	res.Rede = m
	return res
}

func mlpClassificar(m MLP, x [mlpNIn]float64) int {
	fwd := mlpForward(m, x)
	best := 0
	for k := 1; k < mlpNOut; k++ {
		if fwd.Y[k] > fwd.Y[best] {
			best = k
		}
	}
	return best
}
