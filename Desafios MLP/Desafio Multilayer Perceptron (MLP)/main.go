package main

import (
	"math"
)

// =============================================================================
// Constantes da arquitetura
// N_IN  = número de entradas (x1, x2, x3)
// N_HID = número de neurônios na camada oculta
// N_OUT = número de neurônios de saída
// =============================================================================

const (
	N_IN       = 3     // neurônios de entrada
	N_HID      = 2     // neurônios ocultos
	N_OUT      = 3     // neurônios de saída
	ALFA       = 0.01  // taxa de aprendizado
	MAX_CICLOS = 50000 // limite de iterações
	ERRO_ALVO  = 0.001 // critério de parada: erro total ≤ ERRO_ALVO
)

// =============================================================================
// MLP — pesos da rede
// v[i][j]  = peso da entrada i para o neurônio oculto j
// v0[j]    = bias do neurônio oculto j
// w[j][k]  = peso do neurônio oculto j para a saída k
// w0[k]    = bias da saída k
// =============================================================================

type MLP struct {
	v  [N_IN][N_HID]float64  // pesos entrada→oculta
	v0 [N_HID]float64        // bias oculta
	w  [N_HID][N_OUT]float64 // pesos oculta→saída
	w0 [N_OUT]float64        // bias saída
}

// ForwardResult armazena os valores calculados no passo forward:
// zin = pré-ativação dos neurônios ocultos
// z   = saída pós-tanh dos neurônios ocultos
// yin = pré-ativação dos neurônios de saída
// y   = saída final pós-tanh
type ForwardResult struct {
	zin [N_HID]float64
	z   [N_HID]float64
	yin [N_OUT]float64
	y   [N_OUT]float64
}

// BackwardResult armazena todos os deltas e incrementos calculados no backprop
type BackwardResult struct {
	deltaK  [N_OUT]float64        // δ_k: sinal de erro na saída
	deltaInJ [N_HID]float64       // δin_j: sinal propagado para camada oculta
	deltaJ  [N_HID]float64        // δ_j: sinal de erro na camada oculta
	deltaW  [N_HID][N_OUT]float64 // Δw_jk: incremento dos pesos oculta→saída
	deltaW0 [N_OUT]float64        // Δw0_k: incremento dos biases de saída
	deltaV  [N_IN][N_HID]float64  // Δv_ij: incremento dos pesos entrada→oculta
	deltaV0 [N_HID]float64        // Δv0_j: incremento dos biases ocultos
}

// TrainingStep registra o estado completo de um passo de treinamento
// (um padrão em um ciclo), usado para animação no TUI
type TrainingStep struct {
	ciclo     int
	padrao    int
	x         [N_IN]float64
	target    [N_OUT]float64
	fwd       ForwardResult
	bwd       BackwardResult
	erroTotal float64 // erro acumulado do ciclo até este passo
}

// ResultadoTreino contém tudo após o treino completo
type ResultadoTreino struct {
	convergiu     bool
	ciclos        int
	erroFinal     float64
	erroHistorico []float64 // erro por ciclo (para gráfico)
	steps         []TrainingStep
	rede          MLP
}

// =============================================================================
// inicializarPesosSlide — pesos exatos extraídos dos slides da Aula 05
// =============================================================================

func inicializarPesosSlide() MLP {
	var mlp MLP

	// Pesos v (entrada→oculta): v[entrada][oculto]
	mlp.v[0][0] = 0.12
	mlp.v[0][1] = -0.03
	mlp.v[1][0] = -0.04
	mlp.v[1][1] = 0.15
	mlp.v[2][0] = 0.31
	mlp.v[2][1] = -0.41

	// Bias da camada oculta
	mlp.v0[0] = -0.09
	mlp.v0[1] = 0.18

	// Pesos w (oculta→saída): w[oculto][saída]
	// Slide apresenta como matriz 3×2 (saída×oculta), então transpomos:
	// slide row0 (saída 0): -0.05  0.19  → w[0][0]=-0.05, w[1][0]=0.19
	// slide row1 (saída 1): -0.34 -0.09  → w[0][1]=-0.34, w[1][1]=-0.09
	// slide row2 (saída 2):  0.21  0.26  → w[0][2]=0.21,  w[1][2]=0.26
	mlp.w[0][0] = -0.05
	mlp.w[0][1] = -0.34
	mlp.w[0][2] = 0.21
	mlp.w[1][0] = 0.19
	mlp.w[1][1] = -0.09
	mlp.w[1][2] = 0.26

	// Bias da camada de saída
	mlp.w0[0] = 0.18
	mlp.w0[1] = -0.27
	mlp.w0[2] = -0.12

	return mlp
}

// =============================================================================
// tanhDeriv — derivada de tanh usando a saída: f'(y) = (1+y)(1-y)
// =============================================================================

func tanhDeriv(y float64) float64 {
	return (1 + y) * (1 - y)
}

// =============================================================================
// forward — calcula a saída da rede para uma entrada x
//
// Camada oculta:
//   zin_j = v0_j + Σ_i x_i * v[i][j]   (pré-ativação)
//   z_j   = tanh(zin_j)                  (pós-ativação)
//
// Camada de saída:
//   yin_k = w0_k + Σ_j z_j * w[j][k]   (pré-ativação)
//   y_k   = tanh(yin_k)                  (saída final)
// =============================================================================

func forward(mlp MLP, x [N_IN]float64) ForwardResult {
	var fwd ForwardResult

	// Camada oculta
	for j := 0; j < N_HID; j++ {
		fwd.zin[j] = mlp.v0[j]
		for i := 0; i < N_IN; i++ {
			fwd.zin[j] += x[i] * mlp.v[i][j]
		}
		fwd.z[j] = math.Tanh(fwd.zin[j])
	}

	// Camada de saída
	for k := 0; k < N_OUT; k++ {
		fwd.yin[k] = mlp.w0[k]
		for j := 0; j < N_HID; j++ {
			fwd.yin[k] += fwd.z[j] * mlp.w[j][k]
		}
		fwd.y[k] = math.Tanh(fwd.yin[k])
	}

	return fwd
}

// =============================================================================
// backward — calcula os deltas e incrementos de peso para um padrão
//
// δ saída:    δ_k    = (t_k - y_k) * (1+y_k)(1-y_k)
// δin oculta: δin_j  = Σ_k δ_k * w[j][k]
// δ oculta:   δ_j    = δin_j * (1+z_j)(1-z_j)
// Update saída: Δw_jk = α * δ_k * z_j,   Δw0_k = α * δ_k
// Update oculta: Δv_ij = α * δ_j * x_i,  Δv0_j = α * δ_j
// =============================================================================

func backward(mlp MLP, fwd ForwardResult, target [N_OUT]float64, x [N_IN]float64) BackwardResult {
	var bwd BackwardResult

	// δ_k — erro na camada de saída, modulado pela derivada de tanh
	for k := 0; k < N_OUT; k++ {
		bwd.deltaK[k] = (target[k] - fwd.y[k]) * tanhDeriv(fwd.y[k])
	}

	// δin_j — propaga o sinal de erro da saída para a oculta
	for j := 0; j < N_HID; j++ {
		bwd.deltaInJ[j] = 0
		for k := 0; k < N_OUT; k++ {
			bwd.deltaInJ[j] += bwd.deltaK[k] * mlp.w[j][k]
		}
	}

	// δ_j — erro na camada oculta, modulado pela derivada de tanh
	for j := 0; j < N_HID; j++ {
		bwd.deltaJ[j] = bwd.deltaInJ[j] * tanhDeriv(fwd.z[j])
	}

	// Δw_jk e Δw0_k — incrementos dos pesos de saída
	for j := 0; j < N_HID; j++ {
		for k := 0; k < N_OUT; k++ {
			bwd.deltaW[j][k] = ALFA * bwd.deltaK[k] * fwd.z[j]
		}
	}
	for k := 0; k < N_OUT; k++ {
		bwd.deltaW0[k] = ALFA * bwd.deltaK[k]
	}

	// Δv_ij e Δv0_j — incrementos dos pesos da camada oculta
	for i := 0; i < N_IN; i++ {
		for j := 0; j < N_HID; j++ {
			bwd.deltaV[i][j] = ALFA * bwd.deltaJ[j] * x[i]
		}
	}
	for j := 0; j < N_HID; j++ {
		bwd.deltaV0[j] = ALFA * bwd.deltaJ[j]
	}

	return bwd
}

// =============================================================================
// atualizarPesos — aplica os incrementos calculados no backward
// =============================================================================

func atualizarPesos(mlp MLP, bwd BackwardResult) MLP {
	// Update pesos oculta→saída e seus biases
	for j := 0; j < N_HID; j++ {
		for k := 0; k < N_OUT; k++ {
			mlp.w[j][k] += bwd.deltaW[j][k]
		}
	}
	for k := 0; k < N_OUT; k++ {
		mlp.w0[k] += bwd.deltaW0[k]
	}

	// Update pesos entrada→oculta e seus biases
	for i := 0; i < N_IN; i++ {
		for j := 0; j < N_HID; j++ {
			mlp.v[i][j] += bwd.deltaV[i][j]
		}
	}
	for j := 0; j < N_HID; j++ {
		mlp.v0[j] += bwd.deltaV0[j]
	}

	return mlp
}

// =============================================================================
// calcularErro — erro quadrático: E = ½ Σ(t_k - y_k)²
// =============================================================================

func calcularErro(y, t [N_OUT]float64) float64 {
	var e float64
	for k := 0; k < N_OUT; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

// =============================================================================
// treinarMLP — loop principal de treinamento
//
// Itera até MAX_CICLOS ou erro total ≤ ERRO_ALVO.
// Salva todos os steps para animação.
// Um "ciclo" = apresentação dos 3 padrões na sequência.
// =============================================================================

func treinarMLP() ResultadoTreino {
	// Padrões de treinamento (exatos do slide)
	padroes := [3][N_IN]float64{
		{1, 0.5, -1},  // padrão 1
		{1, 0.5, 1},   // padrão 2
		{1, -0.5, -1}, // padrão 3
	}
	targets := [3][N_OUT]float64{
		{1, -1, -1},  // target 1
		{-1, 1, -1},  // target 2
		{-1, -1, 1},  // target 3
	}

	mlp := inicializarPesosSlide()
	var resultado ResultadoTreino
	var erroHistorico []float64

	// Limita os steps salvos para não explodir memória (apenas os primeiros 300)
	const maxStepsSalvos = 300

	for ciclo := 1; ciclo <= MAX_CICLOS; ciclo++ {
		erroTotal := 0.0

		for p := 0; p < 3; p++ {
			fwd := forward(mlp, padroes[p])
			bwd := backward(mlp, fwd, targets[p], padroes[p])
			mlp = atualizarPesos(mlp, bwd)
			erroTotal += calcularErro(fwd.y, targets[p])

			// Salva step para animação (apenas os primeiros N)
			if len(resultado.steps) < maxStepsSalvos {
				resultado.steps = append(resultado.steps, TrainingStep{
					ciclo:     ciclo,
					padrao:    p + 1,
					x:         padroes[p],
					target:    targets[p],
					fwd:       fwd,
					bwd:       bwd,
					erroTotal: erroTotal,
				})
			}
		}

		erroHistorico = append(erroHistorico, erroTotal)

		// Critério de parada
		if erroTotal <= ERRO_ALVO {
			resultado.convergiu = true
			resultado.ciclos = ciclo
			resultado.erroFinal = erroTotal
			resultado.erroHistorico = erroHistorico
			resultado.rede = mlp
			return resultado
		}
	}

	// Não convergiu — retorna estado final
	resultado.convergiu = false
	resultado.ciclos = MAX_CICLOS
	resultado.erroFinal = erroHistorico[len(erroHistorico)-1]
	resultado.erroHistorico = erroHistorico
	resultado.rede = mlp
	return resultado
}

// =============================================================================
// classificar — retorna o índice (0-based) do neurônio de saída com maior valor
// =============================================================================

func classificar(mlp MLP, x [N_IN]float64) int {
	fwd := forward(mlp, x)
	best := 0
	for k := 1; k < N_OUT; k++ {
		if fwd.y[k] > fwd.y[best] {
			best = k
		}
	}
	return best
}

// =============================================================================
// main — ponto de entrada: inicializa e executa o TUI
// =============================================================================

func main() {
	runTUI()
}
