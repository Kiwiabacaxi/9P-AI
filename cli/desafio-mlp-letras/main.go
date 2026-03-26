package main

import (
	"math"
	"math/rand"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// MLP Letras — Reconhecimento de Letras A–Z com Backpropagation
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// ARQUITETURA MLP:
//   - Entradas: grade 5×7 = 35 pixels em bipolar (-1/+1)
//   - Camada Oculta: 15 neurônios
//   - Saída: 26 neurônios (um por letra A–Z), one-hot (+1 correto, -1 demais)
//   - Ativação: tanh em todas as camadas
//   - α = 0.01, MAX_CICLOS = 50000, ERRO_ALVO = 0.5
//   - Pesos iniciais: rand [-0.5, +0.5]
// =============================================================================

const (
	N_LINHAS  = 7
	N_COLUNAS = 5
	N_IN      = 35    // 5×7 pixels
	N_HID     = 15    // neurônios ocultos
	N_OUT     = 26    // letras A–Z
	ALFA      = 0.01  // taxa de aprendizado
	MAX_CICLOS = 50000 // limite de iterações
	ERRO_ALVO  = 0.5   // critério de parada: erro total ≤ ERRO_ALVO
)

var nomesLetras = [N_OUT]string{
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
	"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
}

// =============================================================================
// MLP — pesos da rede
// =============================================================================

type MLP struct {
	v  [N_IN][N_HID]float64  // pesos entrada→oculta
	v0 [N_HID]float64        // bias oculta
	w  [N_HID][N_OUT]float64 // pesos oculta→saída
	w0 [N_OUT]float64        // bias saída
}

type ForwardResult struct {
	zin [N_HID]float64
	z   [N_HID]float64
	yin [N_OUT]float64
	y   [N_OUT]float64
}

type BackwardResult struct {
	deltaK   [N_OUT]float64
	deltaInJ [N_HID]float64
	deltaJ   [N_HID]float64
	deltaW   [N_HID][N_OUT]float64
	deltaW0  [N_OUT]float64
	deltaV   [N_IN][N_HID]float64
	deltaV0  [N_HID]float64
}

type TrainingStep struct {
	ciclo     int
	letraIdx  int
	fwd       ForwardResult
	erroTotal float64
}

type ResultadoTreino struct {
	convergiu     bool
	ciclos        int
	erroFinal     float64
	erroHistorico []float64
	steps         []TrainingStep // primeiros 200 para animação
	rede          MLP
}

// =============================================================================
// inicializarPesos — pesos aleatórios em [-0.5, +0.5]
// =============================================================================

func inicializarPesos() MLP {
	var mlp MLP
	for i := 0; i < N_IN; i++ {
		for j := 0; j < N_HID; j++ {
			mlp.v[i][j] = rand.Float64() - 0.5
		}
	}
	for j := 0; j < N_HID; j++ {
		mlp.v0[j] = rand.Float64() - 0.5
		for k := 0; k < N_OUT; k++ {
			mlp.w[j][k] = rand.Float64() - 0.5
		}
	}
	for k := 0; k < N_OUT; k++ {
		mlp.w0[k] = rand.Float64() - 0.5
	}
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
// =============================================================================

func forward(mlp MLP, x [N_IN]float64) ForwardResult {
	var fwd ForwardResult

	for j := 0; j < N_HID; j++ {
		fwd.zin[j] = mlp.v0[j]
		for i := 0; i < N_IN; i++ {
			fwd.zin[j] += x[i] * mlp.v[i][j]
		}
		fwd.z[j] = math.Tanh(fwd.zin[j])
	}

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
// =============================================================================

func backward(mlp MLP, fwd ForwardResult, target [N_OUT]float64, x [N_IN]float64) BackwardResult {
	var bwd BackwardResult

	for k := 0; k < N_OUT; k++ {
		bwd.deltaK[k] = (target[k] - fwd.y[k]) * tanhDeriv(fwd.y[k])
	}

	for j := 0; j < N_HID; j++ {
		bwd.deltaInJ[j] = 0
		for k := 0; k < N_OUT; k++ {
			bwd.deltaInJ[j] += bwd.deltaK[k] * mlp.w[j][k]
		}
	}

	for j := 0; j < N_HID; j++ {
		bwd.deltaJ[j] = bwd.deltaInJ[j] * tanhDeriv(fwd.z[j])
	}

	for j := 0; j < N_HID; j++ {
		for k := 0; k < N_OUT; k++ {
			bwd.deltaW[j][k] = ALFA * bwd.deltaK[k] * fwd.z[j]
		}
	}
	for k := 0; k < N_OUT; k++ {
		bwd.deltaW0[k] = ALFA * bwd.deltaK[k]
	}

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
	for j := 0; j < N_HID; j++ {
		for k := 0; k < N_OUT; k++ {
			mlp.w[j][k] += bwd.deltaW[j][k]
		}
	}
	for k := 0; k < N_OUT; k++ {
		mlp.w0[k] += bwd.deltaW0[k]
	}

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
// letrasDataset — retorna os 26 padrões 5×7 em bipolar como float64
// =============================================================================

func letrasDataset() [N_OUT][N_IN]float64 {
	grade := [N_OUT][N_LINHAS][N_COLUNAS]int{
		// A
		{
			{-1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// B
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
		},
		// C
		{
			{-1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{-1, 1, 1, 1, 1},
		},
		// D
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
		},
		// E
		{
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// F
		{
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
		},
		// G
		{
			{-1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, 1},
		},
		// H
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// I
		{
			{1, 1, 1, 1, 1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// J
		{
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, -1},
		},
		// K
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, 1, -1},
			{1, -1, 1, -1, -1},
			{1, 1, -1, -1, -1},
			{1, -1, 1, -1, -1},
			{1, -1, -1, 1, -1},
			{1, -1, -1, -1, 1},
		},
		// L
		{
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// M
		{
			{1, -1, -1, -1, 1},
			{1, 1, -1, 1, 1},
			{1, -1, 1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// N: barras laterais, diagonal no meio
		{
			{1, -1, -1, -1, 1},
			{1, 1, -1, -1, 1},
			{1, -1, 1, -1, 1},
			{1, -1, -1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// O: retângulo fechado com centro oco
		{
			{-1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, -1},
		},
		// P: topo fechado, baixo aberto
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
		},
		// Q: como O mas com rabinho no canto inferior direito
		{
			{-1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, 1, -1, 1},
			{1, -1, -1, 1, 1},
			{-1, 1, 1, 1, -1},
		},
		// R: como P mas com perna diagonal no canto inferior direito
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
			{1, -1, 1, -1, -1},
			{1, -1, -1, 1, -1},
			{1, -1, -1, -1, 1},
		},
		// S: topo-direita aberto, baixo-esquerda aberto, barra central
		{
			{-1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{-1, 1, 1, 1, -1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
		},
		// T: barra completa no topo, vertical centralizada
		{
			{1, 1, 1, 1, 1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
		},
		// U: barras laterais, fechada no baixo
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, -1},
		},
		// V: diagonais encontrando-se no centro baixo
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, -1, 1, -1},
			{-1, 1, -1, 1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
		},
		// W: como V mas com pico central subindo
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, 1, -1, 1},
			{1, -1, 1, -1, 1},
			{1, 1, -1, 1, 1},
			{1, 1, -1, 1, 1},
			{-1, 1, -1, 1, -1},
		},
		// X: diagonais cruzando
		{
			{1, -1, -1, -1, 1},
			{-1, 1, -1, 1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, 1, -1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// Y: diagonais encontrando no meio, depois vertical
		{
			{1, -1, -1, -1, 1},
			{-1, 1, -1, 1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
		},
		// Z: barra no topo, diagonal, barra no baixo
		{
			{1, 1, 1, 1, 1},
			{-1, -1, -1, 1, -1},
			{-1, -1, 1, -1, -1},
			{-1, 1, -1, -1, -1},
			{-1, 1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, 1},
		},
	}

	var dataset [N_OUT][N_IN]float64
	for l := 0; l < N_OUT; l++ {
		idx := 0
		for i := 0; i < N_LINHAS; i++ {
			for j := 0; j < N_COLUNAS; j++ {
				dataset[l][idx] = float64(grade[l][i][j])
				idx++
			}
		}
	}
	return dataset
}

// =============================================================================
// targetVetor — one-hot: +1.0 na posição idx, -1.0 nas demais
// =============================================================================

func targetVetor(idx int) [N_OUT]float64 {
	var t [N_OUT]float64
	for i := range t {
		t[i] = -1.0
	}
	t[idx] = 1.0
	return t
}

// =============================================================================
// treinarMLP — loop principal de treinamento
// =============================================================================

func treinarMLP() ResultadoTreino {
	mlp := inicializarPesos()
	dataset := letrasDataset()

	var resultado ResultadoTreino
	var erroHistorico []float64
	const maxStepsSalvos = 200

	for ciclo := 1; ciclo <= MAX_CICLOS; ciclo++ {
		erroTotal := 0.0

		for letraIdx := 0; letraIdx < N_OUT; letraIdx++ {
			x := dataset[letraIdx]
			target := targetVetor(letraIdx)

			fwd := forward(mlp, x)
			bwd := backward(mlp, fwd, target, x)
			mlp = atualizarPesos(mlp, bwd)
			erroTotal += calcularErro(fwd.y, target)

			if len(resultado.steps) < maxStepsSalvos {
				resultado.steps = append(resultado.steps, TrainingStep{
					ciclo:     ciclo,
					letraIdx:  letraIdx,
					fwd:       fwd,
					erroTotal: erroTotal,
				})
			}
		}

		erroHistorico = append(erroHistorico, erroTotal)

		if erroTotal <= ERRO_ALVO {
			resultado.convergiu = true
			resultado.ciclos = ciclo
			resultado.erroFinal = erroTotal
			resultado.erroHistorico = erroHistorico
			resultado.rede = mlp
			return resultado
		}
	}

	resultado.convergiu = false
	resultado.ciclos = MAX_CICLOS
	resultado.erroFinal = erroHistorico[len(erroHistorico)-1]
	resultado.erroHistorico = erroHistorico
	resultado.rede = mlp
	return resultado
}

// =============================================================================
// classificar — retorna o índice do neurônio de saída com maior valor
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
// formataLetraGrid — representação visual 5×7 de uma entrada float64
// =============================================================================

func formataLetraGrid(x [N_IN]float64) string {
	pixelAtivo := lipgloss.NewStyle().Foreground(lipgloss.Color("#FF6EC7")).Render("█")
	pixelInativo := lipgloss.NewStyle().Foreground(lipgloss.Color("#555555")).Render("·")

	var sb strings.Builder
	for i := 0; i < N_LINHAS; i++ {
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			if x[i*N_COLUNAS+j] > 0 {
				sb.WriteString(pixelAtivo)
			} else {
				sb.WriteString(pixelInativo)
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// =============================================================================
// main — ponto de entrada
// =============================================================================

func main() {
	runTUI()
}
