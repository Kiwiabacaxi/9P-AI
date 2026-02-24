package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// TRABALHO 02 - Parte 01: Perceptron para Reconhecimento de Letras A e B
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// Adaptação do Trabalho 01 (Regra de Hebb) para usar o Perceptron.
//
// DIFERENÇA PRINCIPAL: Hebb vs Perceptron
//
//   HEBB (Trabalho 01):
//     Atualiza os pesos para TODA amostra, sempre:
//       w_i = w_i + (x_i * target)
//
//   PERCEPTRON (Trabalho 02):
//     Atualiza os pesos APENAS quando a predição está errada:
//       y_in  = soma(x_i * w_i) + bias
//       y     = ativacao(y_in)
//       SE y != target:
//         w_i  = w_i  + alfa * (target - y) * x_i
//         bias = bias + alfa * (target - y)
//
//     O treinamento repete por CICLOS até que nenhum erro ocorra.
//
// ENTRADAS: Letras A e B representadas em matrizes 7x7 (49 pixels).
//   Pixel "ligado"  =  1  (parte da letra)
//   Pixel "apagado" = -1  (fundo)
//
// SAÍDAS (bipolar):
//   A → target = -1
//   B → target =  1
//
// PARÂMETROS (seguindo o código C do Prof. Jefferson):
//   Pesos iniciais: aleatórios no intervalo [-0.5, +0.5]
//   Bias inicial:    aleatório em [-0.5, +0.5]
// =============================================================================

const ALFA = 0.01
const LINHAS = 7
const COLUNAS = 7
const N_ENTRADAS = LINHAS * COLUNAS // 49 pixels

// ativacao aplica o degrau bipolar.
//
//	y_in >= 0 → saída =  1  (reconheceu como B)
//	y_in <  0 → saída = -1  (reconheceu como A)
func ativacao(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

// letraA define a letra A em uma grade 7x7.
// 1  = pixel ativo (parte da letra)
// -1 = pixel inativo (fundo)
//
//	. # # # # # .
//	# . . . . . #
//	# . . . . . #
//	# # # # # # #
//	# . . . . . #
//	# . . . . . #
//	# . . . . . #
func letraA() [N_ENTRADAS]float64 {
	grade := [LINHAS][COLUNAS]int{
		{-1, 1, 1, 1, 1, 1, -1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
	}
	var entrada [N_ENTRADAS]float64
	idx := 0
	for _, linha := range grade {
		for _, pixel := range linha {
			entrada[idx] = float64(pixel)
			idx++
		}
	}
	return entrada
}

// letraB define a letra B em uma grade 7x7.
//
//	# # # # # . .
//	# . . . . # .
//	# . . . . # .
//	# # # # # . .
//	# . . . . # .
//	# . . . . # .
//	# # # # # . .
func letraB() [N_ENTRADAS]float64 {
	grade := [LINHAS][COLUNAS]int{
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
	}
	var entrada [N_ENTRADAS]float64
	idx := 0
	for _, linha := range grade {
		for _, pixel := range linha {
			entrada[idx] = float64(pixel)
			idx++
		}
	}
	return entrada
}

// formataLetra retorna uma string renderizada colorida da grade de pixels
func formataLetra(entrada [N_ENTRADAS]float64) string {
	var sb strings.Builder
	for i := 0; i < LINHAS; i++ {
		for j := 0; j < COLUNAS; j++ {
			if entrada[i*COLUNAS+j] == 1 {
				sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("# "))
			} else {
				sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render(". "))
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// preparaTreinamento executa o treinamento completo da Rede Neural e salva os logs
// passo-a-passo (épocas) dentro do Bubble Tea Model (`m`).
func (m *model) preparaTreinamento() {
	amostras := [][N_ENTRADAS]float64{letraA(), letraB()}
	targets := []int{-1, 1}
	nomes := []string{"A", "B"}

	// Inicialização dos pesos aleatoriamente entre [-0.5, 0.5]
	for i := range m.pesos {
		m.pesos[i] = rand.Float64() - 0.5
	}
	m.bias = rand.Float64() - 0.5
	m.trainingSteps = []trainingStep{}

	ciclo := 0
	for {
		ciclo++
		condErro := false

		// Para cada amostra apresentada na época
		for s, entrada := range amostras {

			// 1. Calcula o Potencial de Ativação (soma: y_in = bias + somatório(w_i * x_i))
			yIn := m.bias
			for i := 0; i < N_ENTRADAS; i++ {
				yIn += entrada[i] * m.pesos[i]
			}

			// 2. Chama Função de Ativação Bipolar (Degrau)
			y := ativacao(yIn)
			teveErro := false
			var delta float64

			// 3. Regra de Aprendizagem (Perceptron só atualiza caso y seja GABARITO (target) falho)
			if y != targets[s] {
				condErro = true
				teveErro = true

				// Calcula o delta de Erro: Alfa * (T - Y)
				delta = ALFA * float64(targets[s]-y)

				// Atualiza todos os pesos com o delta iterativo
				for i := 0; i < N_ENTRADAS; i++ {
					m.pesos[i] += delta * entrada[i]
				}
				// Atualiza o Base de Viés com o delta
				m.bias += delta
			}

			// Salva o log desse passo para a interface gráfica Bubble Tea exibir bonitinho
			m.trainingSteps = append(m.trainingSteps, trainingStep{
				ciclo:    ciclo,
				amostra:  nomes[s],
				target:   targets[s],
				yIn:      yIn,
				y:        y,
				delta:    delta,
				novoBias: m.bias,
				teveErro: teveErro,
			})
		}

		// Se a rede acertou todas as amostras dessa época, encerra
		if !condErro {
			break
		}
		if ciclo >= 10000 {
			break // Redundância para não ficar em um loop infinito
		}
	}
	m.ciclosTreino = ciclo
}

// operar aplica a rede já treinada com as matrizes fixadas e tira a acurácia.
func (m *model) operar() string {
	amostras := [][N_ENTRADAS]float64{letraA(), letraB()}
	targets := []int{-1, 1}
	nomes := []string{"A", "B"}

	var sb strings.Builder
	acertos := 0

	for s, entrada := range amostras {
		yIn := m.bias
		for i := 0; i < N_ENTRADAS; i++ {
			yIn += entrada[i] * m.pesos[i]
		}

		predicao := ativacao(yIn)
		status := successStyle.Render("✓ OK")
		if predicao == targets[s] {
			acertos++
		} else {
			status = errorStyle.Render("✗ ERRO")
		}

		nomePredicao := "B"
		if predicao == -1 {
			nomePredicao = "A"
		}

		sb.WriteString(fmt.Sprintf("Letra: %s | Alvo: %2d | Predição: %2d (%s) %s\n",
			nomes[s], targets[s], predicao, nomePredicao, status))
	}

	sb.WriteString(fmt.Sprintf("\nAcurácia: %d/%d (%.0f%%)",
		acertos, len(amostras), float64(acertos)/float64(len(amostras))*100))

	return sb.String()
}

func main() {
	// Seed para inicialização dos pesos aleatórios para que variem a cada execução
	rand.Seed(time.Now().UnixNano())

	// Invoca a Interface Gráfica interativa de UI TUI (do arquivo tui.go)
	iniciarTUI()
}
