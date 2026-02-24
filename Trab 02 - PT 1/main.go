package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// TRABALHO 02 - PARTE 01: Perceptron para Reconhecimento de Letras (A e B)
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// Diferença entre Hebb e Perceptron:
//
//   HEBB: Os pesos são atualizados usando a SAÍDA DESEJADA (target).
//         w = w + (x * target)
//         Isso significa que o aprendizado acontece sempre, acertando ou não.
//
//   PERCEPTRON: Os pesos só são atualizados quando a SAÍDA CALCULADA
//               for DIFERENTE da saída desejada (condErro).
//               w[col] = w[col] + alfa * (target - y) * entrada[col]
//               b      = b      + alfa * (target - y)
//
// Isso torna o Perceptron mais robusto: ele só corrige quando erra!
//
// ARQUITETURA:
//   - 49 entradas (matriz 7x7 "achatada" em vetor)
//   - 49 pesos (um por pixel)
//   - 1 bias
//   - 2 padrões: Letra A (target = -1) e Letra B (target = 1)
//
// CÁLCULO DO POTENCIAL DE ATIVAÇÃO:
//   yLiq = (x[0]*w[0]) + (x[1]*w[1]) + ... + (x[48]*w[48]) + bias
//
// REGRA DE ATUALIZAÇÃO (Perceptron - só quando erra):
//   w[i] = w[i] + alfa * (target - y) * x[i]
//   bias = bias + alfa * (target - y)
//
// FUNÇÃO DE ATIVAÇÃO - Degrau Bipolar:
//   yLiq >= 0  →  saída =  1  (reconheceu como B)
//   yLiq <  0  →  saída = -1  (reconheceu como A)
//
// PARÂMETROS (seguindo o código C do Prof. Jefferson):
//   Pesos iniciais: aleatórios no intervalo [-0.5, +0.5]
//   Bias inicial:    aleatório em [-0.5, +0.5]
// =============================================================================

const (
	LINHAS     = 7
	COLUNAS    = 7
	N_ENTRADAS = LINHAS * COLUNAS // 49 pixels por letra
	ALFA       = 0.01             // taxa de aprendizagem
)

// ativacao aplica o degrau bipolar: retorna 1 ou -1.
// yLiq >= 0 → saída =  1  (reconheceu como B)
// yLiq <  0 → saída = -1  (reconheceu como A)
func ativacao(yLiq float64) int {
	if yLiq >= 0 {
		return 1
	}
	return -1
}

// letraA
// 1  = pixel ativo
// -1 = pixel inativo
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

// letraB
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

// formataLetra retorna uma string colorida da grade 7x7.
func formataLetra(entrada [N_ENTRADAS]float64) string {
	var sb strings.Builder
	for i := 0; i < LINHAS; i++ {
		for j := 0; j < COLUNAS; j++ {
			if entrada[i*COLUNAS+j] == 1 {
				sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("█ "))
			} else {
				sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render("· "))
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// preparaTreinamento -> loop completo de treinamento do Perceptron
// salva cada passo/época dentro do model do Bubble Tea (visual).
//
// DIFERENÇA CHAVE para o Hebb:
//   - O loop continua até que NENHUM erro ocorra em um ciclo completo
//     (condErro == false), garantindo convergência se os padrões forem
//     linearmente separáveis.
//   - Os pesos só mudam quando a predição está ERRADA.
func (m *model) preparaTreinamento() {
	amostras := [][N_ENTRADAS]float64{letraA(), letraB()}
	targets := []int{-1, 1} // A = -1, B = 1
	nomes := []string{"A", "B"}

	// Inicialização dos pesos aleatoriamente entre [-0.5, 0.5], slide
	for i := range m.pesos {
		m.pesos[i] = rand.Float64() - 0.5
	}
	m.bias = rand.Float64() - 0.5
	m.trainingSteps = []trainingStep{}

	ciclo := 0
	for {
		ciclo++
		condErro := false

		for s, entrada := range amostras {

			// 1. Calcula o Potencial de Ativação
			//    yLiq = bias + somatório(w[i] * x[i])
			yLiq := m.bias
			for i := 0; i < N_ENTRADAS; i++ {
				yLiq += entrada[i] * m.pesos[i]
			}

			// 2. Aplica a função degrau
			y := ativacao(yLiq)
			teveErro := false
			var delta float64

			// 3. Regra de Aprendizagem:
			//    Só atualiza os pesos se a saída (y) for diferente do target
			if y != targets[s] {
				condErro = true
				teveErro = true

				// Calcula o delta: alfa * (target - y)
				delta = ALFA * float64(targets[s]-y)

				// Atualiza todos os pesos e o bias: w[i] = w[i] + delta * x[i]
				for i := 0; i < N_ENTRADAS; i++ {
					m.pesos[i] += delta * entrada[i]
				}
				m.bias += delta
			}

			// Salva o log para o Bubble Tea (TUI)
			m.trainingSteps = append(m.trainingSteps, trainingStep{
				ciclo:    ciclo,
				amostra:  nomes[s],
				target:   targets[s],
				yLiq:     yLiq,
				y:        y,
				delta:    delta,
				novoBias: m.bias,
				teveErro: teveErro,
			})
		}

		// Acertou tudo -> convergiu
		if !condErro {
			break
		}
		if ciclo >= 10000 {
			break // Proteção contra loop infinito
		}
	}
	m.ciclosTreino = ciclo
}

// aplica a rede já treinada nas letras A e B
// Usa os pesos e bias do treino
func (m *model) operar() string {
	amostras := [][N_ENTRADAS]float64{letraA(), letraB()}
	targets := []int{-1, 1}
	nomes := []string{"A", "B"}

	var sb strings.Builder
	acertos := 0

	for s, entrada := range amostras {
		// Recalcula o potencial com os pesos finais
		yLiq := m.bias
		for i := 0; i < N_ENTRADAS; i++ {
			yLiq += entrada[i] * m.pesos[i]
		}

		predicao := ativacao(yLiq)
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

		sb.WriteString(fmt.Sprintf("Letra: %s │ Alvo: %2d │ Predição: %2d (%s) %s\n",
			nomes[s], targets[s], predicao, nomePredicao, status))
	}

	sb.WriteString(fmt.Sprintf("\nAcurácia: %d/%d (%.0f%%)",
		acertos, len(amostras), float64(acertos)/float64(len(amostras))*100))

	return sb.String()
}

// --- PROGRAMA PRINCIPAL ---

func main() {
	// Seed para inicialização dos pesos aleatórios
	rand.Seed(time.Now().UnixNano())

	// Invoca a Interface TUI interativa (definida em tui.go)
	iniciarTUI()
}
