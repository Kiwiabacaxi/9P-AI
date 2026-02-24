package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// Estilos (Lipgloss)
// =============================================================================

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 2).
			MarginBottom(1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#A3A3A3")).
			MarginBottom(1)

	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(lipgloss.Color("#CCCCCC"))

	selectedItemStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(lipgloss.Color("#00E676")).
				Bold(true)

	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#7D56F4")).
			Padding(1, 2).
			MarginRight(2)

	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00E676")).Bold(true)
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5252")).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#64B5F6"))
)

// =============================================================================
// Bubble Tea Model e Lógica Visual
// =============================================================================

type sessionState int

const (
	stateMenu sessionState = iota
	stateTraining
	stateTrainingDone
	stateOperating
)

type trainingStep struct {
	ciclo    int
	amostra  string
	target   int
	yIn      float64
	y        int
	delta    float64
	novoBias float64
	teveErro bool
}

type model struct {
	state   sessionState
	cursor  int
	choices []string

	spinner spinner.Model

	pesos        [N_ENTRADAS]float64
	bias         float64
	redeTreinada bool
	ciclosTreino int

	// Para animação passo a passo
	trainingSteps   []trainingStep
	currentStepIdx  int
	isAutoOperating bool // Flag para saber se viemos do "Treinar e Mostrar rede"

	resultadoOperacao string
}

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	return model{
		state:   stateMenu,
		choices: []string{"Treinar a rede passo a passo", "Operar a rede", "Treinar e Mostrar rede", "Sair"},
		spinner: s,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

type trainingTickMsg time.Time

func tickTraining() tea.Cmd {
	return tea.Tick(time.Millisecond*600, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

// Update recebe todas as mensagens (botões apertados, ticks de tempo, etc).
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q", "esc":
			if m.state != stateMenu {
				m.state = stateMenu
				return m, nil
			}
			return m, tea.Quit

		case "up", "k":
			if m.state == stateMenu {
				if m.cursor > 0 {
					m.cursor--
				}
			}
		case "down", "j":
			if m.state == stateMenu {
				if m.cursor < len(m.choices)-1 {
					m.cursor++
				}
			}
		case "enter", " ":
			if m.state == stateMenu {
				switch m.cursor {
				case 0: // Treinar
					m.isAutoOperating = false
					m.state = stateTraining
					m.currentStepIdx = 0

					// Prepara/Processa os cálculos matemáticos da rede
					m.preparaTreinamento()

					return m, tea.Batch(m.spinner.Tick, tickTraining())
				case 1: // Operar
					if !m.redeTreinada {
						m.resultadoOperacao = errorStyle.Render("ERRO:") + " A rede ainda não foi treinada!\nTreine-a primeiro (Opção 1)."
					} else {
						// Operando com os pesos gravados no 'model' internamente
						m.resultadoOperacao = m.operar()
					}
					m.state = stateOperating
					return m, nil
				case 2: // Treinar e Operar
					m.isAutoOperating = true
					m.state = stateTraining
					m.currentStepIdx = 0

					m.preparaTreinamento()

					return m, tea.Batch(m.spinner.Tick, tickTraining())
				case 3: // Sair
					return m, tea.Quit
				}
			} else if m.state == stateTraining {
				if m.currentStepIdx >= len(m.trainingSteps) {
					m.redeTreinada = true
					m.state = stateTrainingDone
					return m, nil
				}
			} else if m.state == stateTrainingDone || m.state == stateOperating {
				if m.state == stateTrainingDone && m.isAutoOperating {
					m.resultadoOperacao = m.operar()
					m.state = stateOperating
					return m, nil
				}
				m.state = stateMenu
			}
		}

	case trainingTickMsg:
		if m.state == stateTraining {
			if m.currentStepIdx < len(m.trainingSteps) {
				m.currentStepIdx++
				return m, tickTraining()
			}
			// Aguarda o usuário pressionar Enter (tratado no KeyMsg)
			return m, nil
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	return m, nil
}

// View é responsável por 'printar' (escrever os pixels da TUI) baseando-se no estado.
func (m model) View() string {
	var sb strings.Builder

	// Header
	sb.WriteString(titleStyle.Render(" PERCEPTRON - RECONHECIMENTO DE LETRAS "))
	sb.WriteString("\n")
	sb.WriteString(subtitleStyle.Render("Redes Neurais Artificiais - Trabalho 02"))
	sb.WriteString("\n\n")

	switch m.state {
	case stateMenu:
		sb.WriteString("O que você deseja fazer?\n\n")

		for i, choice := range m.choices {
			cursor := " "
			var style lipgloss.Style
			if m.cursor == i {
				cursor = ">"
				style = selectedItemStyle
			} else {
				style = menuItemStyle
			}
			sb.WriteString(style.Render(fmt.Sprintf("%s %s", cursor, choice)) + "\n")
		}

		sb.WriteString("\n(Use setas para mover, Enter para selecionar, q para sair)")

	case stateTraining:
		if m.currentStepIdx < len(m.trainingSteps) {
			sb.WriteString(fmt.Sprintf("%s Treinando a rede...\n\n", m.spinner.View()))
		} else {
			sb.WriteString(successStyle.Render("✓ Treinamento finalizado.") + "\n\n")
		}

		// Chama as funções do main.go pra montar a matriz fixada na tela superior
		strA := "Letra A (target = -1):\n\n" + formataLetra(letraA())
		strB := "Letra B (target = 1):\n\n" + formataLetra(letraB())
		sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, boxStyle.Render(strA), boxStyle.Render(strB)))
		sb.WriteString("\n\n")

		// Evita encher o buffer do terminal, renderiza apenas as últimas 5 execuções (logs de epochs)
		startIdx := 0
		if m.currentStepIdx > 5 {
			startIdx = m.currentStepIdx - 5
		}

		// Mostra histórico com as cores do Lipgloss
		for i := startIdx; i < m.currentStepIdx; i++ {
			step := m.trainingSteps[i]
			sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render(strings.Repeat("-", 50)) + "\n")
			sb.WriteString(fmt.Sprintf("Ciclo %d | Amostra: Letra %s\n", step.ciclo, step.amostra))
			sb.WriteString(fmt.Sprintf("y_in (potencial) = %s\n", infoStyle.Render(fmt.Sprintf("%.4f", step.yIn))))
			sb.WriteString(fmt.Sprintf("y    (ativacao)  = %d\n", step.y))
			sb.WriteString(fmt.Sprintf("target           = %d\n", step.target))

			if step.teveErro {
				sb.WriteString(errorStyle.Render("Resultado: ERRO  → atualizando pesos") + "\n")
				sb.WriteString(fmt.Sprintf("delta = %.4f | novo bias = %.4f\n", step.delta, step.novoBias))
			} else {
				sb.WriteString(successStyle.Render("Resultado: OK    → pesos mantidos") + "\n")
			}
		}

		if m.currentStepIdx == len(m.trainingSteps) {
			sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render(strings.Repeat("=", 50)) + "\n")
			sb.WriteString(successStyle.Render(fmt.Sprintf("Ciclo %d completo — nenhum erro. Convergência!", m.ciclosTreino)) + "\n\n")
			sb.WriteString("Pressione Enter para prosseguir.")
		}

	case stateTrainingDone:
		sb.WriteString(successStyle.Render("✓ Rede Treinada com Sucesso!"))
		sb.WriteString(fmt.Sprintf("\n\nConvergência alcançada em %s ciclos.", infoStyle.Render(fmt.Sprintf("%d", m.ciclosTreino))))
		sb.WriteString(fmt.Sprintf("\nBias final ajustado: %s\n\n", infoStyle.Render(fmt.Sprintf("%.4f", m.bias))))

		if m.isAutoOperating {
			sb.WriteString("Pressione Enter para ver os testes de operação da rede.")
		} else {
			sb.WriteString("Pressione Enter para voltar ao menu.")
		}

	case stateOperating:
		sb.WriteString("--- Teste Final ---\n\n")
		if m.redeTreinada {
			sb.WriteString(fmt.Sprintf("Bias utilizado: %.4f\n\n", m.bias))
		}
		sb.WriteString(m.resultadoOperacao)
		sb.WriteString("\n\nPressione Enter para voltar ao menu.")
	}

	return lipgloss.NewStyle().Padding(1, 2).Render(sb.String())
}

// iniciarTUI é chamada pelo main()
func iniciarTUI() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Erro ao iniciar Bubble Tea: %v\n", err)
		os.Exit(1)
	}
}
