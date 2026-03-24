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
// Estilos Lipgloss — paleta visual da TUI
// =============================================================================

var (
	// Título principal no topo da tela
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 2).
			MarginBottom(1)

	// Subtítulo com nome da disciplina
	subtitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#A3A3A3")).
			MarginBottom(1)

	// Item do menu (não selecionado)
	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(lipgloss.Color("#CCCCCC"))

	// Item do menu (selecionado)
	selectedItemStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(lipgloss.Color("#00E676")).
				Bold(true)

	// Caixa com borda arredondada para exibir as letras
	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#7D56F4")).
			Padding(1, 2).
			MarginRight(2)

	// Estilo para mensagens de sucesso
	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00E676")).Bold(true)

	// Estilo para mensagens de erro
	errorStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5252")).Bold(true)

	// Estilo para informações numéricas destacadas
	infoStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#64B5F6"))

	// Estilo para separadores
	dimStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))

	// Estilo para hints de navegação
	hintStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#888888")).Italic(true)
)

// =============================================================================
// Estados e estruturas do Bubble Tea
// =============================================================================

// sessionState controla em qual tela a TUI está
type sessionState int

const (
	stateMenu         sessionState = iota // menu principal
	stateTraining                         // treinando passo a passo
	stateTrainingDone                     // treino concluído
	stateOperating                        // exibindo resultado da operação
)

// trainingStep guarda os dados de um passo do treinamento para a TUI animar
type trainingStep struct {
	ciclo    int
	amostra  string
	target   int
	yLiq     float64
	y        int
	delta    float64
	novoBias float64
	teveErro bool
}

// model é o estado central do Bubble Tea
type model struct {
	state   sessionState
	cursor  int
	choices []string

	spinner spinner.Model

	pesos        [N_ENTRADAS]float64
	bias         float64
	redeTreinada bool
	ciclosTreino int

	// Animação passo a passo do treinamento
	trainingSteps  []trainingStep
	currentStepIdx int

	// Flag: se viemos de "Treinar e Operar", já opera automaticamente
	isAutoOperating bool

	resultadoOperacao string
}

// =============================================================================
// Inicialização do Bubble Tea
// =============================================================================

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	return model{
		state: stateMenu,
		choices: []string{
			"⚡ Treinar a rede passo a passo",
			"🔍 Operar a rede",
			"🚀 Treinar e Mostrar rede",
			"🚪 Sair",
		},
		spinner: s,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Mensagem de tick para animação do treinamento
// =============================================================================

type trainingTickMsg time.Time

func tickTraining() tea.Cmd {
	return tea.Tick(time.Millisecond*500, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

// =============================================================================
// Update — trata todas as mensagens (teclas, ticks, etc.)
// =============================================================================

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
			if m.state == stateMenu && m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.state == stateMenu && m.cursor < len(m.choices)-1 {
				m.cursor++
			}
		case "enter", " ":
			if m.state == stateMenu {
				switch m.cursor {
				case 0: // Treinar passo a passo
					m.isAutoOperating = false
					m.state = stateTraining
					m.currentStepIdx = 0
					m.preparaTreinamento()
					return m, tea.Batch(m.spinner.Tick, tickTraining())

				case 1: // Operar
					if !m.redeTreinada {
						m.resultadoOperacao = errorStyle.Render("⚠ ERRO:") +
							" A rede ainda não foi treinada!\n  Treine-a primeiro (Opção 1)."
					} else {
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
				// Se a animação já terminou, avança
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
			return m, nil
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	return m, nil
}

// =============================================================================
// View — renderiza a interface baseando-se no estado atual
// =============================================================================

func (m model) View() string {
	var sb strings.Builder

	// ─── Header ───
	sb.WriteString(titleStyle.Render(" 🧠 PERCEPTRON — RECONHECIMENTO DE LETRAS "))
	sb.WriteString("\n")
	sb.WriteString(subtitleStyle.Render("Redes Neurais Artificiais • Trabalho 02"))
	sb.WriteString("\n\n")

	switch m.state {

	// ─── MENU PRINCIPAL ───
	case stateMenu:
		sb.WriteString("O que você deseja fazer?\n\n")

		for i, choice := range m.choices {
			cursor := "  "
			var style lipgloss.Style
			if m.cursor == i {
				cursor = "▸ "
				style = selectedItemStyle
			} else {
				style = menuItemStyle
			}
			sb.WriteString(style.Render(cursor+choice) + "\n")
		}

		sb.WriteString("\n")
		sb.WriteString(hintStyle.Render("  ↑↓ mover • Enter selecionar • q sair"))

	// ─── TREINAMENTO PASSO A PASSO ───
	case stateTraining:
		if m.currentStepIdx < len(m.trainingSteps) {
			sb.WriteString(fmt.Sprintf("%s Treinando a rede...\n\n", m.spinner.View()))
		} else {
			sb.WriteString(successStyle.Render("✓ Treinamento finalizado!") + "\n\n")
		}

		// Letras A e B lado a lado em caixas
		strA := infoStyle.Render("Letra A") + " (target = -1)\n\n" + formataLetra(letraA())
		strB := infoStyle.Render("Letra B") + " (target =  1)\n\n" + formataLetra(letraB())
		sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, boxStyle.Render(strA), boxStyle.Render(strB)))
		sb.WriteString("\n\n")

		// Mostra apenas os últimos 6 passos para não encher a tela
		startIdx := 0
		if m.currentStepIdx > 6 {
			startIdx = m.currentStepIdx - 6
		}

		for i := startIdx; i < m.currentStepIdx; i++ {
			step := m.trainingSteps[i]
			sb.WriteString(dimStyle.Render(strings.Repeat("─", 52)) + "\n")
			sb.WriteString(fmt.Sprintf("  Ciclo %s │ Amostra: Letra %s\n",
				infoStyle.Render(fmt.Sprintf("%d", step.ciclo)),
				infoStyle.Render(step.amostra)))
			sb.WriteString(fmt.Sprintf("  y_in (potencial) = %s\n",
				infoStyle.Render(fmt.Sprintf("%.4f", step.yLiq))))
			sb.WriteString(fmt.Sprintf("  y    (ativação)  = %d\n", step.y))
			sb.WriteString(fmt.Sprintf("  target           = %d\n", step.target))

			if step.teveErro {
				sb.WriteString("  " + errorStyle.Render("✗ ERRO → atualizando pesos") + "\n")
				sb.WriteString(fmt.Sprintf("  delta = %.4f │ novo bias = %.4f\n", step.delta, step.novoBias))
			} else {
				sb.WriteString("  " + successStyle.Render("✓ OK   → pesos mantidos") + "\n")
			}
		}

		if m.currentStepIdx == len(m.trainingSteps) {
			sb.WriteString(dimStyle.Render(strings.Repeat("═", 52)) + "\n")
			sb.WriteString(successStyle.Render(
				fmt.Sprintf("  Ciclo %d completo — nenhum erro. Convergência!", m.ciclosTreino)) + "\n\n")
			sb.WriteString(hintStyle.Render("  Pressione Enter para prosseguir"))
		}

	// ─── TREINO CONCLUÍDO ───
	case stateTrainingDone:
		sb.WriteString(successStyle.Render("✓ Rede Treinada com Sucesso!"))
		sb.WriteString(fmt.Sprintf("\n\n  Convergência alcançada em %s ciclos.",
			infoStyle.Render(fmt.Sprintf("%d", m.ciclosTreino))))
		sb.WriteString(fmt.Sprintf("\n  Bias final ajustado: %s\n\n",
			infoStyle.Render(fmt.Sprintf("%.4f", m.bias))))

		if m.isAutoOperating {
			sb.WriteString(hintStyle.Render("  Pressione Enter para ver os testes de operação da rede"))
		} else {
			sb.WriteString(hintStyle.Render("  Pressione Enter para voltar ao menu"))
		}

	// ─── OPERAÇÃO / TESTE ───
	case stateOperating:
		sb.WriteString(infoStyle.Render("── Teste Final ──") + "\n\n")
		if m.redeTreinada {
			sb.WriteString(fmt.Sprintf("  Bias utilizado: %s\n\n",
				infoStyle.Render(fmt.Sprintf("%.4f", m.bias))))
		}

		// Letras exibidas junto ao resultado
		strA := infoStyle.Render("Letra A") + "\n\n" + formataLetra(letraA())
		strB := infoStyle.Render("Letra B") + "\n\n" + formataLetra(letraB())
		sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, boxStyle.Render(strA), boxStyle.Render(strB)))
		sb.WriteString("\n\n")

		sb.WriteString(m.resultadoOperacao)
		sb.WriteString("\n\n")
		sb.WriteString(hintStyle.Render("  Pressione Enter para voltar ao menu"))
	}

	return lipgloss.NewStyle().Padding(1, 2).Render(sb.String())
}

// =============================================================================
// iniciarTUI — ponto de entrada da interface, chamada pelo main()
// =============================================================================

func iniciarTUI() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Erro ao iniciar Bubble Tea: %v\n", err)
		os.Exit(1)
	}
}
