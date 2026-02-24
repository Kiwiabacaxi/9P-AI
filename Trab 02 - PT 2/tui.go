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
// Estilos Lipgloss
// =============================================================================

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FAFAFA")).
			Background(lipgloss.Color("#7D56F4")).
			Padding(0, 2).
			MarginBottom(1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#A3A3A3"))

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
			Padding(1, 2)

	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00E676")).Bold(true)
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5252")).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFC107")).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#64B5F6"))
	dimStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	hintStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#888888")).Italic(true)
	boldStyle    = lipgloss.NewStyle().Bold(true)
)

// =============================================================================
// Estados do Bubble Tea
// =============================================================================

type sessionState int

const (
	stateMenu     sessionState = iota // escolher porta
	stateTraining                     // animação do treino
	stateResult                       // pesos finais + tabela de testes
)

type model struct {
	state   sessionState
	cursor  int
	choices []string
	portas  []Porta

	spinner spinner.Model

	resultado      *resultadoTreino
	currentStepIdx int
}

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	portas := todasAsPortas()
	choices := make([]string, len(portas)+2)
	for i, p := range portas {
		extra := ""
		if p.nome == "XOR" {
			extra = " ⚠"
		}
		choices[i] = fmt.Sprintf("Porta %s%s", p.nome, extra)
	}
	choices[len(portas)] = "🚀 Treinar TODAS"
	choices[len(portas)+1] = "🚪 Sair"

	return model{
		state:   stateMenu,
		choices: choices,
		portas:  portas,
		spinner: s,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Tick de animação
// =============================================================================

type trainingTickMsg time.Time

func tickTraining() tea.Cmd {
	return tea.Tick(time.Millisecond*200, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

// Mensagem usada pra treinar a próxima porta na sequência "Todas"
type nextPortaMsg struct{}

// =============================================================================
// Update
// =============================================================================

// Estado auxiliar pra treinar todas em sequência
var (
	treinandoTodas  bool
	portaAtualIdx   int
	resultadosTodas []resultadoTreino
)

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		case "esc":
			if m.state != stateMenu {
				m.state = stateMenu
				treinandoTodas = false
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
				totalPortas := len(m.portas)

				if m.cursor < totalPortas {
					// Treinar uma porta individual
					treinandoTodas = false
					r := treinarPorta(m.portas[m.cursor])
					m.resultado = &r
					m.currentStepIdx = 0
					m.state = stateTraining
					return m, tea.Batch(m.spinner.Tick, tickTraining())

				} else if m.cursor == totalPortas {
					// Treinar TODAS em sequência
					treinandoTodas = true
					portaAtualIdx = 0
					resultadosTodas = nil
					r := treinarPorta(m.portas[0])
					m.resultado = &r
					m.currentStepIdx = 0
					m.state = stateTraining
					return m, tea.Batch(m.spinner.Tick, tickTraining())

				} else {
					return m, tea.Quit
				}

			} else if m.state == stateTraining {
				// Skip: pula direto pro resultado
				if m.resultado != nil {
					m.currentStepIdx = len(m.resultado.steps)
					m.state = stateResult
					return m, nil
				}
			} else if m.state == stateResult {
				if treinandoTodas {
					resultadosTodas = append(resultadosTodas, *m.resultado)
					portaAtualIdx++
					if portaAtualIdx < len(m.portas) {
						r := treinarPorta(m.portas[portaAtualIdx])
						m.resultado = &r
						m.currentStepIdx = 0
						m.state = stateTraining
						return m, tea.Batch(m.spinner.Tick, tickTraining())
					}
					// Todas concluídas — volta ao menu
					treinandoTodas = false
				}
				m.state = stateMenu
			}
		}

	case trainingTickMsg:
		if m.state == stateTraining && m.resultado != nil {
			if m.currentStepIdx < len(m.resultado.steps) {
				m.currentStepIdx++
				// Pula batches inteiros se tiver muitos steps (pra não ficar lento no XOR)
				if len(m.resultado.steps) > 40 && m.currentStepIdx < len(m.resultado.steps)-8 {
					// Avança mais rápido
					skip := len(m.resultado.steps) / 20
					if skip < 4 {
						skip = 4
					}
					m.currentStepIdx += skip
					if m.currentStepIdx > len(m.resultado.steps) {
						m.currentStepIdx = len(m.resultado.steps)
					}
				}
				return m, tickTraining()
			}
			// Animação acabou
			m.state = stateResult
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
// View
// =============================================================================

func (m model) View() string {
	var sb strings.Builder

	sb.WriteString(titleStyle.Render(" ⚡ PERCEPTRON — PORTAS LÓGICAS "))
	sb.WriteString("\n")
	sb.WriteString(subtitleStyle.Render("Redes Neurais Artificiais • Trabalho 02 - Parte 02"))
	sb.WriteString("\n\n")

	switch m.state {

	// ─── MENU ───
	case stateMenu:
		sb.WriteString("Qual porta lógica deseja treinar?\n\n")

		for i, choice := range m.choices {
			cursor := "  "
			style := menuItemStyle
			if m.cursor == i {
				cursor = "▸ "
				style = selectedItemStyle
			}
			sb.WriteString(style.Render(cursor+choice) + "\n")
		}

		sb.WriteString("\n")
		sb.WriteString(hintStyle.Render("  ↑↓ mover • Enter selecionar • q sair"))

		// Mostra resumo se acabamos de treinar todas
		if len(resultadosTodas) > 0 {
			sb.WriteString("\n\n")
			sb.WriteString(boxStyle.Render(renderResumoTodas()))
		}

	// ─── TREINAMENTO ───
	case stateTraining:
		if m.resultado == nil {
			break
		}
		r := m.resultado
		porta := r.porta

		if m.currentStepIdx < len(r.steps) {
			sb.WriteString(fmt.Sprintf("%s Treinando porta %s...\n\n",
				m.spinner.View(), boldStyle.Render(porta.nome)))
		} else {
			sb.WriteString(successStyle.Render(fmt.Sprintf("✓ Porta %s — treinamento finalizado!", porta.nome)) + "\n\n")
		}

		// Tabela verdade
		sb.WriteString(boxStyle.Render(formataTabela(porta)))
		sb.WriteString("\n\n")

		// Últimos N passos
		startIdx := 0
		visible := 8
		if m.currentStepIdx > visible {
			startIdx = m.currentStepIdx - visible
		}
		for i := startIdx; i < m.currentStepIdx && i < len(r.steps); i++ {
			step := r.steps[i]
			sb.WriteString(dimStyle.Render(strings.Repeat("─", 50)) + "\n")
			sb.WriteString(fmt.Sprintf("  Ciclo %s │ [%2d, %2d] → target: %2d\n",
				infoStyle.Render(fmt.Sprintf("%d", step.ciclo)),
				step.x1, step.x2, step.target))
			sb.WriteString(fmt.Sprintf("  y_in = %s │ y = %d\n",
				infoStyle.Render(fmt.Sprintf("%.4f", step.yLiq)), step.y))
			if step.teveErro {
				sb.WriteString("  " + errorStyle.Render("✗ erro → pesos corrigidos") + "\n")
			} else {
				sb.WriteString("  " + successStyle.Render("✓ ok") + "\n")
			}
		}

		sb.WriteString("\n")
		sb.WriteString(hintStyle.Render("  Enter para pular animação"))

	// ─── RESULTADO ───
	case stateResult:
		if m.resultado == nil {
			break
		}
		r := m.resultado

		if r.convergiu {
			sb.WriteString(successStyle.Render(fmt.Sprintf("✓ Porta %s — Convergiu!", r.porta.nome)))
		} else {
			sb.WriteString(warnStyle.Render(fmt.Sprintf("⚠ Porta %s — NÃO convergiu! (limite: %d ciclos)", r.porta.nome, MAX_CICLOS)))
			sb.WriteString("\n")
			sb.WriteString(dimStyle.Render("  O XOR não é linearmente separável — Perceptron simples não resolve."))
		}

		sb.WriteString("\n\n")

		// Pesos finais em destaque
		pesosBox := fmt.Sprintf(
			"%s\n\n  W1   = %s\n  W2   = %s\n  Bias = %s\n  Ciclos: %s",
			boldStyle.Render("Pesos Finais"),
			infoStyle.Render(fmt.Sprintf("%.4f", r.w1)),
			infoStyle.Render(fmt.Sprintf("%.4f", r.w2)),
			infoStyle.Render(fmt.Sprintf("%.4f", r.bias)),
			infoStyle.Render(fmt.Sprintf("%d", r.ciclos)),
		)
		sb.WriteString(boxStyle.Render(pesosBox))
		sb.WriteString("\n\n")

		// Tabela de testes
		sb.WriteString(boldStyle.Render("Teste com pesos aprendidos:") + "\n\n")
		acertos := 0
		for _, t := range r.testes {
			status := successStyle.Render("✓")
			if !t.acertou {
				status = errorStyle.Render("✗")
			} else {
				acertos++
			}
			sb.WriteString(fmt.Sprintf("  [%2d, %2d] │ alvo: %2d │ pred: %2d │ y_in: %6.3f  %s\n",
				t.x1, t.x2, t.target, t.predicao, t.yLiq, status))
		}
		sb.WriteString(fmt.Sprintf("\n  Acurácia: %s\n",
			infoStyle.Render(fmt.Sprintf("%d/4 (%.0f%%)", acertos, float64(acertos)/4*100))))

		sb.WriteString("\n")
		if treinandoTodas && portaAtualIdx < len(m.portas)-1 {
			sb.WriteString(hintStyle.Render(fmt.Sprintf("  Enter para próxima porta (%s)",
				m.portas[portaAtualIdx+1].nome)))
		} else {
			sb.WriteString(hintStyle.Render("  Enter para voltar ao menu"))
		}
	}

	return lipgloss.NewStyle().Padding(1, 2).Render(sb.String())
}

// renderResumoTodas mostra um resumo de todas as portas treinadas.
func renderResumoTodas() string {
	var sb strings.Builder
	sb.WriteString(boldStyle.Render("📊 Resumo — Todas as Portas") + "\n\n")
	sb.WriteString("  Porta │   W1    │   W2    │  Bias   │ Ciclos │ Status\n")
	sb.WriteString("  ──────┼─────────┼─────────┼─────────┼────────┼────────\n")

	for _, r := range resultadosTodas {
		status := successStyle.Render("  ✓")
		if !r.convergiu {
			status = errorStyle.Render("  ✗")
		}
		sb.WriteString(fmt.Sprintf("  %-5s │ %7.4f │ %7.4f │ %7.4f │ %6d │%s\n",
			r.porta.nome, r.w1, r.w2, r.bias, r.ciclos, status))
	}
	return sb.String()
}

// =============================================================================
// iniciarTUI — ponto de entrada
// =============================================================================

func iniciarTUI() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Erro ao iniciar TUI: %v\n", err)
		os.Exit(1)
	}
}
