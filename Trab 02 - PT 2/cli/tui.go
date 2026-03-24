package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// Paleta de cores — neon/cyberpunk
// =============================================================================

var (
	// Cores principais
	neonCyan    = lipgloss.Color("#00FFFF")
	neonMagenta = lipgloss.Color("#FF00FF")
	neonGreen   = lipgloss.Color("#39FF14")
	neonPink    = lipgloss.Color("#FF6EC7")
	neonYellow  = lipgloss.Color("#FFE600")
	softWhite   = lipgloss.Color("#E0E0E0")
	dimGray     = lipgloss.Color("#555555")
	darkBg      = lipgloss.Color("#1A1A2E")
)

// =============================================================================
// Estilos Lipgloss — tema neon
// =============================================================================

var (
	// Header com gradiente magenta → cyan
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#1A1A2E")).
			Background(neonCyan).
			Padding(0, 2).
			MarginBottom(1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(neonMagenta).
			Italic(true)

	// Menu
	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(softWhite)

	selectedItemStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(neonCyan).
				Bold(true)

	// Caixa com borda dupla neon
	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForeground(neonMagenta).
			Padding(1, 2)

	// Status
	successStyle = lipgloss.NewStyle().Foreground(neonGreen).Bold(true)
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF3030")).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(neonYellow).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(neonCyan)
	dimStyle     = lipgloss.NewStyle().Foreground(dimGray)
	hintStyle    = lipgloss.NewStyle().Foreground(dimGray).Italic(true)
	boldStyle    = lipgloss.NewStyle().Bold(true).Foreground(softWhite)

	// Estilo especial pro peso final
	weightStyle = lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	labelStyle  = lipgloss.NewStyle().Foreground(neonPink)
)

// =============================================================================
// Estados do Bubble Tea
// =============================================================================

type sessionState int

const (
	stateMenu sessionState = iota
	stateTraining
	stateResult
)

type model struct {
	state   sessionState
	cursor  int
	choices []string
	portas  []Porta

	spinner  spinner.Model
	progress progress.Model

	resultado      *resultadoTreino
	currentStepIdx int
}

func initialModel() model {
	// Spinner neon
	s := spinner.New()
	s.Spinner = spinner.Jump
	s.Style = lipgloss.NewStyle().Foreground(neonCyan)

	// Progress bar com gradiente magenta → cyan
	p := progress.New(
		progress.WithGradient("#FF00FF", "#00FFFF"),
		progress.WithWidth(40),
		progress.WithoutPercentage(),
	)

	portas := todasAsPortas()
	choices := make([]string, len(portas)+2)
	for i, p := range portas {
		icon := "⚡"
		if p.nome == "XOR" {
			icon = "⚠ "
		}
		choices[i] = fmt.Sprintf("%s Porta %s", icon, p.nome)
	}
	choices[len(portas)] = "🚀 Treinar TODAS"
	choices[len(portas)+1] = "⏻  Sair"

	return model{
		state:    stateMenu,
		choices:  choices,
		portas:   portas,
		spinner:  s,
		progress: p,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Animação
// =============================================================================

type trainingTickMsg time.Time

func tickTraining() tea.Cmd {
	return tea.Tick(time.Millisecond*150, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

type nextPortaMsg struct{}

// =============================================================================
// Update
// =============================================================================

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
					treinandoTodas = false
					r := treinarPorta(m.portas[m.cursor])
					m.resultado = &r
					m.currentStepIdx = 0
					m.state = stateTraining
					return m, tea.Batch(m.spinner.Tick, tickTraining())

				} else if m.cursor == totalPortas {
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
					treinandoTodas = false
				}
				m.state = stateMenu
			}
		}

	case trainingTickMsg:
		if m.state == stateTraining && m.resultado != nil {
			if m.currentStepIdx < len(m.resultado.steps) {
				m.currentStepIdx++
				if len(m.resultado.steps) > 40 && m.currentStepIdx < len(m.resultado.steps)-8 {
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

	// ─── HEADER ───
	sb.WriteString(titleStyle.Render(" ⚡ PERCEPTRON — PORTAS LÓGICAS "))
	sb.WriteString("\n")
	sb.WriteString(subtitleStyle.Render("  Redes Neurais Artificiais • Trabalho 02 — Parte 02"))
	sb.WriteString("\n\n")

	switch m.state {

	// ─── MENU ───
	case stateMenu:
		sb.WriteString(lipgloss.NewStyle().Foreground(softWhite).Render("Qual porta lógica deseja treinar?") + "\n\n")

		for i, choice := range m.choices {
			cursor := "  "
			style := menuItemStyle
			if m.cursor == i {
				cursor = lipgloss.NewStyle().Foreground(neonCyan).Render("▸ ")
				style = selectedItemStyle
			}
			sb.WriteString(style.Render(cursor+choice) + "\n")
		}

		sb.WriteString("\n")
		sb.WriteString(hintStyle.Render("  ↑↓ mover • Enter selecionar • q sair"))

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
		total := len(r.steps)

		if m.currentStepIdx < total {
			// Spinner + nome da porta
			sb.WriteString(fmt.Sprintf("%s Treinando porta %s...\n\n",
				m.spinner.View(),
				lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render(porta.nome)))

			// ── BARRA DE PROGRESSO ──
			pct := float64(m.currentStepIdx) / float64(total)
			sb.WriteString("  " + m.progress.ViewAs(pct))
			sb.WriteString(fmt.Sprintf("  %s\n\n",
				dimStyle.Render(fmt.Sprintf("%d/%d steps", m.currentStepIdx, total))))
		} else {
			sb.WriteString(successStyle.Render(fmt.Sprintf("✓ Porta %s — treinamento completo!", porta.nome)) + "\n\n")
			sb.WriteString("  " + m.progress.ViewAs(1.0))
			sb.WriteString(fmt.Sprintf("  %s\n\n",
				successStyle.Render(fmt.Sprintf("%d/%d steps", total, total))))
		}

		// Tabela verdade
		sb.WriteString(boxStyle.Render(formataTabela(porta)))
		sb.WriteString("\n\n")

		// Log dos últimos passos
		startIdx := 0
		visible := 6
		if m.currentStepIdx > visible {
			startIdx = m.currentStepIdx - visible
		}
		for i := startIdx; i < m.currentStepIdx && i < total; i++ {
			step := r.steps[i]
			sb.WriteString(dimStyle.Render(strings.Repeat("─", 52)) + "\n")
			sb.WriteString(fmt.Sprintf("  Ciclo %s │ [%2d, %2d] → alvo: %2d\n",
				infoStyle.Render(fmt.Sprintf("%d", step.ciclo)),
				step.x1, step.x2, step.target))
			sb.WriteString(fmt.Sprintf("  y_in = %s │ y = %d\n",
				weightStyle.Render(fmt.Sprintf("%.4f", step.yLiq)), step.y))
			if step.teveErro {
				sb.WriteString("  " + errorStyle.Render("✗ erro → pesos corrigidos") + "\n")
			} else {
				sb.WriteString("  " + successStyle.Render("✓ ok") + "\n")
			}
		}

		sb.WriteString("\n")
		sb.WriteString(hintStyle.Render("  Enter pula animação • Esc volta"))

	// ─── RESULTADO ───
	case stateResult:
		if m.resultado == nil {
			break
		}
		r := m.resultado

		if r.convergiu {
			sb.WriteString(successStyle.Render(fmt.Sprintf("✓ Porta %s — Convergiu!", r.porta.nome)))
		} else {
			sb.WriteString(warnStyle.Render(fmt.Sprintf("⚠ Porta %s — NÃO convergiu! (%d ciclos)", r.porta.nome, MAX_CICLOS)))
			sb.WriteString("\n")
			sb.WriteString(dimStyle.Render("  XOR não é linearmente separável — Perceptron simples não resolve."))
		}

		sb.WriteString("\n\n")

		// Pesos finais com labels rosas e valores cyan
		pesosBox := fmt.Sprintf(
			"%s\n\n  %s %s\n  %s %s\n  %s %s\n  %s %s",
			lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("⬡ Pesos Finais"),
			labelStyle.Render("W1   ="),
			weightStyle.Render(fmt.Sprintf("%+.4f", r.w1)),
			labelStyle.Render("W2   ="),
			weightStyle.Render(fmt.Sprintf("%+.4f", r.w2)),
			labelStyle.Render("Bias ="),
			weightStyle.Render(fmt.Sprintf("%+.4f", r.bias)),
			labelStyle.Render("Ciclos:"),
			weightStyle.Render(fmt.Sprintf("%d", r.ciclos)),
		)
		sb.WriteString(boxStyle.Render(pesosBox))
		sb.WriteString("\n\n")

		// Teste
		sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("⬡ Teste com Pesos Aprendidos") + "\n\n")
		acertos := 0
		for _, t := range r.testes {
			status := successStyle.Render("✓")
			if !t.acertou {
				status = errorStyle.Render("✗")
			} else {
				acertos++
			}
			sb.WriteString(fmt.Sprintf("  [%2d, %2d] │ alvo: %2d │ pred: %2d │ y_in: %s  %s\n",
				t.x1, t.x2, t.target, t.predicao,
				infoStyle.Render(fmt.Sprintf("%6.3f", t.yLiq)), status))
		}

		pctStr := fmt.Sprintf("%d/4 (%.0f%%)", acertos, float64(acertos)/4*100)
		if acertos == 4 {
			sb.WriteString(fmt.Sprintf("\n  Acurácia: %s\n", successStyle.Render(pctStr)))
		} else {
			sb.WriteString(fmt.Sprintf("\n  Acurácia: %s\n", warnStyle.Render(pctStr)))
		}

		sb.WriteString("\n")
		if treinandoTodas && portaAtualIdx < len(m.portas)-1 {
			sb.WriteString(hintStyle.Render(fmt.Sprintf("  Enter → próxima porta (%s) • Esc volta",
				m.portas[portaAtualIdx+1].nome)))
		} else {
			sb.WriteString(hintStyle.Render("  Enter → menu • Esc sair"))
		}
	}

	return lipgloss.NewStyle().Padding(1, 2).Render(sb.String())
}

// renderResumoTodas mostra resumo comparativo de todas as portas.
func renderResumoTodas() string {
	var sb strings.Builder
	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("📊 Resumo — Todas as Portas") + "\n\n")

	// Header da tabela
	headerLine := fmt.Sprintf("  %s │ %s │ %s │ %s │ %s │ %s",
		labelStyle.Render("Porta"),
		labelStyle.Render("   W1   "),
		labelStyle.Render("   W2   "),
		labelStyle.Render("  Bias  "),
		labelStyle.Render("Ciclos"),
		labelStyle.Render("  "))
	sb.WriteString(headerLine + "\n")
	sb.WriteString(dimStyle.Render("  ──────┼──────────┼──────────┼──────────┼────────┼────") + "\n")

	for _, r := range resultadosTodas {
		status := successStyle.Render(" ✓")
		if !r.convergiu {
			status = errorStyle.Render(" ✗")
		}
		sb.WriteString(fmt.Sprintf("  %-5s │ %s │ %s │ %s │ %s │%s\n",
			lipgloss.NewStyle().Foreground(softWhite).Render(r.porta.nome),
			weightStyle.Render(fmt.Sprintf("%+8.4f", r.w1)),
			weightStyle.Render(fmt.Sprintf("%+8.4f", r.w2)),
			weightStyle.Render(fmt.Sprintf("%+8.4f", r.bias)),
			infoStyle.Render(fmt.Sprintf("%6d", r.ciclos)),
			status))
	}
	return sb.String()
}

// =============================================================================
// iniciarTUI
// =============================================================================

func iniciarTUI() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Erro ao iniciar TUI: %v\n", err)
		os.Exit(1)
	}
}
