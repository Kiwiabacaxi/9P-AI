package main

import (
	"fmt"
	"math"
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
// Estilos Lipgloss
// =============================================================================

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#1A1A2E")).
			Background(neonCyan).
			Padding(0, 2).
			MarginBottom(1)

	subtitleStyle = lipgloss.NewStyle().
			Foreground(neonMagenta).
			Italic(true)

	menuItemStyle = lipgloss.NewStyle().
			PaddingLeft(2).
			Foreground(softWhite)

	selectedItemStyle = lipgloss.NewStyle().
				PaddingLeft(2).
				Foreground(neonCyan).
				Bold(true)

	boxStyle = lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForeground(neonMagenta).
			Padding(1, 2)

	successStyle = lipgloss.NewStyle().Foreground(neonGreen).Bold(true)
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF3030")).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(neonYellow).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(neonCyan)
	dimStyle     = lipgloss.NewStyle().Foreground(dimGray)
	hintStyle    = lipgloss.NewStyle().Foreground(dimGray).Italic(true)
	weightStyle  = lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	labelStyle   = lipgloss.NewStyle().Foreground(neonPink)
)

// =============================================================================
// Estados
// =============================================================================

type sessionState int

const (
	stateMenu sessionState = iota
	stateTraining
	stateTrainingDone
	stateDrawLetter
	stateRecognition
)

type model struct {
	state   sessionState
	cursor  int
	choices []string
	spinner spinner.Model
	progress progress.Model

	resultado      *ResultadoTreino
	currentStepIdx int

	grade     [N_ENTRADAS]int
	cursorLin int
	cursorCol int

	letraReconhecida int
	yInsReconhecidos [N_LETRAS]float64
}

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Jump
	s.Style = lipgloss.NewStyle().Foreground(neonCyan)

	p := progress.New(
		progress.WithGradient("#FF00FF", "#00FFFF"),
		progress.WithWidth(40),
		progress.WithoutPercentage(),
	)

	choices := []string{
		"Treinar MADALINE",
		"Desenhar letra",
		"Sair",
	}

	// Inicializa grade com -1
	var grade [N_ENTRADAS]int
	for i := range grade {
		grade[i] = -1
	}

	return model{
		state:    stateMenu,
		choices:  choices,
		spinner:  s,
		progress: p,
		grade:    grade,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Animação
// =============================================================================

type trainingTickMsg time.Time
type trainingDoneMsg struct{}

func tickTraining() tea.Cmd {
	return tea.Tick(time.Millisecond*80, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

// =============================================================================
// Update
// =============================================================================

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.state == stateMenu {
				return m, tea.Quit
			}

		case "esc":
			switch m.state {
			case stateTraining:
				if m.resultado != nil {
					m.currentStepIdx = len(m.resultado.steps)
					m.state = stateTrainingDone
				}
			case stateTrainingDone, stateDrawLetter, stateRecognition:
				m.state = stateMenu
			default:
				return m, tea.Quit
			}
			return m, nil

		case "up", "k":
			switch m.state {
			case stateMenu:
				if m.cursor > 0 {
					m.cursor--
				}
			case stateDrawLetter:
				if m.cursorLin > 0 {
					m.cursorLin--
				}
			}

		case "down", "j":
			switch m.state {
			case stateMenu:
				if m.cursor < len(m.choices)-1 {
					m.cursor++
				}
			case stateDrawLetter:
				if m.cursorLin < N_LINHAS-1 {
					m.cursorLin++
				}
			}

		case "left", "h":
			if m.state == stateDrawLetter && m.cursorCol > 0 {
				m.cursorCol--
			}

		case "right", "l":
			if m.state == stateDrawLetter && m.cursorCol < N_COLUNAS-1 {
				m.cursorCol++
			}

		case " ":
			switch m.state {
			case stateMenu:
				return m.handleMenuEnter()
			case stateDrawLetter:
				// Toggle pixel
				idx := m.cursorLin*N_COLUNAS + m.cursorCol
				if m.grade[idx] == 1 {
					m.grade[idx] = -1
				} else {
					m.grade[idx] = 1
				}
			}

		case "enter":
			switch m.state {
			case stateMenu:
				return m.handleMenuEnter()
			case stateTraining:
				// Pula animação
				if m.resultado != nil {
					m.currentStepIdx = len(m.resultado.steps)
					m.state = stateTrainingDone
				}
			case stateTrainingDone:
				m.state = stateMenu
			case stateDrawLetter:
				// Reconhecer
				if m.resultado != nil {
					idx, yIns := reconhecer(m.resultado.rede, m.grade)
					m.letraReconhecida = idx
					m.yInsReconhecidos = yIns
					m.state = stateRecognition
				} else {
					m.state = stateMenu
				}
			case stateRecognition:
				m.state = stateDrawLetter
			}

		case "r", "R":
			if m.state == stateDrawLetter {
				for i := range m.grade {
					m.grade[i] = -1
				}
			}
		}

	case trainingTickMsg:
		if m.state == stateTraining && m.resultado != nil {
			total := len(m.resultado.steps)
			if m.currentStepIdx < total {
				// Aceleração: manter animação < 10s
				skip := 1
				if total > 500 {
					skip = total / 100
					if skip < 3 {
						skip = 3
					}
				}
				m.currentStepIdx += skip
				if m.currentStepIdx > total {
					m.currentStepIdx = total
				}
				if m.currentStepIdx >= total {
					m.state = stateTrainingDone
					return m, nil
				}
				return m, tickTraining()
			}
			m.state = stateTrainingDone
			return m, nil
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m model) handleMenuEnter() (tea.Model, tea.Cmd) {
	switch m.cursor {
	case 0: // Treinar
		r := treinarMADALINE()
		m.resultado = &r
		m.currentStepIdx = 0
		m.state = stateTraining
		return m, tea.Batch(m.spinner.Tick, tickTraining())
	case 1: // Desenhar letra
		if m.resultado == nil {
			// Treinar primeiro automaticamente
			r := treinarMADALINE()
			m.resultado = &r
		}
		for i := range m.grade {
			m.grade[i] = -1
		}
		m.cursorLin = 0
		m.cursorCol = 0
		m.state = stateDrawLetter
		return m, nil
	case 2: // Sair
		return m, tea.Quit
	}
	return m, nil
}

// =============================================================================
// View
// =============================================================================

func (m model) View() string {
	var sb strings.Builder

	sb.WriteString(titleStyle.Render(" MADALINE — RECONHECIMENTO DE LETRAS A–M "))
	sb.WriteString("\n")
	sb.WriteString(subtitleStyle.Render("  Redes Neurais Artificiais • Trabalho 03 — Parte 01"))
	sb.WriteString("\n\n")

	switch m.state {

	case stateMenu:
		sb.WriteString(renderMenu(m))

	case stateTraining:
		sb.WriteString(renderTraining(m))

	case stateTrainingDone:
		sb.WriteString(renderTrainingDone(m))

	case stateDrawLetter:
		sb.WriteString(renderDrawLetter(m))

	case stateRecognition:
		sb.WriteString(renderRecognition(m))
	}

	return lipgloss.NewStyle().Padding(1, 2).Render(sb.String())
}

// =============================================================================
// Render: Menu
// =============================================================================

func renderMenu(m model) string {
	var sb strings.Builder

	if m.resultado != nil {
		if m.resultado.convergiu {
			sb.WriteString(successStyle.Render("✓ Rede treinada e convergida"))
		} else {
			sb.WriteString(warnStyle.Render(fmt.Sprintf("⚠ Treinamento sem convergência (%d ciclos)", MAX_CICLOS)))
		}
		sb.WriteString("\n\n")
	} else {
		sb.WriteString(dimStyle.Render("Rede não treinada ainda.") + "\n\n")
	}

	sb.WriteString(lipgloss.NewStyle().Foreground(softWhite).Render("O que deseja fazer?") + "\n\n")

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
	return sb.String()
}

// =============================================================================
// Render: Treinamento (animação 3 colunas)
// =============================================================================

func renderTraining(m model) string {
	if m.resultado == nil {
		return ""
	}
	r := m.resultado
	total := len(r.steps)

	var sb strings.Builder

	if m.currentStepIdx >= total {
		m.state = stateTrainingDone
		return renderTrainingDone(m)
	}

	step := r.steps[m.currentStepIdx]

	// Progresso
	pct := float64(m.currentStepIdx) / float64(max(total, 1))
	sb.WriteString(fmt.Sprintf("%s Treinando MADALINE...\n\n", m.spinner.View()))
	sb.WriteString("  " + m.progress.ViewAs(pct))
	sb.WriteString(fmt.Sprintf("  %s\n\n",
		dimStyle.Render(fmt.Sprintf("step %d/%d", m.currentStepIdx, total))))

	// --- 3 colunas ---
	col1 := renderColEntrada(step)
	col2 := renderColADALINE(step)
	col3 := renderColSaida(step)

	row := lipgloss.JoinHorizontal(lipgloss.Top,
		boxStyle.Render(col1),
		"  ",
		boxStyle.Render(col2),
		"  ",
		boxStyle.Render(col3),
	)
	sb.WriteString(row)
	sb.WriteString("\n\n")

	// Status bar
	erroCount := 0
	for _, e := range step.erros {
		if e {
			erroCount++
		}
	}
	sb.WriteString(fmt.Sprintf("  Ciclo: %s  │  Letra: %s  │  Erros: %s\n",
		infoStyle.Render(fmt.Sprintf("%d", step.ciclo)),
		weightStyle.Render(nomesLetras[step.letraIdx]),
		func() string {
			if erroCount > 0 {
				return errorStyle.Render(fmt.Sprintf("%d ADALINEs corrigidas", erroCount))
			}
			return successStyle.Render("nenhum erro")
		}(),
	))

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  Enter pula animação • Esc avança para resultado"))
	return sb.String()
}

func renderColEntrada(step TrainingStep) string {
	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]

	var sb strings.Builder
	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("  ENTRADA") + "\n\n")
	sb.WriteString(formataLetraGrid(entrada))
	return sb.String()
}

func renderColADALINE(step TrainingStep) string {
	var sb strings.Builder
	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("  CAMADA ADALINE") + "\n\n")

	// Encontrar maior y_in absoluto para normalizar barras
	maxAbs := 0.01
	for _, v := range step.yIn {
		if abs := math.Abs(v); abs > maxAbs {
			maxAbs = abs
		}
	}

	// Encontrar argmax
	best := 0
	for j := 1; j < N_LETRAS; j++ {
		if step.yIn[j] > step.yIn[best] {
			best = j
		}
	}

	const barMaxLen = 12
	for j := 0; j < N_LETRAS; j++ {
		barLen := int(math.Abs(step.yIn[j]) / maxAbs * barMaxLen)
		if barLen < 1 {
			barLen = 1
		}
		bar := strings.Repeat("█", barLen)
		// Pad
		bar = fmt.Sprintf("%-12s", bar)

		yInStr := fmt.Sprintf("%+6.2f", step.yIn[j])

		var linha string
		if step.erros[j] {
			// Corrigida — neonCyan
			linha = fmt.Sprintf(" %s %s %s %s",
				lipgloss.NewStyle().Foreground(neonCyan).Bold(true).Render(nomesLetras[j]),
				lipgloss.NewStyle().Foreground(neonCyan).Render(bar),
				lipgloss.NewStyle().Foreground(neonCyan).Bold(true).Render(yInStr),
				lipgloss.NewStyle().Foreground(neonYellow).Render("← CORRIGIDA"),
			)
		} else if j == best {
			// Vencedor — neonGreen
			linha = fmt.Sprintf(" %s %s %s",
				lipgloss.NewStyle().Foreground(neonGreen).Bold(true).Render(nomesLetras[j]),
				lipgloss.NewStyle().Foreground(neonGreen).Render(bar),
				lipgloss.NewStyle().Foreground(neonGreen).Bold(true).Render(yInStr),
			)
		} else {
			linha = fmt.Sprintf(" %s %s %s",
				lipgloss.NewStyle().Foreground(softWhite).Render(nomesLetras[j]),
				lipgloss.NewStyle().Foreground(dimGray).Render(bar),
				dimStyle.Render(yInStr),
			)
		}
		sb.WriteString(linha + "\n")
	}
	return sb.String()
}

func renderColSaida(step TrainingStep) string {
	var sb strings.Builder
	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("  SAÍDA") + "\n\n")

	// argmax
	best := 0
	for j := 1; j < N_LETRAS; j++ {
		if step.yIn[j] > step.yIn[best] {
			best = j
		}
	}

	vencedor := nomesLetras[best]
	sb.WriteString(lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(neonGreen).
		Foreground(neonGreen).
		Bold(true).
		Padding(1, 3).
		Render(vencedor))
	sb.WriteString("\n\n")
	sb.WriteString(dimStyle.Render("argmax\ny_in"))

	return sb.String()
}

// =============================================================================
// Render: Treinamento Concluído
// =============================================================================

func renderTrainingDone(m model) string {
	if m.resultado == nil {
		return ""
	}
	r := m.resultado

	var sb strings.Builder

	if r.convergiu {
		sb.WriteString(successStyle.Render("✓ MADALINE convergiu!"))
	} else {
		sb.WriteString(warnStyle.Render(fmt.Sprintf("⚠ Não convergiu após %d ciclos", MAX_CICLOS)))
	}
	sb.WriteString("\n\n")

	// Resumo
	totalSteps := len(r.steps)
	infoBox := fmt.Sprintf(
		"%s\n\n  %s %s\n  %s %s\n  %s %s",
		lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("Resumo do Treinamento"),
		labelStyle.Render("Convergiu:  "),
		func() string {
			if r.convergiu {
				return successStyle.Render("SIM")
			}
			return errorStyle.Render("NÃO")
		}(),
		labelStyle.Render("Ciclos:     "),
		weightStyle.Render(fmt.Sprintf("%d", r.ciclos)),
		labelStyle.Render("Steps c/erro:"),
		weightStyle.Render(fmt.Sprintf("%d", totalSteps)),
	)
	sb.WriteString(boxStyle.Render(infoBox))
	sb.WriteString("\n\n")

	sb.WriteString(hintStyle.Render("  Enter → menu • Esc → menu"))
	return sb.String()
}

// =============================================================================
// Render: Desenho de Letra
// =============================================================================

func renderDrawLetter(m model) string {
	var sb strings.Builder

	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("Desenhe uma letra") + "\n\n")

	// Grade 5×7 interativa
	cursorStyle := lipgloss.NewStyle().Background(neonCyan).Foreground(darkBg).Bold(true)
	cursorActiveStyle := lipgloss.NewStyle().Background(neonYellow).Foreground(darkBg).Bold(true)
	pixelOnStyle := lipgloss.NewStyle().Foreground(neonPink)
	pixelOffStyle := lipgloss.NewStyle().Foreground(dimGray)

	for i := 0; i < N_LINHAS; i++ {
		sb.WriteString("  ")
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			idx := i*N_COLUNAS + j
			isCursor := (i == m.cursorLin && j == m.cursorCol)
			isOn := m.grade[idx] == 1

			if isCursor {
				if isOn {
					sb.WriteString(cursorActiveStyle.Render("█"))
				} else {
					sb.WriteString(cursorStyle.Render("·"))
				}
			} else if isOn {
				sb.WriteString(pixelOnStyle.Render("█"))
			} else {
				sb.WriteString(pixelOffStyle.Render("·"))
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  ↑↓←→ mover • Espaço toggle • R resetar • Enter reconhecer • Esc menu"))
	return sb.String()
}

// =============================================================================
// Render: Reconhecimento
// =============================================================================

func renderRecognition(m model) string {
	var sb strings.Builder

	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("Resultado do Reconhecimento") + "\n\n")

	// Letra desenhada vs reconhecida
	dataset := letrasDataset()
	letraRec := dataset[m.letraReconhecida]

	gradeDesenhada := formataLetraGrid(m.grade)
	gradeReconhecida := formataLetraGrid(letraRec)

	leftBox := boxStyle.Render(
		lipgloss.NewStyle().Foreground(softWhite).Bold(true).Render("Sua entrada") + "\n\n" + gradeDesenhada,
	)
	rightBox := boxStyle.Render(
		lipgloss.NewStyle().Foreground(neonGreen).Bold(true).Render("Reconhecida: "+nomesLetras[m.letraReconhecida]) + "\n\n" + gradeReconhecida,
	)

	row := lipgloss.JoinHorizontal(lipgloss.Top, leftBox, "    ", rightBox)
	sb.WriteString(row)
	sb.WriteString("\n\n")

	// Barras de confiança para todas as letras
	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).Render("Ativações y_in por letra:") + "\n\n")

	maxAbs := 0.01
	for _, v := range m.yInsReconhecidos {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}

	const barMax = 20
	for j := 0; j < N_LETRAS; j++ {
		barLen := int((m.yInsReconhecidos[j]+maxAbs) / (2 * maxAbs) * barMax)
		if barLen < 0 {
			barLen = 0
		}
		if barLen > barMax {
			barLen = barMax
		}
		bar := strings.Repeat("█", barLen)
		bar = fmt.Sprintf("%-20s", bar)

		yStr := fmt.Sprintf("%+7.3f", m.yInsReconhecidos[j])

		if j == m.letraReconhecida {
			sb.WriteString(fmt.Sprintf("  %s %s %s  %s\n",
				lipgloss.NewStyle().Foreground(neonGreen).Bold(true).Render(nomesLetras[j]),
				lipgloss.NewStyle().Foreground(neonGreen).Render(bar),
				weightStyle.Render(yStr),
				successStyle.Render("← VENCEDOR"),
			))
		} else {
			sb.WriteString(fmt.Sprintf("  %s %s %s\n",
				lipgloss.NewStyle().Foreground(softWhite).Render(nomesLetras[j]),
				dimStyle.Render(bar),
				dimStyle.Render(yStr),
			))
		}
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  Enter → desenhar novamente • Esc → menu"))
	return sb.String()
}

// =============================================================================
// Utilitários
// =============================================================================

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
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
