package main

import (
	"fmt"
	"os"
	"strings"

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
	neonOrange  = lipgloss.Color("#FF8C00")
	neonRed     = lipgloss.Color("#FF3030")
	softWhite   = lipgloss.Color("#E0E0E0")
	dimGray     = lipgloss.Color("#555555")
	midGray     = lipgloss.Color("#888888")
	darkBg      = lipgloss.Color("#1A1A2E")
)

// =============================================================================
// Estilos Lipgloss
// =============================================================================

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(darkBg).
			Background(neonCyan).
			Padding(0, 2)

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

	thinBox = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(neonCyan).
		Padding(0, 1)

	slideBox = lipgloss.NewStyle().
			Border(lipgloss.ThickBorder()).
			BorderForeground(neonMagenta).
			Padding(1, 3)

	successStyle = lipgloss.NewStyle().Foreground(neonGreen).Bold(true)
	errorStyle   = lipgloss.NewStyle().Foreground(neonRed).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(neonYellow).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(neonCyan)
	dimStyle     = lipgloss.NewStyle().Foreground(dimGray)
	midStyle     = lipgloss.NewStyle().Foreground(midGray)
	hintStyle    = lipgloss.NewStyle().Foreground(dimGray).Italic(true)
	weightStyle  = lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	labelStyle   = lipgloss.NewStyle().Foreground(neonPink)
	boldWhite    = lipgloss.NewStyle().Foreground(softWhite).Bold(true)
)

// suppress unused variable warnings for styles only in palette
var (
	_ = weightStyle
	_ = midStyle
	_ = neonOrange
	_ = boxStyle
	_ = slideBox
	_ = thinBox
)

// =============================================================================
// Estados do TUI
// =============================================================================

type sessionState int

const (
	stateMenu         sessionState = iota
	stateTrainingDone              // resumo + curva de erro ASCII
	stateResult                    // testa as 26 letras
	stateTest                      // grade interativa 5×7 clicável
)

// =============================================================================
// Model principal
// =============================================================================

type model struct {
	state   sessionState
	cursor  int
	choices []string
	spinner spinner.Model

	winW int
	winH int

	resultado *ResultadoTreino

	// stateResult — navegação entre letras
	resultLetraIdx int

	// stateTest — grade interativa 5×7
	testGrade [N_IN]float64
	testRow   int
	testCol   int
}

func initialModel() model {
	s := spinner.New()
	s.Spinner = spinner.Jump
	s.Style = lipgloss.NewStyle().Foreground(neonCyan)

	choices := []string{
		"Treinar MLP Letras    — aprende A–Z com backprop",
		"Ver resultado         — testa as 26 letras",
		"Testar letra          — inserir padrão manual (grade 5×7 clicável)",
		"Sair",
	}

	var testGrade [N_IN]float64
	for i := range testGrade {
		testGrade[i] = -1.0
	}

	return model{
		state:     stateMenu,
		choices:   choices,
		spinner:   s,
		winW:      120,
		winH:      40,
		testGrade: testGrade,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Update
// =============================================================================

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.winW = msg.Width
		m.winH = msg.Height

	case tea.KeyMsg:
		switch m.state {
		case stateMenu:
			switch msg.String() {
			case "ctrl+c", "q":
				return m, tea.Quit
			case "up", "k":
				if m.cursor > 0 {
					m.cursor--
				}
			case "down", "j":
				if m.cursor < len(m.choices)-1 {
					m.cursor++
				}
			case "enter", " ":
				return m.handleMenuEnter()
			}

		case stateTrainingDone:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "enter", " ":
				m.state = stateMenu
				return m, nil
			}

		case stateResult:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "q":
				m.state = stateMenu
				return m, nil
			case "left", "h":
				if m.resultLetraIdx > 0 {
					m.resultLetraIdx--
				}
			case "right", "l":
				if m.resultLetraIdx < N_OUT-1 {
					m.resultLetraIdx++
				}
			case "enter", " ":
				m.state = stateMenu
				return m, nil
			}

		case stateTest:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "q":
				m.state = stateMenu
				return m, nil
			case "up":
				if m.testRow > 0 {
					m.testRow--
				}
			case "down":
				if m.testRow < N_LINHAS-1 {
					m.testRow++
				}
			case "left":
				if m.testCol > 0 {
					m.testCol--
				}
			case "right":
				if m.testCol < N_COLUNAS-1 {
					m.testCol++
				}
			case " ", "enter":
				cellIdx := m.testRow*N_COLUNAS + m.testCol
				if m.testGrade[cellIdx] > 0 {
					m.testGrade[cellIdx] = -1.0
				} else {
					m.testGrade[cellIdx] = 1.0
				}
			case "r":
				for i := range m.testGrade {
					m.testGrade[i] = -1.0
				}
			}
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
	case 0: // Treinar MLP
		res := treinarMLP()
		m.resultado = &res
		m.state = stateTrainingDone
		return m, nil

	case 1: // Ver resultado
		if m.resultado == nil {
			res := treinarMLP()
			m.resultado = &res
		}
		m.resultLetraIdx = 0
		m.state = stateResult
		return m, nil

	case 2: // Testar letra
		if m.resultado == nil {
			res := treinarMLP()
			m.resultado = &res
		}
		for i := range m.testGrade {
			m.testGrade[i] = -1.0
		}
		m.testRow = 0
		m.testCol = 0
		m.state = stateTest
		return m, nil

	case 3: // Sair
		return m, tea.Quit
	}
	return m, nil
}

// =============================================================================
// View
// =============================================================================

func (m model) View() string {
	switch m.state {
	case stateMenu:
		return m.viewMenu()
	case stateTrainingDone:
		return m.viewTrainingDone()
	case stateResult:
		return m.viewResult()
	case stateTest:
		return m.viewTest()
	}
	return ""
}

// =============================================================================
// Menu
// =============================================================================

func (m model) viewMenu() string {
	var sb strings.Builder

	title := titleStyle.Render(" MLP Letras — Reconhecimento A–Z ")
	sub := subtitleStyle.Render("Backpropagation · 35 entradas · 15 ocultos · 26 saídas")
	sb.WriteString("\n  " + title + "\n")
	sb.WriteString("  " + sub + "\n\n")

	if m.resultado != nil {
		var trained string
		if m.resultado.convergiu {
			trained = successStyle.Render(fmt.Sprintf("  [treinado: %d ciclos, erro=%.4f]", m.resultado.ciclos, m.resultado.erroFinal))
		} else {
			trained = warnStyle.Render(fmt.Sprintf("  [treinado: %d ciclos, não convergiu, erro=%.4f]", m.resultado.ciclos, m.resultado.erroFinal))
		}
		sb.WriteString(trained + "\n\n")
	}

	for i, choice := range m.choices {
		if i == m.cursor {
			sb.WriteString(selectedItemStyle.Render("▸ "+choice) + "\n")
		} else {
			sb.WriteString(menuItemStyle.Render("  "+choice) + "\n")
		}
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  ↑↓ navegar  ·  enter selecionar  ·  q sair") + "\n")
	return sb.String()
}

// =============================================================================
// Treinamento concluído — resumo + curva de erro
// =============================================================================

func (m model) viewTrainingDone() string {
	if m.resultado == nil {
		return ""
	}
	res := m.resultado
	var sb strings.Builder

	header := titleStyle.Render(" Treinamento Concluído ")
	sb.WriteString("\n  " + header + "\n\n")

	if res.convergiu {
		sb.WriteString(successStyle.Render(fmt.Sprintf("  ✓ Convergiu em %d ciclos!", res.ciclos)) + "\n")
	} else {
		sb.WriteString(warnStyle.Render(fmt.Sprintf("  ⚠ Não convergiu em %d ciclos", res.ciclos)) + "\n")
	}
	sb.WriteString(infoStyle.Render(fmt.Sprintf("  Erro final: %.6f  (alvo: %.3f)", res.erroFinal, ERRO_ALVO)) + "\n\n")

	// Acurácia nos dados de treino
	dataset := letrasDataset()
	acertos := 0
	for i := 0; i < N_OUT; i++ {
		pred := classificar(res.rede, dataset[i])
		if pred == i {
			acertos++
		}
	}
	pct := float64(acertos) / float64(N_OUT) * 100.0
	sb.WriteString(boldWhite.Render(fmt.Sprintf("  Acurácia nos dados de treino: %d/%d (%.0f%%)", acertos, N_OUT, pct)) + "\n\n")

	// Curva de erro ASCII
	sb.WriteString(labelStyle.Render("  Curva de Erro por Ciclo:") + "\n")
	sb.WriteString(renderErroCurve(res.erroHistorico, 60, 8))

	sb.WriteString("\n\n")
	sb.WriteString(hintStyle.Render("  enter/esc — voltar ao menu") + "\n")
	return sb.String()
}

// renderErroCurve — gráfico ASCII da curva de erro
func renderErroCurve(hist []float64, width, height int) string {
	if len(hist) == 0 {
		return ""
	}

	maxVal := hist[0]
	minVal := hist[len(hist)-1]
	if minVal < 0 {
		minVal = 0
	}

	sampStep := 1
	if len(hist) > width {
		sampStep = len(hist) / width
	}
	var pts []float64
	for i := 0; i < len(hist); i += sampStep {
		pts = append(pts, hist[i])
	}
	if len(pts) > width {
		pts = pts[:width]
	}

	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, len(pts))
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	for col, val := range pts {
		norm := 0.0
		if maxVal > minVal {
			norm = (val - minVal) / (maxVal - minVal)
		}
		row := height - 1 - int(norm*float64(height-1))
		if row < 0 {
			row = 0
		}
		if row >= height {
			row = height - 1
		}
		grid[row][col] = '•'
	}

	var sb strings.Builder
	for row := 0; row < height; row++ {
		label := "    "
		if row == 0 {
			label = fmt.Sprintf("%5.1f", maxVal)
		} else if row == height-1 {
			label = fmt.Sprintf("%5.1f", minVal)
		}
		sb.WriteString(dimStyle.Render(label))
		sb.WriteString(dimStyle.Render("│"))
		for _, ch := range grid[row] {
			if ch == '•' {
				sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Render("•"))
			} else {
				sb.WriteString(" ")
			}
		}
		sb.WriteString("\n")
	}
	padding := len(pts) - 10
	if padding < 0 {
		padding = 0
	}
	sb.WriteString(dimStyle.Render("      └" + strings.Repeat("─", len(pts)) + "\n"))
	sb.WriteString(dimStyle.Render(fmt.Sprintf("       1%s%d ciclos\n",
		strings.Repeat(" ", padding), len(hist))))
	return sb.String()
}

// =============================================================================
// Resultado — testa as 26 letras, ←→ para navegar
// =============================================================================

func (m model) viewResult() string {
	if m.resultado == nil {
		return ""
	}
	var sb strings.Builder

	header := titleStyle.Render(" Resultado — Teste das 26 Letras ")
	sb.WriteString("\n  " + header + "\n\n")

	dataset := letrasDataset()
	acertos := 0
	for i := 0; i < N_OUT; i++ {
		pred := classificar(m.resultado.rede, dataset[i])
		if pred == i {
			acertos++
		}
	}
	pct := float64(acertos) / float64(N_OUT) * 100.0
	sb.WriteString(boldWhite.Render(fmt.Sprintf("  Acurácia: %d/%d (%.0f%%)", acertos, N_OUT, pct)) + "\n\n")

	// Letra atual
	idx := m.resultLetraIdx
	x := dataset[idx]
	pred := classificar(m.resultado.rede, x)
	fwd := forward(m.resultado.rede, x)
	conf := fwd.y[pred]

	if pred == idx {
		sb.WriteString(successStyle.Render(fmt.Sprintf("  Letra: %s  ✓  Classificado como: %s", nomesLetras[idx], nomesLetras[pred])) + "\n")
	} else {
		sb.WriteString(errorStyle.Render(fmt.Sprintf("  Letra: %s  ✗  Classificado como: %s", nomesLetras[idx], nomesLetras[pred])) + "\n")
	}
	sb.WriteString(infoStyle.Render(fmt.Sprintf("  Confiança: %.4f", conf)) + "\n\n")

	// Grade da letra
	grid := formataLetraGridWithCursor(x, -1, -1)
	for _, line := range strings.Split(strings.TrimRight(grid, "\n"), "\n") {
		sb.WriteString("  " + line + "\n")
	}
	sb.WriteString("\n")

	// Mini-lista de todas as letras com ✓/✗
	sb.WriteString(dimStyle.Render("  Todas as letras:") + "\n")
	for i := 0; i < N_OUT; i++ {
		p := classificar(m.resultado.rede, dataset[i])
		marker := successStyle.Render("✓")
		if p != i {
			marker = errorStyle.Render("✗")
		}
		lStyle := dimStyle
		if i == idx {
			lStyle = boldWhite
		}
		sb.WriteString("  " + marker + " " + lStyle.Render(nomesLetras[i]) + "  ")
		if (i+1)%13 == 0 {
			sb.WriteString("\n")
		}
	}
	sb.WriteString("\n\n")

	// Navegação
	nav := fmt.Sprintf("  [%d/%d]  ", idx+1, N_OUT)
	if idx > 0 {
		nav += infoStyle.Render("← prev  ")
	} else {
		nav += dimStyle.Render("        ")
	}
	if idx < N_OUT-1 {
		nav += infoStyle.Render("→ next")
	}
	sb.WriteString(nav + "\n")
	sb.WriteString(hintStyle.Render("  ←→ navegar letras  ·  esc/q voltar ao menu") + "\n")
	return sb.String()
}

// =============================================================================
// Teste interativo — grade 5×7 editável com cursor
// =============================================================================

func (m model) viewTest() string {
	var sb strings.Builder

	header := titleStyle.Render(" Testar Letra — Grade 5×7 Interativa ")
	sb.WriteString("\n  " + header + "\n\n")

	// Grade com cursor
	grid := formataLetraGridWithCursor(m.testGrade, m.testRow, m.testCol)
	sb.WriteString("  " + labelStyle.Render("Grade (cursor destacado):") + "\n")
	for _, line := range strings.Split(strings.TrimRight(grid, "\n"), "\n") {
		sb.WriteString("  " + line + "\n")
	}
	sb.WriteString("\n")

	// Classificação em tempo real
	if m.resultado != nil {
		pred := classificar(m.resultado.rede, m.testGrade)
		fwd := forward(m.resultado.rede, m.testGrade)
		conf := fwd.y[pred]

		sb.WriteString(boldWhite.Render(fmt.Sprintf("  Classificação: %s", nomesLetras[pred])) + "\n")
		sb.WriteString(infoStyle.Render(fmt.Sprintf("  Confiança: %.4f", conf)) + "\n\n")

		// Top-5 candidatos usando cópia dos scores
		scores := fwd.y // array copy (value semantics)
		sb.WriteString(dimStyle.Render("  Top-5 candidatos:") + "\n")
		for shown := 0; shown < 5; shown++ {
			best := 0
			for i := 1; i < N_OUT; i++ {
				if scores[i] > scores[best] {
					best = i
				}
			}
			bar := renderConfBar(scores[best], 20)
			sb.WriteString(fmt.Sprintf("    %s  %s  %.4f\n",
				boldWhite.Render(nomesLetras[best]),
				bar,
				scores[best],
			))
			scores[best] = -999.0
		}
	} else {
		sb.WriteString(warnStyle.Render("  Nenhuma rede treinada. Volte ao menu e treine primeiro.") + "\n")
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  ↑↓←→ mover cursor  ·  espaço/enter toggle  ·  r resetar  ·  q voltar") + "\n")
	return sb.String()
}

// formataLetraGridWithCursor — grade 5×7 com cursor opcionalmente destacado
// Se curRow < 0, não mostra cursor
func formataLetraGridWithCursor(x [N_IN]float64, curRow, curCol int) string {
	pixelAtivo := lipgloss.NewStyle().Foreground(lipgloss.Color("#FF6EC7")).Render("█")
	pixelInativo := lipgloss.NewStyle().Foreground(lipgloss.Color("#555555")).Render("·")
	cursorAtivo := lipgloss.NewStyle().Foreground(lipgloss.Color("#1A1A2E")).Background(lipgloss.Color("#FF6EC7")).Render("█")
	cursorInativo := lipgloss.NewStyle().Foreground(lipgloss.Color("#1A1A2E")).Background(lipgloss.Color("#888888")).Render("·")

	var sb strings.Builder
	for i := 0; i < N_LINHAS; i++ {
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			isCursor := (curRow >= 0 && i == curRow && j == curCol)
			isOn := x[i*N_COLUNAS+j] > 0
			if isCursor {
				if isOn {
					sb.WriteString(cursorAtivo)
				} else {
					sb.WriteString(cursorInativo)
				}
			} else {
				if isOn {
					sb.WriteString(pixelAtivo)
				} else {
					sb.WriteString(pixelInativo)
				}
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// renderConfBar — barra de confiança proporcional ao valor (range [-1,+1])
func renderConfBar(val float64, width int) string {
	norm := (val + 1.0) / 2.0
	if norm < 0 {
		norm = 0
	}
	if norm > 1 {
		norm = 1
	}
	filled := int(norm * float64(width))
	empty := width - filled

	return lipgloss.NewStyle().Foreground(neonGreen).Render(strings.Repeat("█", filled)) +
		dimStyle.Render(strings.Repeat("░", empty))
}

// =============================================================================
// runTUI — ponto de entrada do TUI
// =============================================================================

func runTUI() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Erro: %v\n", err)
		os.Exit(1)
	}
}
