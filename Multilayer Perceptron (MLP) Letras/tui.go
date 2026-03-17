package main

import (
	"fmt"
	"math"
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
	mouseDown bool // arrastar com botão esquerdo pressionado
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

	case tea.MouseMsg:
		if m.state == stateTest {
			isLeft := msg.Button == tea.MouseButtonLeft
			switch {
			case isLeft && msg.Action == tea.MouseActionPress:
				m.mouseDown = true
				m = m.paintAtMouse(msg.X, msg.Y, true)
			case msg.Action == tea.MouseActionRelease:
				m.mouseDown = false
			case isLeft && msg.Action == tea.MouseActionMotion && m.mouseDown:
				m = m.paintAtMouse(msg.X, msg.Y, false)
			}
		}

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
	var inner string
	switch m.state {
	case stateMenu:
		inner = m.viewMenu()
	case stateTrainingDone:
		inner = m.viewTrainingDone()
	case stateResult:
		inner = m.viewResult()
	case stateTest:
		inner = m.viewTest()
	}
	// Mesmo padrão do Trab 03: container Padding(1,2) em todos os estados.
	// Isso garante que os offsets de mouse sejam previsíveis e consistentes.
	return lipgloss.NewStyle().Padding(1, 2).Render(inner)
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
// renderErroCurve — gráfico ASCII com escala logarítmica no eixo Y.
// Escala log é essencial para curvas de erro de redes neurais, que caem
// exponencialmente — escala linear comprime toda a variação no fundo.
func renderErroCurve(hist []float64, width, height int) string {
	if len(hist) == 0 {
		return ""
	}

	// Amostra 1 ponto por coluna (distribui uniformemente)
	pts := make([]float64, width)
	n := len(hist)
	for col := 0; col < width; col++ {
		srcIdx := col * (n - 1) / (width - 1)
		if srcIdx >= n {
			srcIdx = n - 1
		}
		pts[col] = hist[srcIdx]
	}

	// Converte para log — garante que valores > 0
	logPts := make([]float64, width)
	for i, v := range pts {
		if v < 1e-10 {
			v = 1e-10
		}
		logPts[i] = math.Log(v)
	}

	logMax := logPts[0]
	logMin := logPts[0]
	for _, v := range logPts {
		if v > logMax {
			logMax = v
		}
		if v < logMin {
			logMin = v
		}
	}
	logRange := logMax - logMin
	if logRange < 0.001 {
		logRange = 0.001
	}

	// Monta grid
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}
	for col, lv := range logPts {
		norm := (lv - logMin) / logRange
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
	dotStyle := lipgloss.NewStyle().Foreground(neonMagenta)
	for row := 0; row < height; row++ {
		// label: valor real (não log) nas linhas de topo e fundo
		label := "       "
		if row == 0 {
			label = fmt.Sprintf("%7.2f", pts[0]) // valor inicial (maior)
		} else if row == height-1 {
			label = fmt.Sprintf("%7.3f", pts[width-1]) // valor final (menor)
		}
		sb.WriteString(dimStyle.Render(label))
		sb.WriteString(dimStyle.Render("│"))
		for _, ch := range grid[row] {
			if ch == '•' {
				sb.WriteString(dotStyle.Render("•"))
			} else {
				sb.WriteString(" ")
			}
		}
		sb.WriteString("\n")
	}
	sb.WriteString(dimStyle.Render("       └" + strings.Repeat("─", width) + "\n"))
	padding := width - 12
	if padding < 0 {
		padding = 0
	}
	sb.WriteString(dimStyle.Render(fmt.Sprintf("        1%s%d ciclos  (escala log)\n",
		strings.Repeat(" ", padding), n)))
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

// paintAtMouse — mapeia coordenadas do mouse para a célula da grade.
// O layout do viewTest coloca a grade em:
//   col = 2 + j*2   (2 chars de indent, depois cada célula ocupa 2 chars)
//   row = gradeStartRow + i
// gradeStartRow é calculado a partir do número de linhas antes da grade:
//   linha 0: \n inicial
//   linha 1: header
//   linha 2: \n após header
//   linha 3: label "Grade (cursor destacado):"
//   linha 4+: pixels da grade
// toggle=true alterna o pixel; toggle=false (drag) só acende.
func (m model) paintAtMouse(mx, my int, toggle bool) model {
	// Padding(1,2) do container + layout interno do viewTest:
	//   row: 1(pad_top) + 1(\n) + 1(header) + 2(\n\n) + 1(label) = 6
	//   col: 2(pad_left) + 2("  " indent) = 4, cada célula ocupa 2 chars
	// Idêntico ao Trab 03 (gradeScreenRow=6, gradeScreenCol=4).
	const gradeStartRow = 5
	const gradeStartCol = 4

	row := my - gradeStartRow
	col := (mx - gradeStartCol) / 2

	if row < 0 || row >= N_LINHAS || col < 0 || col >= N_COLUNAS {
		return m
	}

	// move cursor de teclado para a posição clicada (feedback visual)
	m.testRow = row
	m.testCol = col

	idx := row*N_COLUNAS + col
	if toggle {
		if m.testGrade[idx] > 0 {
			m.testGrade[idx] = -1.0
		} else {
			m.testGrade[idx] = 1.0
		}
	} else {
		m.testGrade[idx] = 1.0
	}
	return m
}

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
	sb.WriteString(hintStyle.Render("  ↑↓←→ mover cursor  ·  espaço/enter toggle  ·  click/drag pintar  ·  r resetar  ·  q voltar") + "\n")
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
	p := tea.NewProgram(initialModel(), tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Erro: %v\n", err)
		os.Exit(1)
	}
}
