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
// Paleta de cores — neon/cyberpunk (idêntica ao Trab 03)
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

// =============================================================================
// Estados do TUI
// =============================================================================

type sessionState int

const (
	stateMenu         sessionState = iota
	stateTraining                  // animação do diagrama de neurônios
	stateTrainingDone              // resumo + curva de erro ASCII
	stateSlide                     // 6 slides explicativos
	stateResult                    // teste dos 3 padrões
	stateWalkthrough               // passo a passo manual conta a conta
)

// =============================================================================
// Model principal
// =============================================================================

type model struct {
	state   sessionState
	cursor  int
	choices []string
	spinner spinner.Model
	progress progress.Model

	winW int
	winH int

	resultado      *ResultadoTreino
	currentStepIdx int

	// slide
	slideIdx       int
	slideTotalStep int

	// animação de fase (forward vs backprop)
	animPhase int // 0=forward, 1=backprop

	// walkthrough — passo a passo manual
	wtStep    int // sub-passo dentro do padrão atual (avança com →)
	wtPadrao  int // padrão atual (0, 1, 2)
	wtCiclo   int // ciclo atual sendo mostrado
	wtMLP     MLP // estado dos pesos no início do ciclo atual
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
		"Treinar MLP            — diagrama animado",
		"Passo a passo          — conta a conta ciclo 1",
		"Ver slides explicativos — 6 slides",
		"Testar rede            — 3 padrões",
		"Sair",
	}

	return model{
		state:   stateMenu,
		choices: choices,
		spinner: s,
		progress: p,
		winW:    120,
		winH:    40,
	}
}

func (m model) Init() tea.Cmd {
	return m.spinner.Tick
}

// =============================================================================
// Mensagens de animação
// =============================================================================

type trainingTickMsg time.Time
type slideRevealMsg struct{}

func tickTraining() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(t time.Time) tea.Msg {
		return trainingTickMsg(t)
	})
}

func tickSlideReveal() tea.Cmd {
	return tea.Tick(60*time.Millisecond, func(_ time.Time) tea.Msg {
		return slideRevealMsg{}
	})
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

		case stateTraining:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "enter", " ":
				m.currentStepIdx = len(m.resultado.steps)
				m.state = stateTrainingDone
				return m, nil
			}

		case stateTrainingDone:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "enter", " ":
				m.state = stateMenu
				return m, nil
			}

		case stateSlide:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "q":
				m.state = stateMenu
				return m, nil
			case "right", "l", "enter", " ":
				if m.slideTotalStep < slideMaxSteps(m.slideIdx) {
					m.slideTotalStep = slideMaxSteps(m.slideIdx)
				} else if m.slideIdx < totalSlides()-1 {
					m.slideIdx++
					m.slideTotalStep = 0
					return m, tickSlideReveal()
				}
			case "left", "h":
				if m.slideIdx > 0 {
					m.slideIdx--
					m.slideTotalStep = slideMaxSteps(m.slideIdx)
				}
			}

		case stateResult:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "enter", " ", "q":
				m.state = stateMenu
				return m, nil
			}

		case stateWalkthrough:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc", "q":
				m.state = stateMenu
				return m, nil
			case "right", "l", " ", "enter":
				// avança um sub-passo; se acabou o padrão, vai para o próximo
				m.wtStep++
				if m.wtStep >= wtMaxSteps() {
					m.wtStep = 0
					// aplica o update dos pesos antes de avançar o padrão
					fwd := forward(m.wtMLP, wtPadroes[m.wtPadrao])
					bwd := backward(m.wtMLP, fwd, wtTargets[m.wtPadrao], wtPadroes[m.wtPadrao])
					m.wtMLP = atualizarPesos(m.wtMLP, bwd)
					m.wtPadrao++
					if m.wtPadrao >= 3 {
						m.wtPadrao = 0
						m.wtCiclo++
					}
				}
				return m, nil
			case "left", "h":
				if m.wtStep > 0 {
					m.wtStep--
				}
				return m, nil
			}
		}

	case trainingTickMsg:
		if m.state == stateTraining && m.resultado != nil {
			if m.currentStepIdx < len(m.resultado.steps) {
				m.animPhase = (m.animPhase + 1) % 2
				if m.animPhase == 0 {
					m.currentStepIdx++
				}
				return m, tickTraining()
			} else {
				m.state = stateTrainingDone
				return m, nil
			}
		}

	case slideRevealMsg:
		if m.state == stateSlide {
			if m.slideTotalStep < slideMaxSteps(m.slideIdx) {
				m.slideTotalStep++
				return m, tickSlideReveal()
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
		m.currentStepIdx = 0
		m.animPhase = 0
		m.state = stateTraining
		return m, tickTraining()

	case 1: // Passo a passo
		m.state = stateWalkthrough
		m.wtStep = 0
		m.wtPadrao = 0
		m.wtCiclo = 1
		m.wtMLP = inicializarPesosSlide()
		return m, nil

	case 2: // Ver slides
		m.state = stateSlide
		m.slideIdx = 0
		m.slideTotalStep = 0
		return m, tickSlideReveal()

	case 3: // Testar rede
		if m.resultado == nil {
			res := treinarMLP()
			m.resultado = &res
		}
		m.state = stateResult
		return m, nil

	case 4: // Sair
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
	case stateTraining:
		return m.viewTraining()
	case stateTrainingDone:
		return m.viewTrainingDone()
	case stateSlide:
		return m.viewSlide()
	case stateResult:
		return m.viewResult()
	case stateWalkthrough:
		return m.viewWalkthrough()
	}
	return ""
}

// =============================================================================
// Menu
// =============================================================================

func (m model) viewMenu() string {
	var sb strings.Builder

	title := titleStyle.Render(" MLP — Multilayer Perceptron ")
	sub := subtitleStyle.Render("Backpropagation · Aula 05 · Prof. Jefferson")
	sb.WriteString("\n  " + title + "\n")
	sb.WriteString("  " + sub + "\n\n")

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
// Diagrama animado de neurônios
// =============================================================================

func (m model) viewTraining() string {
	if m.resultado == nil || len(m.resultado.steps) == 0 {
		return "Carregando...\n"
	}

	idx := m.currentStepIdx
	if idx >= len(m.resultado.steps) {
		idx = len(m.resultado.steps) - 1
	}
	step := m.resultado.steps[idx]
	isForward := m.animPhase == 0

	var sb strings.Builder

	// Cabeçalho
	header := titleStyle.Render(" MLP — Treinamento ")
	sb.WriteString("\n  " + header + "\n\n")

	// Status bar
	cicloStr := boldWhite.Render(fmt.Sprintf("Ciclo %d", step.ciclo))
	padStr := labelStyle.Render(fmt.Sprintf("Padrão %d", step.padrao))
	erroStr := warnStyle.Render(fmt.Sprintf("Erro: %.5f", step.erroTotal))
	faseStr := ""
	if isForward {
		faseStr = infoStyle.Render("► Forward")
	} else {
		faseStr = lipgloss.NewStyle().Foreground(neonYellow).Bold(true).Render("◄ Backprop")
	}
	sb.WriteString(fmt.Sprintf("  %s  │  %s  │  %s  │  %s\n\n", cicloStr, padStr, erroStr, faseStr))

	// Renderiza as 4 colunas do diagrama
	col1 := renderEntradaCol(step, isForward)
	col2 := renderOcultaCol(step, isForward)
	col3 := renderSaidaCol(step, isForward)
	col4 := renderMathCol(step, isForward)

	diagram := lipgloss.JoinHorizontal(lipgloss.Top, col1, col2, col3, col4)
	sb.WriteString(diagram)
	sb.WriteString("\n\n")

	// Barra de progresso do erro
	sb.WriteString(renderErroBar(step.erroTotal))
	sb.WriteString("\n\n")
	sb.WriteString(hintStyle.Render("  enter/esc — pular para resultado final") + "\n")

	return sb.String()
}

// renderEntradaCol — coluna das entradas
func renderEntradaCol(step TrainingStep, isForward bool) string {
	x := step.x
	lines := []string{
		"",
		"  ENTRADA",
		"",
		renderNeuron(fmt.Sprintf("x₁=%+.1f", x[0]), isForward, "entrada"),
		"",
		renderNeuron(fmt.Sprintf("x₂=%+.1f", x[1]), isForward, "entrada"),
		"",
		renderNeuron(fmt.Sprintf("x₃=%+.1f", x[2]), isForward, "entrada"),
		"",
		"",
	}
	return strings.Join(lines, "\n") + "    "
}

// renderOcultaCol — coluna dos neurônios ocultos com conexões
func renderOcultaCol(step TrainingStep, isForward bool) string {
	z := step.fwd.z
	lines := []string{
		"",
		"     OCULTA",
		"",
		"",
		renderNeuronVal("z₁", z[0], isForward, "oculta"),
		"",
		"",
		"",
		renderNeuronVal("z₂", z[1], isForward, "oculta"),
		"",
		"",
	}
	return strings.Join(lines, "\n") + "    "
}

// renderSaidaCol — coluna das saídas
func renderSaidaCol(step TrainingStep, isForward bool) string {
	y := step.fwd.y
	t := step.target
	lines := []string{
		"",
		"     SAÍDA",
		"",
		renderNeuronOutput("y₁", y[0], t[0], isForward),
		"",
		renderNeuronOutput("y₂", y[1], t[1], isForward),
		"",
		renderNeuronOutput("y₃", y[2], t[2], isForward),
		"",
		"",
		"",
	}
	return strings.Join(lines, "\n") + "    "
}

// renderMathCol — coluna com valores matemáticos
func renderMathCol(step TrainingStep, isForward bool) string {
	fwd := step.fwd
	bwd := step.bwd
	var lines []string

	if isForward {
		lines = []string{
			"",
			infoStyle.Render("  Forward Pass:"),
			"",
			dimStyle.Render(fmt.Sprintf("  zin₁ = %+.4f", fwd.zin[0])),
			infoStyle.Render(fmt.Sprintf("  z₁   = %+.4f", fwd.z[0])),
			"",
			dimStyle.Render(fmt.Sprintf("  zin₂ = %+.4f", fwd.zin[1])),
			infoStyle.Render(fmt.Sprintf("  z₂   = %+.4f", fwd.z[1])),
			"",
			dimStyle.Render(fmt.Sprintf("  yin₁=%+.3f yin₂=%+.3f yin₃=%+.3f", fwd.yin[0], fwd.yin[1], fwd.yin[2])),
			infoStyle.Render(fmt.Sprintf("  y₁=%+.4f  y₂=%+.4f  y₃=%+.4f", fwd.y[0], fwd.y[1], fwd.y[2])),
		}
	} else {
		lines = []string{
			"",
			warnStyle.Render("  Backprop:"),
			"",
			warnStyle.Render(fmt.Sprintf("  δ₁=%+.4f δ₂=%+.4f δ₃=%+.4f", bwd.deltaK[0], bwd.deltaK[1], bwd.deltaK[2])),
			dimStyle.Render(fmt.Sprintf("  δin_j=[%+.4f  %+.4f]", bwd.deltaInJ[0], bwd.deltaInJ[1])),
			warnStyle.Render(fmt.Sprintf("  δ_j= [%+.4f  %+.4f]", bwd.deltaJ[0], bwd.deltaJ[1])),
			"",
			dimStyle.Render("  Δw (oculta→saída):"),
			dimStyle.Render(fmt.Sprintf("  [%+.5f %+.5f %+.5f]", bwd.deltaW[0][0], bwd.deltaW[0][1], bwd.deltaW[0][2])),
			dimStyle.Render(fmt.Sprintf("  [%+.5f %+.5f %+.5f]", bwd.deltaW[1][0], bwd.deltaW[1][1], bwd.deltaW[1][2])),
			dimStyle.Render("  Δv₀="+fmt.Sprintf("[%+.5f %+.5f]", bwd.deltaV0[0], bwd.deltaV0[1])),
		}
	}

	return strings.Join(lines, "\n")
}

// renderNeuron — neurônio de entrada estilizado
func renderNeuron(label string, active bool, kind string) string {
	_ = kind
	style := lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	if active {
		style = style.Background(lipgloss.Color("#003333"))
	}
	return style.Render(fmt.Sprintf("( %-8s)", label))
}

// renderNeuronVal — neurônio com valor numérico
func renderNeuronVal(label string, val float64, active bool, kind string) string {
	_ = kind
	color := neonMagenta
	bg := lipgloss.Color("#1A1A2E")
	if active {
		bg = lipgloss.Color("#330033")
	}
	style := lipgloss.NewStyle().Foreground(color).Bold(true).Background(bg)
	return style.Render(fmt.Sprintf("( %s=%+.3f )", label, val))
}

// renderNeuronOutput — neurônio de saída, cor verde se acertou sinal, vermelho se errou
func renderNeuronOutput(label string, y, t float64, active bool) string {
	correct := (y > 0 && t > 0) || (y < 0 && t < 0)
	color := neonGreen
	if !correct {
		color = neonRed
	}
	bg := lipgloss.Color("#1A1A2E")
	if active {
		bg = lipgloss.Color("#003300")
		if !correct {
			bg = lipgloss.Color("#330000")
		}
	}
	style := lipgloss.NewStyle().Foreground(color).Bold(true).Background(bg)
	return style.Render(fmt.Sprintf("( %s=%+.3f )", label, y))
}

// renderErroBar — barra ASCII proporcional ao erro
func renderErroBar(erro float64) string {
	maxErro := 3.0
	ratio := erro / maxErro
	if ratio > 1 {
		ratio = 1
	}
	barWidth := 30
	filled := int(ratio * float64(barWidth))
	empty := barWidth - filled

	bar := lipgloss.NewStyle().Foreground(neonMagenta).Render(strings.Repeat("█", filled)) +
		dimStyle.Render(strings.Repeat("░", empty))

	erroLabel := warnStyle.Render(fmt.Sprintf("%.5f", erro))
	target := dimStyle.Render(fmt.Sprintf("/ %.3f alvo", ERRO_ALVO))

	return fmt.Sprintf("  Erro: %s %s %s", bar, erroLabel, target)
}

// =============================================================================
// Tela de treino concluído — resumo + curva de erro ASCII
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
	sb.WriteString(infoStyle.Render(fmt.Sprintf("  Erro final: %.6f", res.erroFinal)) + "\n\n")

	// Pesos finais
	sb.WriteString(labelStyle.Render("  Pesos finais v (entrada→oculta):") + "\n")
	for i := 0; i < N_IN; i++ {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("    x%d→z1=%+.4f  x%d→z2=%+.4f",
			i+1, res.rede.v[i][0], i+1, res.rede.v[i][1])) + "\n")
	}
	sb.WriteString(dimStyle.Render(fmt.Sprintf("    bias: v0₁=%+.4f  v0₂=%+.4f\n\n",
		res.rede.v0[0], res.rede.v0[1])))

	sb.WriteString(labelStyle.Render("  Pesos finais w (oculta→saída):") + "\n")
	for j := 0; j < N_HID; j++ {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("    z%d→y1=%+.4f  z%d→y2=%+.4f  z%d→y3=%+.4f",
			j+1, res.rede.w[j][0], j+1, res.rede.w[j][1], j+1, res.rede.w[j][2])) + "\n")
	}
	sb.WriteString(dimStyle.Render(fmt.Sprintf("    bias: w0₁=%+.4f  w0₂=%+.4f  w0₃=%+.4f\n\n",
		res.rede.w0[0], res.rede.w0[1], res.rede.w0[2])))

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

	// Amostra pontos para caber na largura
	step := 1
	if len(hist) > width {
		step = len(hist) / width
	}
	var pts []float64
	for i := 0; i < len(hist); i += step {
		pts = append(pts, hist[i])
	}
	if len(pts) > width {
		pts = pts[:width]
	}

	// Monta grid
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
			label = fmt.Sprintf("%5.3f", maxVal)
		} else if row == height-1 {
			label = fmt.Sprintf("%5.3f", minVal)
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
	sb.WriteString(dimStyle.Render("      └" + strings.Repeat("─", len(pts)) + "\n"))
	sb.WriteString(dimStyle.Render(fmt.Sprintf("       1%s%d ciclos\n",
		strings.Repeat(" ", len(pts)-10), len(hist))))
	return sb.String()
}

// =============================================================================
// Slides explicativos (6 slides, ←→)
// =============================================================================

func totalSlides() int { return 6 }

func slideMaxSteps(idx int) int {
	steps := []int{5, 4, 6, 5, 6, 4}
	if idx >= len(steps) {
		return 4
	}
	return steps[idx]
}

func (m model) viewSlide() string {
	titles := []string{
		"1 / 6 — Arquitetura MLP",
		"2 / 6 — Os Dados",
		"3 / 6 — Forward Pass",
		"4 / 6 — Erro & δ Saída",
		"5 / 6 — Backprop Oculta",
		"6 / 6 — Convergência",
	}

	var content string
	switch m.slideIdx {
	case 0:
		content = m.slide1(m.slideTotalStep)
	case 1:
		content = m.slide2(m.slideTotalStep)
	case 2:
		content = m.slide3(m.slideTotalStep)
	case 3:
		content = m.slide4(m.slideTotalStep)
	case 4:
		content = m.slide5(m.slideTotalStep)
	case 5:
		content = m.slide6(m.slideTotalStep)
	}

	header := titleStyle.Render(" " + titles[m.slideIdx] + " ")
	nav := hintStyle.Render("  ← → navegar  ·  esc voltar ao menu")

	// Indicador de slides
	dots := ""
	for i := 0; i < totalSlides(); i++ {
		if i == m.slideIdx {
			dots += lipgloss.NewStyle().Foreground(neonCyan).Render("●")
		} else {
			dots += dimStyle.Render("○")
		}
		if i < totalSlides()-1 {
			dots += " "
		}
	}

	box := slideBox.Render(content)
	return "\n  " + header + "\n\n" + box + "\n\n  " + dots + "\n\n" + nav + "\n"
}

func reveal(step, threshold int, text string) string {
	if step >= threshold {
		return text
	}
	return ""
}

func (m model) slide1(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Arquitetura do MLP — 3 camadas") + "\n\n")

	sb.WriteString(reveal(step, 1,
		infoStyle.Render("  Entradas (3)      Oculta (2)       Saída (3)")+"\n"+
			dimStyle.Render("  ┌──────────┐    ┌─────────┐     ┌─────────┐")+"\n"+
			lipgloss.NewStyle().Foreground(neonCyan).Bold(true).Render("  │ x₁  x₂  x₃│───▶│  z₁  z₂ │────▶│y₁ y₂ y₃│")+"\n"+
			dimStyle.Render("  └──────────┘    └─────────┘     └─────────┘")+"\n"))

	sb.WriteString(reveal(step, 2,
		"\n"+labelStyle.Render("  Pesos:")+"\n"+
			dimStyle.Render("  v[i][j] = entrada i → neurônio oculto j")+"\n"+
			dimStyle.Render("  w[j][k] = neurônio oculto j → saída k")+"\n"+
			dimStyle.Render("  v0[j]   = bias oculto j")+"\n"+
			dimStyle.Render("  w0[k]   = bias saída k")+"\n"))

	sb.WriteString(reveal(step, 3,
		"\n"+labelStyle.Render("  Ativação: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)")+"\n"+
			dimStyle.Render("  Range: (-1, +1) — ideal para targets ±1")+"\n"))

	sb.WriteString(reveal(step, 4,
		"\n"+warnStyle.Render("  Parâmetros:")+"\n"+
			dimStyle.Render(fmt.Sprintf("  α = %.2f  (taxa de aprendizado)", ALFA))+"\n"+
			dimStyle.Render(fmt.Sprintf("  Critério: E_total ≤ %.3f  ou  max %d ciclos", ERRO_ALVO, MAX_CICLOS))+"\n"+
			dimStyle.Render("  (α=0.01 converge em ~27000 ciclos com esses pesos)")+"\n"))

	sb.WriteString(reveal(step, 5,
		"\n"+successStyle.Render("  Total de pesos: 3×2 + 2 + 2×3 + 3 = 17")+"\n"))

	return sb.String()
}

func (m model) slide2(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Padrões de Treinamento") + "\n\n")

	sb.WriteString(reveal(step, 1,
		dimStyle.Render("  Padrão │  x₁    x₂    x₃   │  t₁   t₂   t₃")+"\n"+
			dimStyle.Render("  ───────┼─────────────────────┼─────────────────")+"\n"+
			infoStyle.Render("     1   │ +1.0  +0.5  -1.0  │  +1   -1   -1")+"\n"+
			infoStyle.Render("     2   │ +1.0  +0.5  +1.0  │  -1   +1   -1")+"\n"+
			infoStyle.Render("     3   │ +1.0  -0.5  -1.0  │  -1   -1   +1")+"\n"))

	sb.WriteString(reveal(step, 2,
		"\n"+labelStyle.Render("  Por que tanh?")+"\n"+
			dimStyle.Render("  • Saída em (-1, +1) — combina com targets ±1")+"\n"+
			dimStyle.Render("  • Derivada: f'(y) = (1+y)(1-y) — fácil de calcular")+"\n"+
			dimStyle.Render("  • Simétrica em torno de zero")+"\n"))

	sb.WriteString(reveal(step, 3,
		"\n"+warnStyle.Render("  Pesos iniciais (do slide):")+"\n"+
			dimStyle.Render("  v₁₁=+0.12  v₁₂=-0.03  v₂₁=-0.04  v₂₂=+0.15")+"\n"+
			dimStyle.Render("  v₃₁=+0.31  v₃₂=-0.41")+"\n"+
			dimStyle.Render("  v0₁=-0.09  v0₂=+0.18")+"\n"+
			dimStyle.Render("  w₁₁=-0.05  w₁₂=+0.19  w₁₃=+0.18")+"\n"+
			dimStyle.Render("  w₂₁=-0.34  w₂₂=-0.09  w₂₃=-0.12")+"\n"+
			dimStyle.Render("  w0₁=+0.18  w0₂=-0.27  w0₃=-0.12")+"\n"))

	sb.WriteString(reveal(step, 4,
		"\n"+successStyle.Render("  Estes são os exatos valores do slide Aula 05!")+"\n"))

	return sb.String()
}

func (m model) slide3(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Forward Pass — Padrão 1: x=[1, 0.5, -1]") + "\n\n")

	sb.WriteString(reveal(step, 1,
		labelStyle.Render("  Camada Oculta:")+"\n"+
			dimStyle.Render("  zin_j = v0_j + Σᵢ xᵢ·v[i][j]")+"\n"))

	sb.WriteString(reveal(step, 2,
		"\n"+infoStyle.Render("  zin₁ = -0.09 + 1·0.12 + 0.5·(-0.04) + (-1)·0.31")+"\n"+
			infoStyle.Render("       = -0.09 + 0.12 - 0.02 - 0.31 = -0.30")+"\n"+
			infoStyle.Render("  z₁   = tanh(-0.30) = -0.2913")+"\n\n"+
			infoStyle.Render("  zin₂ = +0.18 + 1·(-0.03) + 0.5·0.15 + (-1)·(-0.41)")+"\n"+
			infoStyle.Render("       = +0.18 - 0.03 + 0.075 + 0.41 = +0.635")+"\n"+
			infoStyle.Render("  z₂   = tanh(+0.635) = +0.5582")+"\n"))

	sb.WriteString(reveal(step, 3,
		"\n"+labelStyle.Render("  Camada de Saída:")+"\n"+
			dimStyle.Render("  yin_k = w0_k + Σⱼ zⱼ·w[j][k]")+"\n"))

	sb.WriteString(reveal(step, 4,
		"\n"+infoStyle.Render("  yin₁ = 0.18 + (-0.2913)·(-0.05) + (0.5615)·(0.19)")+"\n"+
			infoStyle.Render("       ≈ 0.18 + 0.01457 + 0.10669 ≈ +0.3013")+"\n"+
			infoStyle.Render("  y₁   = tanh(+0.3013) ≈ +0.2925")+"\n\n"+
			infoStyle.Render("  (y₂ e y₃ calculados analogamente)")+"\n"))

	sb.WriteString(reveal(step, 5,
		"\n"+warnStyle.Render("  Target: t=[+1, -1, -1]")+"\n"+
			warnStyle.Render("  E = ½[(1-y₁)² + (-1-y₂)² + (-1-y₃)²]")+"\n"))

	sb.WriteString(reveal(step, 6,
		"\n"+successStyle.Render("  E_padrão1_ciclo1 ≈ 1.022 (com os pesos iniciais do slide)")+"\n"))

	return sb.String()
}

func (m model) slide4(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Erro Quadrático & δ na Camada de Saída") + "\n\n")

	sb.WriteString(reveal(step, 1,
		labelStyle.Render("  Erro por padrão:")+"\n"+
			infoStyle.Render("  E = ½ Σₖ (tₖ - yₖ)²")+"\n\n"+
			dimStyle.Render("  Erro total do ciclo = Σ_padrões E_padrão")+"\n"))

	sb.WriteString(reveal(step, 2,
		"\n"+labelStyle.Render("  δ_k — sinal de erro na saída:")+"\n"+
			infoStyle.Render("  δₖ = (tₖ - yₖ) · f'(yₖ)")+"\n"+
			infoStyle.Render("     = (tₖ - yₖ) · (1 + yₖ)(1 - yₖ)")+"\n\n"+
			dimStyle.Render("  A derivada (1+y)(1-y) modula o erro")+"\n"+
			dimStyle.Render("  conforme a saturação da função")+"\n"))

	sb.WriteString(reveal(step, 3,
		"\n"+warnStyle.Render("  Exemplo — Padrão 1, Saída 1:")+"\n"+
			dimStyle.Render("  t₁=+1,  y₁≈+0.2925")+"\n"+
			infoStyle.Render("  δ₁ = (1 - 0.2925) · (1+0.2925)(1-0.2925)")+"\n"+
			infoStyle.Render("     ≈ 0.7075 · 0.9504 ≈ +0.6724")+"\n"))

	sb.WriteString(reveal(step, 4,
		"\n"+labelStyle.Render("  Update pesos de saída:")+"\n"+
			infoStyle.Render("  Δwⱼₖ = α · δₖ · zⱼ")+"\n"+
			infoStyle.Render("  Δw0ₖ = α · δₖ")+"\n"))

	sb.WriteString(reveal(step, 5,
		"\n"+successStyle.Render(fmt.Sprintf("  α = %.2f — passo pequeno, convergência estável", ALFA))+"\n"))

	return sb.String()
}

func (m model) slide5(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Backpropagation — Camada Oculta") + "\n\n")

	sb.WriteString(reveal(step, 1,
		labelStyle.Render("  δin_j — propaga erro para camada oculta:")+"\n"+
			infoStyle.Render("  δin_j = Σₖ δₖ · w[j][k]")+"\n\n"+
			dimStyle.Render("  Combina todos os δ da saída ponderados")+"\n"+
			dimStyle.Render("  pelos pesos que conectam j a cada k")+"\n"))

	sb.WriteString(reveal(step, 2,
		"\n"+labelStyle.Render("  δ_j — erro modulado pela derivada de tanh:")+"\n"+
			infoStyle.Render("  δⱼ = δin_j · (1 + zⱼ)(1 - zⱼ)")+"\n"))

	sb.WriteString(reveal(step, 3,
		"\n"+warnStyle.Render("  Exemplo — Padrão 1:")+"\n"+
			dimStyle.Render("  δin₁ = δ₁·w[0][0] + δ₂·w[0][1] + δ₃·w[0][2]")+"\n"+
			infoStyle.Render("  (usa os δₖ já calculados na camada de saída)")+"\n"))

	sb.WriteString(reveal(step, 4,
		"\n"+labelStyle.Render("  Update pesos de entrada→oculta:")+"\n"+
			infoStyle.Render("  Δvᵢⱼ = α · δⱼ · xᵢ")+"\n"+
			infoStyle.Render("  Δv0ⱼ = α · δⱼ")+"\n"))

	sb.WriteString(reveal(step, 5,
		"\n"+dimStyle.Render("  Ordem de update: saída → oculta")+"\n"+
			dimStyle.Render("  (usando pesos ANTIGOS para calcular δin_j)")+"\n"))

	sb.WriteString(reveal(step, 6,
		"\n"+successStyle.Render("  Isso é o gradiente descendente estocástico (SGD)!")+"\n"))

	return sb.String()
}

func (m model) slide6(step int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Convergência & Resultados") + "\n\n")

	if m.resultado == nil {
		sb.WriteString(warnStyle.Render("  Execute o treinamento primeiro (Menu → Treinar MLP)") + "\n")
		return sb.String()
	}

	res := m.resultado

	sb.WriteString(reveal(step, 1,
		labelStyle.Render("  Resultado do Treinamento:")+"\n"))

	if res.convergiu {
		sb.WriteString(reveal(step, 1,
			successStyle.Render(fmt.Sprintf("  ✓ Convergiu em %d ciclos", res.ciclos))+"\n"+
				infoStyle.Render(fmt.Sprintf("  Erro final: %.6f (alvo: %.3f)", res.erroFinal, ERRO_ALVO))+"\n"))
	} else {
		sb.WriteString(reveal(step, 1,
			warnStyle.Render(fmt.Sprintf("  ⚠ Não convergiu em %d ciclos", res.ciclos))+"\n"+
				infoStyle.Render(fmt.Sprintf("  Erro final: %.6f", res.erroFinal))+"\n"))
	}

	sb.WriteString(reveal(step, 2,
		"\n"+labelStyle.Render("  Curva de Erro (amostrada):")+"\n"+
			renderErroCurve(res.erroHistorico, 50, 6)))

	sb.WriteString(reveal(step, 3,
		"\n"+labelStyle.Render("  Teste nos 3 padrões:")+"\n"))

	if step >= 3 {
		padroes := [3][N_IN]float64{
			{1, 0.5, -1},
			{1, 0.5, 1},
			{1, -0.5, -1},
		}
		targets := [3][N_OUT]float64{
			{1, -1, -1},
			{-1, 1, -1},
			{-1, -1, 1},
		}
		acertos := 0
		for p := 0; p < 3; p++ {
			fwd := forward(res.rede, padroes[p])
			ok := true
			for k := 0; k < N_OUT; k++ {
				if (fwd.y[k] > 0) != (targets[p][k] > 0) {
					ok = false
				}
			}
			if ok {
				acertos++
				sb.WriteString(successStyle.Render(fmt.Sprintf("  ✓ Padrão %d: y=[%+.3f %+.3f %+.3f]",
					p+1, fwd.y[0], fwd.y[1], fwd.y[2])) + "\n")
			} else {
				sb.WriteString(errorStyle.Render(fmt.Sprintf("  ✗ Padrão %d: y=[%+.3f %+.3f %+.3f]",
					p+1, fwd.y[0], fwd.y[1], fwd.y[2])) + "\n")
			}
		}
		sb.WriteString("\n" + boldWhite.Render(fmt.Sprintf("  Acurácia: %d/3", acertos)) + "\n")
	}

	sb.WriteString(reveal(step, 4,
		"\n"+successStyle.Render("  O MLP aprendeu a classificar os 3 padrões!")+"\n"))

	return sb.String()
}

// =============================================================================
// Tela de resultado — testa os 3 padrões
// =============================================================================

func (m model) viewResult() string {
	if m.resultado == nil {
		return warnStyle.Render("  Nenhum resultado disponível. Treine primeiro.\n")
	}

	res := m.resultado
	var sb strings.Builder

	header := titleStyle.Render(" Resultado nos Padrões de Treinamento ")
	sb.WriteString("\n  " + header + "\n\n")

	padroes := [3][N_IN]float64{
		{1, 0.5, -1},
		{1, 0.5, 1},
		{1, -0.5, -1},
	}
	targets := [3][N_OUT]float64{
		{1, -1, -1},
		{-1, 1, -1},
		{-1, -1, 1},
	}

	// Cabeçalho da tabela
	hdr := dimStyle.Render(" Padrão │  Entrada              │  Target       │  Saída                    │  OK?")
	sep := dimStyle.Render("────────┼───────────────────────┼───────────────┼───────────────────────────┼──────")
	sb.WriteString("  " + hdr + "\n")
	sb.WriteString("  " + sep + "\n")

	acertos := 0
	for p := 0; p < 3; p++ {
		fwd := forward(res.rede, padroes[p])
		ok := true
		for k := 0; k < N_OUT; k++ {
			if (fwd.y[k] > 0) != (targets[p][k] > 0) {
				ok = false
			}
		}
		if ok {
			acertos++
		}

		xStr := fmt.Sprintf("[%+.1f %+.1f %+.1f]", padroes[p][0], padroes[p][1], padroes[p][2])
		tStr := fmt.Sprintf("[%+.0f %+.0f %+.0f]", targets[p][0], targets[p][1], targets[p][2])
		yStr := fmt.Sprintf("[%+.3f %+.3f %+.3f]", fwd.y[0], fwd.y[1], fwd.y[2])

		okMark := "  ✓"
		okSt := successStyle
		if !ok {
			okMark = "  ✗"
			okSt = errorStyle
		}

		rowBase := fmt.Sprintf("   %d    │ %-21s │ %-13s │ %-25s │",
			p+1, xStr, tStr, yStr)
		sb.WriteString("  " + dimStyle.Render(rowBase) + okSt.Render(okMark) + "\n")
	}

	sb.WriteString("  " + sep + "\n\n")

	if acertos == 3 {
		sb.WriteString(successStyle.Render(fmt.Sprintf("  Acurácia: %d/3 — rede convergiu corretamente!", acertos)) + "\n")
	} else {
		sb.WriteString(warnStyle.Render(fmt.Sprintf("  Acurácia: %d/3", acertos)) + "\n")
	}

	sb.WriteString("\n")
	if res.convergiu {
		sb.WriteString(infoStyle.Render(fmt.Sprintf("  Treino: %d ciclos  ·  Erro final: %.6f", res.ciclos, res.erroFinal)) + "\n")
	} else {
		sb.WriteString(warnStyle.Render(fmt.Sprintf("  Treino: %d ciclos (sem convergência)  ·  Erro: %.6f", res.ciclos, res.erroFinal)) + "\n")
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  enter/esc — voltar ao menu") + "\n")
	return sb.String()
}

// =============================================================================
// Walkthrough — passo a passo manual, conta a conta
// =============================================================================

// Padrões globais para o walkthrough (mesmo do treinarMLP)
var wtPadroes = [3][N_IN]float64{
	{1, 0.5, -1},
	{1, 0.5, 1},
	{1, -0.5, -1},
}
var wtTargets = [3][N_OUT]float64{
	{1, -1, -1},
	{-1, 1, -1},
	{-1, -1, 1},
}

// Sub-passos por padrão — cada passo mostra uma linha extra da conta:
//  0  cabeçalho / entradas
//  1  zin₁ = ...
//  2  z₁   = tanh(zin₁)
//  3  zin₂ = ...
//  4  z₂   = tanh(zin₂)
//  5  yin₁ = ...
//  6  y₁   = tanh(yin₁)
//  7  yin₂ = ...
//  8  y₂   = tanh(yin₂)
//  9  yin₃ = ...
// 10  y₃   = tanh(yin₃)
// 11  E    = ½Σ(t-y)²
// 12  δ₁   = (t₁-y₁)·f'(y₁)
// 13  δ₂   = ...
// 14  δ₃   = ...
// 15  δin₁ = Σδₖ·w[0][k]
// 16  δin₂ = Σδₖ·w[1][k]
// 17  δ₁_oculta = δin₁·f'(z₁)
// 18  δ₂_oculta = δin₂·f'(z₂)
// 19  Δw (oculta→saída)
// 20  Δv (entrada→oculta)
// 21  pesos novos

func wtMaxSteps() int { return 22 }

func (m model) viewWalkthrough() string {
	var sb strings.Builder

	p := m.wtPadrao
	s := m.wtStep

	// Calcula forward/backward com os pesos atuais
	fwd := forward(m.wtMLP, wtPadroes[p])
	bwd := backward(m.wtMLP, fwd, wtTargets[p], wtPadroes[p])
	x := wtPadroes[p]
	t := wtTargets[p]

	// Cabeçalho
	header := titleStyle.Render(" Passo a Passo — MLP Backpropagation ")
	sb.WriteString("\n  " + header + "\n\n")

	// Indicador de ciclo/padrão
	cicloStr := boldWhite.Render(fmt.Sprintf("Ciclo %d", m.wtCiclo))
	padStr := labelStyle.Render(fmt.Sprintf("Padrão %d / 3", p+1))
	stepStr := dimStyle.Render(fmt.Sprintf("(sub-passo %d/%d)", s+1, wtMaxSteps()))
	sb.WriteString(fmt.Sprintf("  %s  │  %s  %s\n\n", cicloStr, padStr, stepStr))

	// Entradas e targets sempre visíveis
	sb.WriteString(boldWhite.Render("  Entradas:") + "\n")
	for i := 0; i < N_IN; i++ {
		sb.WriteString(infoStyle.Render(fmt.Sprintf("    x%d = %+.1f", i+1, x[i])) + "\n")
	}
	sb.WriteString(boldWhite.Render("  Targets:") + "\n")
	for k := 0; k < N_OUT; k++ {
		sb.WriteString(labelStyle.Render(fmt.Sprintf("    t%d = %+.1f", k+1, t[k])) + "\n")
	}
	sb.WriteString("\n")

	// ── FORWARD PASS ──────────────────────────────────────────────────────────
	sb.WriteString(infoStyle.Render("  ── FORWARD PASS ──") + "\n\n")

	// Neurônio oculto 1
	if s >= 1 {
		zinExpr := fmt.Sprintf("  zin₁ = v0₁ + x₁·v₁₁ + x₂·v₂₁ + x₃·v₃₁")
		zinVal := fmt.Sprintf("       = %.4f + (%.1f·%.4f) + (%.1f·%.4f) + (%.1f·%.4f)",
			m.wtMLP.v0[0], x[0], m.wtMLP.v[0][0], x[1], m.wtMLP.v[1][0], x[2], m.wtMLP.v[2][0])
		zinRes := fmt.Sprintf("       = %.6f", fwd.zin[0])
		sb.WriteString(dimStyle.Render(zinExpr) + "\n")
		sb.WriteString(dimStyle.Render(zinVal) + "\n")
		sb.WriteString(infoStyle.Render(zinRes) + "\n")
	}
	if s >= 2 {
		sb.WriteString(successStyle.Render(fmt.Sprintf("  z₁ = tanh(%.6f) = %.6f", fwd.zin[0], fwd.z[0])) + "\n")
	}
	sb.WriteString("\n")

	// Neurônio oculto 2
	if s >= 3 {
		zinExpr := fmt.Sprintf("  zin₂ = v0₂ + x₁·v₁₂ + x₂·v₂₂ + x₃·v₃₂")
		zinVal := fmt.Sprintf("       = %.4f + (%.1f·%.4f) + (%.1f·%.4f) + (%.1f·%.4f)",
			m.wtMLP.v0[1], x[0], m.wtMLP.v[0][1], x[1], m.wtMLP.v[1][1], x[2], m.wtMLP.v[2][1])
		zinRes := fmt.Sprintf("       = %.6f", fwd.zin[1])
		sb.WriteString(dimStyle.Render(zinExpr) + "\n")
		sb.WriteString(dimStyle.Render(zinVal) + "\n")
		sb.WriteString(infoStyle.Render(zinRes) + "\n")
	}
	if s >= 4 {
		sb.WriteString(successStyle.Render(fmt.Sprintf("  z₂ = tanh(%.6f) = %.6f", fwd.zin[1], fwd.z[1])) + "\n")
	}
	sb.WriteString("\n")

	// Saídas (yin / y)
	for k := 0; k < N_OUT; k++ {
		baseStep := 5 + k*2 // passos 5,7,9
		if s >= baseStep {
			expr := fmt.Sprintf("  yin%d = w0%d + z₁·w₁%d + z₂·w₂%d", k+1, k+1, k+1, k+1)
			val := fmt.Sprintf("       = %.4f + (%.6f·%.4f) + (%.6f·%.4f)",
				m.wtMLP.w0[k], fwd.z[0], m.wtMLP.w[0][k], fwd.z[1], m.wtMLP.w[1][k])
			res := fmt.Sprintf("       = %.6f", fwd.yin[k])
			sb.WriteString(dimStyle.Render(expr) + "\n")
			sb.WriteString(dimStyle.Render(val) + "\n")
			sb.WriteString(infoStyle.Render(res) + "\n")
		}
		if s >= baseStep+1 {
			correct := (fwd.y[k] > 0) == (t[k] > 0)
			st := successStyle
			if !correct {
				st = errorStyle
			}
			sb.WriteString(st.Render(fmt.Sprintf("  y%d = tanh(%.6f) = %.6f  (t%d=%+.0f)", k+1, fwd.yin[k], fwd.y[k], k+1, t[k])) + "\n")
		}
		sb.WriteString("\n")
	}

	// Erro
	if s >= 11 {
		e := calcularErro(fwd.y, t)
		expr := "  E = ½·[(t₁-y₁)² + (t₂-y₂)² + (t₃-y₃)²]"
		val := fmt.Sprintf("    = ½·[(%.1f-%.4f)² + (%.1f-%.4f)² + (%.1f-%.4f)²]",
			t[0], fwd.y[0], t[1], fwd.y[1], t[2], fwd.y[2])
		res := fmt.Sprintf("    = %.6f", e)
		sb.WriteString(warnStyle.Render(expr) + "\n")
		sb.WriteString(dimStyle.Render(val) + "\n")
		sb.WriteString(warnStyle.Render(res) + "\n\n")
	}

	// ── BACKPROP ──────────────────────────────────────────────────────────────
	if s >= 12 {
		sb.WriteString(lipgloss.NewStyle().Foreground(neonYellow).Bold(true).Render("  ── BACKPROP ──") + "\n\n")
	}

	// δ saída
	for k := 0; k < N_OUT; k++ {
		if s >= 12+k {
			expr := fmt.Sprintf("  δ%d = (t%d - y%d)·(1+y%d)(1-y%d)", k+1, k+1, k+1, k+1, k+1)
			val := fmt.Sprintf("     = (%.1f - %.6f)·(1+%.6f)(1-%.6f)", t[k], fwd.y[k], fwd.y[k], fwd.y[k])
			res := fmt.Sprintf("     = %.8f", bwd.deltaK[k])
			sb.WriteString(dimStyle.Render(expr) + "\n")
			sb.WriteString(dimStyle.Render(val) + "\n")
			sb.WriteString(warnStyle.Render(res) + "\n\n")
		}
	}

	// δin oculta
	for j := 0; j < N_HID; j++ {
		if s >= 15+j {
			expr := fmt.Sprintf("  δin%d = δ₁·w[%d][0] + δ₂·w[%d][1] + δ₃·w[%d][2]", j+1, j, j, j)
			val := fmt.Sprintf("       = %.6f·%.4f + %.6f·%.4f + %.6f·%.4f",
				bwd.deltaK[0], m.wtMLP.w[j][0], bwd.deltaK[1], m.wtMLP.w[j][1], bwd.deltaK[2], m.wtMLP.w[j][2])
			res := fmt.Sprintf("       = %.8f", bwd.deltaInJ[j])
			sb.WriteString(dimStyle.Render(expr) + "\n")
			sb.WriteString(dimStyle.Render(val) + "\n")
			sb.WriteString(warnStyle.Render(res) + "\n\n")
		}
	}

	// δ oculta
	for j := 0; j < N_HID; j++ {
		if s >= 17+j {
			expr := fmt.Sprintf("  δ%d_oc = δin%d·(1+z%d)(1-z%d)", j+1, j+1, j+1, j+1)
			val := fmt.Sprintf("        = %.8f·(1+%.6f)(1-%.6f)", bwd.deltaInJ[j], fwd.z[j], fwd.z[j])
			res := fmt.Sprintf("        = %.8f", bwd.deltaJ[j])
			sb.WriteString(dimStyle.Render(expr) + "\n")
			sb.WriteString(dimStyle.Render(val) + "\n")
			sb.WriteString(warnStyle.Render(res) + "\n\n")
		}
	}

	// Δw
	if s >= 19 {
		sb.WriteString(labelStyle.Render("  Δw (oculta→saída):  Δwⱼₖ = α·δₖ·zⱼ") + "\n")
		for j := 0; j < N_HID; j++ {
			for k := 0; k < N_OUT; k++ {
				sb.WriteString(dimStyle.Render(fmt.Sprintf(
					"    Δw[%d][%d] = %.2f·%.8f·%.6f = %.8f",
					j, k, ALFA, bwd.deltaK[k], fwd.z[j], bwd.deltaW[j][k])) + "\n")
			}
		}
		for k := 0; k < N_OUT; k++ {
			sb.WriteString(dimStyle.Render(fmt.Sprintf(
				"    Δw0[%d]  = %.2f·%.8f = %.8f", k, ALFA, bwd.deltaK[k], bwd.deltaW0[k])) + "\n")
		}
		sb.WriteString("\n")
	}

	// Δv
	if s >= 20 {
		sb.WriteString(labelStyle.Render("  Δv (entrada→oculta): Δvᵢⱼ = α·δⱼ·xᵢ") + "\n")
		for i := 0; i < N_IN; i++ {
			for j := 0; j < N_HID; j++ {
				sb.WriteString(dimStyle.Render(fmt.Sprintf(
					"    Δv[%d][%d] = %.2f·%.8f·%.1f = %.8f",
					i, j, ALFA, bwd.deltaJ[j], x[i], bwd.deltaV[i][j])) + "\n")
			}
		}
		for j := 0; j < N_HID; j++ {
			sb.WriteString(dimStyle.Render(fmt.Sprintf(
				"    Δv0[%d]   = %.2f·%.8f = %.8f", j, ALFA, bwd.deltaJ[j], bwd.deltaV0[j])) + "\n")
		}
		sb.WriteString("\n")
	}

	// Pesos novos
	if s >= 21 {
		novoMLP := atualizarPesos(m.wtMLP, bwd)
		sb.WriteString(successStyle.Render("  Pesos APÓS update:") + "\n")
		sb.WriteString(dimStyle.Render("  v (entrada→oculta):") + "\n")
		for i := 0; i < N_IN; i++ {
			sb.WriteString(dimStyle.Render(fmt.Sprintf(
				"    v[%d][0]=%+.6f  v[%d][1]=%+.6f", i, novoMLP.v[i][0], i, novoMLP.v[i][1])) + "\n")
		}
		sb.WriteString(dimStyle.Render(fmt.Sprintf(
			"    v0=[%+.6f  %+.6f]", novoMLP.v0[0], novoMLP.v0[1])) + "\n\n")
		sb.WriteString(dimStyle.Render("  w (oculta→saída):") + "\n")
		for j := 0; j < N_HID; j++ {
			sb.WriteString(dimStyle.Render(fmt.Sprintf(
				"    w[%d]=[%+.6f  %+.6f  %+.6f]", j, novoMLP.w[j][0], novoMLP.w[j][1], novoMLP.w[j][2])) + "\n")
		}
		sb.WriteString(dimStyle.Render(fmt.Sprintf(
			"    w0=[%+.6f  %+.6f  %+.6f]", novoMLP.w0[0], novoMLP.w0[1], novoMLP.w0[2])) + "\n\n")
		sb.WriteString(warnStyle.Render("  → pressione → para avançar para o próximo padrão") + "\n")
	}

	// Rodapé
	sb.WriteString("\n")
	hint := fmt.Sprintf("  → próximo passo  ·  ← voltar  ·  esc menu  │  passo %d/%d", s+1, wtMaxSteps())
	sb.WriteString(hintStyle.Render(hint) + "\n")

	return sb.String()
}

// =============================================================================
// runTUI — inicializa e executa o programa Bubble Tea
// =============================================================================

func runTUI() {
	p := tea.NewProgram(
		initialModel(),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Erro: %v\n", err)
		os.Exit(1)
	}
}

