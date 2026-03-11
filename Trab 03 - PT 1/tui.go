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
	neonOrange  = lipgloss.Color("#FF8C00")
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
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF3030")).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(neonYellow).Bold(true)
	infoStyle    = lipgloss.NewStyle().Foreground(neonCyan)
	dimStyle     = lipgloss.NewStyle().Foreground(dimGray)
	midStyle     = lipgloss.NewStyle().Foreground(midGray)
	hintStyle    = lipgloss.NewStyle().Foreground(dimGray).Italic(true)
	weightStyle  = lipgloss.NewStyle().Foreground(neonCyan).Bold(true)
	labelStyle   = lipgloss.NewStyle().Foreground(neonPink)
	orangeStyle  = lipgloss.NewStyle().Foreground(neonOrange).Bold(true)
	boldWhite    = lipgloss.NewStyle().Foreground(softWhite).Bold(true)
)

// =============================================================================
// Estados
// =============================================================================

type sessionState int

const (
	stateMenu sessionState = iota
	stateTraining
	stateTrainingArch
	stateTrainingDone
	stateSlide     // apresentação estilo slide
	stateDrawLetter
	stateRecognition
)

// =============================================================================
// Model
// =============================================================================

type model struct {
	state    sessionState
	cursor   int
	choices  []string
	spinner  spinner.Model
	progress progress.Model

	winW int // largura do terminal
	winH int // altura do terminal

	resultado      *ResultadoTreino
	currentStepIdx int

	// slide
	slideIdx       int // slide atual (0-based)
	slideTotalStep int // sub-animação dentro do slide (para revelar elementos)
	slideStep      *TrainingStep // step fixado quando entrou no slide

	// grade de desenho
	grade     [N_ENTRADAS]int
	cursorLin int
	cursorCol int
	mouseDown bool

	// reconhecimento
	letraReconhecida int
	yInsReconhecidos [N_LETRAS]float64
}

// Coordenadas fixas da grade no terminal (calculadas a partir da estrutura do View)
// Padding outer: Padding(1,2) → top=1, left=2
// Header: 1 linha título + 1 linha subtítulo + 1 linha blank = 3
// Total linhas até grade: padding_top(1) + 3 + título_estado(1) + blank(1) = 6 (0-indexed: linha 6)
// Colunas: padding_left(2) + indentação("  ")=2 = 4
const gradeScreenRow = 6
const gradeScreenCol = 4

// largura de cada célula na grade: "█ " = 2 chars (exceto última coluna que não tem espaço)
// mas usamos separador " " entre células, então: posição da célula j = gradeScreenCol + j*2
func gradeColX(j int) int { return gradeScreenCol + j*2 }
func gradeRowY(i int) int { return gradeScreenRow + i }

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
		"Treinar MADALINE     — animação barra",
		"Treinar MADALINE     — diagrama arquitetura",
		"Apresentação MADALINE — slides interativos",
		"Desenhar e reconhecer letra",
		"Sair",
	}

	var grade [N_ENTRADAS]int
	for i := range grade {
		grade[i] = -1
	}

	return model{
		state:   stateMenu,
		choices: choices,
		spinner: s,
		progress: p,
		grade:   grade,
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
	return tea.Tick(80*time.Millisecond, func(t time.Time) tea.Msg {
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

	// ── Mouse ──────────────────────────────────────────────────────────────
	case tea.MouseMsg:
		if m.state == stateDrawLetter {
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

	// ── Teclado ────────────────────────────────────────────────────────────
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

		case stateTraining, stateTrainingArch:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc":
				m.currentStepIdx = len(m.resultado.steps)
				m.state = stateTrainingDone
				return m, nil
			case "enter", " ":
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
					// ainda revelando — pula para revelar completo
					m.slideTotalStep = slideMaxSteps(m.slideIdx)
				} else if m.slideIdx < totalSlides()-1 {
					m.slideIdx++
					m.slideTotalStep = 0
					return m, tickSlideReveal()
				}
			case "left", "h":
				if m.slideIdx > 0 {
					m.slideIdx--
					m.slideTotalStep = slideMaxSteps(m.slideIdx) // mostra completo ao voltar
				}
			}

		case stateDrawLetter:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc":
				m.state = stateMenu
				return m, nil
			case "up", "k":
				if m.cursorLin > 0 {
					m.cursorLin--
				}
			case "down", "j":
				if m.cursorLin < N_LINHAS-1 {
					m.cursorLin++
				}
			case "left", "h":
				if m.cursorCol > 0 {
					m.cursorCol--
				}
			case "right", "l":
				if m.cursorCol < N_COLUNAS-1 {
					m.cursorCol++
				}
			case " ":
				idx := m.cursorLin*N_COLUNAS + m.cursorCol
				if m.grade[idx] == 1 {
					m.grade[idx] = -1
				} else {
					m.grade[idx] = 1
				}
			case "enter":
				if m.resultado != nil {
					idx, yIns := reconhecer(m.resultado.rede, m.grade)
					m.letraReconhecida = idx
					m.yInsReconhecidos = yIns
					m.state = stateRecognition
				}
			case "r", "R":
				for i := range m.grade {
					m.grade[i] = -1
				}
			}

		case stateRecognition:
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc":
				m.state = stateMenu
				return m, nil
			case "enter", " ":
				m.state = stateDrawLetter
				return m, nil
			}
		}

	// ── Tick treinamento ───────────────────────────────────────────────────
	case trainingTickMsg:
		if (m.state == stateTraining || m.state == stateTrainingArch) && m.resultado != nil {
			total := len(m.resultado.steps)
			if m.currentStepIdx < total {
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

	// ── Tick slide reveal ─────────────────────────────────────────────────
	case slideRevealMsg:
		if m.state == stateSlide {
			max_ := slideMaxSteps(m.slideIdx)
			if m.slideTotalStep < max_ {
				m.slideTotalStep++
				return m, tickSlideReveal()
			}
		}

	// ── Spinner ────────────────────────────────────────────────────────────
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	}

	return m, nil
}

// paintAtMouse: toggle no click, só ativa no drag
func (m model) paintAtMouse(mx, my int, toggle bool) model {
	for i := 0; i < N_LINHAS; i++ {
		for j := 0; j < N_COLUNAS; j++ {
			// célula ocupa posições X: gradeColX(j) e gradeColX(j)+0 (1 char "█")
			// separador: 1 espaço entre células → célula j começa em gradeScreenCol + j*2
			cx := gradeScreenCol + j*2
			cy := gradeScreenRow + i
			if mx == cx && my == cy {
				idx := i*N_COLUNAS + j
				if toggle {
					if m.grade[idx] == 1 {
						m.grade[idx] = -1
					} else {
						m.grade[idx] = 1
					}
				} else {
					m.grade[idx] = 1
				}
				return m
			}
		}
	}
	return m
}

func (m model) handleMenuEnter() (tea.Model, tea.Cmd) {
	switch m.cursor {
	case 0: // treinar barra
		r := treinarMADALINE()
		m.resultado = &r
		m.currentStepIdx = 0
		m.state = stateTraining
		return m, tea.Batch(m.spinner.Tick, tickTraining())

	case 1: // treinar arquitetura
		r := treinarMADALINE()
		m.resultado = &r
		m.currentStepIdx = 0
		m.state = stateTrainingArch
		return m, tea.Batch(m.spinner.Tick, tickTraining())

	case 2: // slides
		// Garante que há uma rede treinada para os slides usarem dados reais
		if m.resultado == nil {
			r := treinarMADALINE()
			m.resultado = &r
		}
		// Fixa um step com erro para os slides mostrarem
		for i, s := range m.resultado.steps {
			for _, e := range s.erros {
				if e {
					step := m.resultado.steps[i]
					m.slideStep = &step
					break
				}
			}
			if m.slideStep != nil {
				break
			}
		}
		m.slideIdx = 0
		m.slideTotalStep = 0
		m.state = stateSlide
		return m, tickSlideReveal()

	case 3: // desenhar
		if m.resultado == nil {
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

	case 4:
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
	case stateTrainingArch:
		sb.WriteString(renderTrainingArch(m))
	case stateTrainingDone:
		sb.WriteString(renderTrainingDone(m))
	case stateSlide:
		sb.WriteString(renderSlide(m))
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
			sb.WriteString(successStyle.Render(fmt.Sprintf("✓ Rede treinada — convergiu em %d ciclos", m.resultado.ciclos)))
		} else {
			sb.WriteString(warnStyle.Render(fmt.Sprintf("⚠ Sem convergência após %d ciclos", MAX_CICLOS)))
		}
	} else {
		sb.WriteString(dimStyle.Render("Rede não treinada ainda."))
	}
	sb.WriteString("\n\n")

	sb.WriteString(boldWhite.Render("O que deseja fazer?") + "\n\n")

	for i, choice := range m.choices {
		cursor := "  "
		style := menuItemStyle
		if m.cursor == i {
			cursor = infoStyle.Render("▸ ")
			style = selectedItemStyle
		}
		sb.WriteString(style.Render(cursor+choice) + "\n")
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  ↑↓ mover • Enter selecionar • q sair"))
	return sb.String()
}

// =============================================================================
// Render: Treinamento — barra original
// =============================================================================

func renderTraining(m model) string {
	if m.resultado == nil {
		return ""
	}
	r := m.resultado
	total := len(r.steps)
	if m.currentStepIdx >= total {
		return renderTrainingDone(m)
	}
	step := r.steps[m.currentStepIdx]

	var sb strings.Builder
	pct := float64(m.currentStepIdx) / float64(maxInt(total, 1))
	sb.WriteString(fmt.Sprintf("%s Treinando MADALINE...\n\n", m.spinner.View()))
	sb.WriteString("  " + m.progress.ViewAs(pct))
	sb.WriteString(fmt.Sprintf("  %s\n\n", dimStyle.Render(fmt.Sprintf("step %d/%d", m.currentStepIdx, total))))

	col1 := renderColEntrada(step)
	col2 := renderColADALINE(step)
	col3 := renderColSaida(step)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		boxStyle.Render(col1), "  ",
		boxStyle.Render(col2), "  ",
		boxStyle.Render(col3),
	))
	sb.WriteString("\n\n")

	erroCount := 0
	for _, e := range step.erros {
		if e {
			erroCount++
		}
	}
	sb.WriteString(fmt.Sprintf("  Ciclo: %s  │  Letra: %s  │  ",
		infoStyle.Render(fmt.Sprintf("%d", step.ciclo)),
		weightStyle.Render(nomesLetras[step.letraIdx]),
	))
	if erroCount > 0 {
		sb.WriteString(errorStyle.Render(fmt.Sprintf("%d ADALINEs corrigidas", erroCount)))
	} else {
		sb.WriteString(successStyle.Render("nenhum erro"))
	}
	sb.WriteString("\n\n")
	sb.WriteString(hintStyle.Render("  Enter pula • Esc resultado"))
	return sb.String()
}

func renderColEntrada(step TrainingStep) string {
	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]
	var sb strings.Builder
	sb.WriteString(labelStyle.Bold(true).Render("  ENTRADA") + "\n\n")
	sb.WriteString(formataLetraGrid(entrada))
	return sb.String()
}

func renderColADALINE(step TrainingStep) string {
	var sb strings.Builder
	sb.WriteString(labelStyle.Bold(true).Render("  CAMADA ADALINE") + "\n\n")

	maxAbs := 0.01
	for _, v := range step.yIn {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	best := argmax(step.yIn[:])

	for j := 0; j < N_LETRAS; j++ {
		barLen := int(math.Abs(step.yIn[j]) / maxAbs * 12)
		if barLen < 1 {
			barLen = 1
		}
		bar := fmt.Sprintf("%-12s", strings.Repeat("█", barLen))
		yStr := fmt.Sprintf("%+6.2f", step.yIn[j])

		var linha string
		if step.erros[j] {
			linha = fmt.Sprintf(" %s %s %s %s",
				infoStyle.Bold(true).Render(nomesLetras[j]),
				infoStyle.Render(bar),
				infoStyle.Bold(true).Render(yStr),
				warnStyle.Render("← CORRIGIDA"),
			)
		} else if j == best {
			linha = fmt.Sprintf(" %s %s %s",
				successStyle.Render(nomesLetras[j]),
				successStyle.Render(bar),
				successStyle.Render(yStr),
			)
		} else {
			linha = fmt.Sprintf(" %s %s %s",
				boldWhite.Render(nomesLetras[j]),
				dimStyle.Render(bar),
				dimStyle.Render(yStr),
			)
		}
		sb.WriteString(linha + "\n")
	}
	return sb.String()
}

func renderColSaida(step TrainingStep) string {
	var sb strings.Builder
	sb.WriteString(labelStyle.Bold(true).Render("  SAÍDA") + "\n\n")
	best := argmax(step.yIn[:])
	sb.WriteString(lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).BorderForeground(neonGreen).
		Foreground(neonGreen).Bold(true).Padding(1, 3).
		Render(nomesLetras[best]))
	sb.WriteString("\n\n")
	sb.WriteString(dimStyle.Render("argmax\ny_in"))
	return sb.String()
}

// =============================================================================
// Render: Treinamento — arquitetura
// =============================================================================

func renderTrainingArch(m model) string {
	if m.resultado == nil {
		return ""
	}
	r := m.resultado
	total := len(r.steps)
	if m.currentStepIdx >= total {
		return renderTrainingDone(m)
	}
	step := r.steps[m.currentStepIdx]

	var sb strings.Builder
	pct := float64(m.currentStepIdx) / float64(maxInt(total, 1))
	sb.WriteString(fmt.Sprintf("%s Treinando MADALINE...\n\n", m.spinner.View()))
	sb.WriteString("  " + m.progress.ViewAs(pct))
	sb.WriteString(fmt.Sprintf("  %s\n\n",
		dimStyle.Render(fmt.Sprintf("step %d/%d  •  ciclo %d  •  letra %s",
			m.currentStepIdx, total, step.ciclo, nomesLetras[step.letraIdx]))))

	colDiag := renderArchDiagram(step, r)
	colMath := renderArchMath(step, r)
	colOut := renderArchSaida(step)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(colDiag), "  ",
		thinBox.Render(colMath), "  ",
		thinBox.Render(colOut),
	))
	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  Enter pula • Esc resultado"))
	return sb.String()
}

func renderArchDiagram(step TrainingStep, r *ResultadoTreino) string {
	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]
	var sb strings.Builder

	sb.WriteString(labelStyle.Bold(true).Render("ENTRADA") + " " +
		dimStyle.Render(fmt.Sprintf("(letra %s)", nomesLetras[step.letraIdx])) + "\n\n")

	for i := 0; i < N_LINHAS; i++ {
		for j := 0; j < N_COLUNAS; j++ {
			if entrada[i*N_COLUNAS+j] == 1 {
				sb.WriteString(lipgloss.NewStyle().Foreground(neonPink).Render("█"))
			} else {
				sb.WriteString(dimStyle.Render("·"))
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n")
	sb.WriteString(dimStyle.Render("  ╔══╪══╗") + "\n")
	sb.WriteString(dimStyle.Render("  ║35 x║") + "\n")
	sb.WriteString(dimStyle.Render("  ╚══╪══╝") + "\n\n")

	sb.WriteString(labelStyle.Bold(true).Render("ADALINES") + "\n\n")
	best := argmax(step.yIn[:])

	for j := 0; j < N_LETRAS; j++ {
		var circ, info string
		yStr := fmt.Sprintf("%+5.2f", step.yIn[j])
		if step.erros[j] {
			circ = infoStyle.Bold(true).Render("◉")
			info = fmt.Sprintf("%s %s %s",
				infoStyle.Bold(true).Render(nomesLetras[j]),
				infoStyle.Render(yStr),
				warnStyle.Render("✦"),
			)
		} else if j == best {
			circ = successStyle.Render("◉")
			info = successStyle.Render(fmt.Sprintf("%s %s ★", nomesLetras[j], yStr))
		} else {
			circ = dimStyle.Render("○")
			info = fmt.Sprintf("%s %s",
				midStyle.Render(nomesLetras[j]),
				dimStyle.Render(yStr),
			)
		}
		sb.WriteString(fmt.Sprintf("  %s %s\n", circ, info))
	}

	sb.WriteString("\n" + dimStyle.Render("  │\nargmax\n  │\n  ▼"))
	return sb.String()
}

func renderArchMath(step TrainingStep, r *ResultadoTreino) string {
	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]

	// foco: primeira ADALINE com erro
	foco := -1
	for j := 0; j < N_LETRAS; j++ {
		if step.erros[j] {
			foco = j
			break
		}
	}
	if foco == -1 {
		foco = argmax(step.yIn[:])
	}

	u := r.rede.unidades[foco]
	var sb strings.Builder

	var cor lipgloss.Color
	var statusStr string
	if step.erros[foco] {
		cor = neonCyan
		delta := ALFA * float64(step.target[foco]-step.y[foco])
		statusStr = fmt.Sprintf("%s  t=%+d  y=%+d  δ=%+.4f",
			warnStyle.Render("CORRIGIDA"),
			step.target[foco], step.y[foco], delta,
		)
	} else {
		cor = neonGreen
		statusStr = successStyle.Render("OK — sem correção")
	}

	sb.WriteString(lipgloss.NewStyle().Foreground(cor).Bold(true).
		Render(fmt.Sprintf("ADALINE-%s  (em foco)", nomesLetras[foco])) + "\n")
	sb.WriteString(statusStr + "\n\n")

	sb.WriteString(labelStyle.Bold(true).Render("Potencial de ativação:") + "\n")
	sb.WriteString(dimStyle.Render("  y_in = bias + Σ( wᵢ · xᵢ )") + "\n\n")
	sb.WriteString(fmt.Sprintf("  %s  %s\n", labelStyle.Render("bias"), weightStyle.Render(fmt.Sprintf("%+.6f", u.bias))))

	sb.WriteString("\n" + lipgloss.NewStyle().Foreground(neonPink).Bold(true).Render("  pixels ATIVOS (xᵢ=+1):") + "\n")
	countA, somaA := 0, 0.0
	for i := 0; i < N_ENTRADAS; i++ {
		if entrada[i] == 1 {
			c := u.pesos[i]
			somaA += c
			if countA < 10 {
				sb.WriteString(fmt.Sprintf("    w%-2d %s ×(+1)= %s\n",
					i,
					weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i])),
					lipgloss.NewStyle().Foreground(neonPink).Render(fmt.Sprintf("%+.4f", c)),
				))
			}
			countA++
		}
	}
	if countA > 10 {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("    …+%d omitidos  Σ=%+.4f\n", countA-10, somaA)))
	}

	sb.WriteString("\n" + dimStyle.Bold(true).Render("  pixels INATIVOS (xᵢ=−1):") + "\n")
	countI, somaI := 0, 0.0
	for i := 0; i < N_ENTRADAS; i++ {
		if entrada[i] == -1 {
			c := u.pesos[i] * -1
			somaI += c
			if countI < 6 {
				sb.WriteString(fmt.Sprintf("    w%-2d %s ×(−1)= %s\n",
					i,
					weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i])),
					dimStyle.Render(fmt.Sprintf("%+.4f", c)),
				))
			}
			countI++
		}
	}
	if countI > 6 {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("    …+%d omitidos  Σ=%+.4f\n", countI-6, somaI)))
	}

	sb.WriteString("\n" + dimStyle.Render(strings.Repeat("─", 40)) + "\n")
	yIn := step.yIn[foco]
	yAct := step.y[foco]
	sb.WriteString(fmt.Sprintf("  %s %s\n", labelStyle.Render("y_in   ="), weightStyle.Render(fmt.Sprintf("%+.6f", yIn))))
	sb.WriteString(fmt.Sprintf("  %s(%s) = %s\n", labelStyle.Render("f"), infoStyle.Render(fmt.Sprintf("%+.4f", yIn)),
		func() string {
			if yAct == 1 {
				return successStyle.Render("+1")
			}
			return errorStyle.Render("−1")
		}()))
	sb.WriteString(fmt.Sprintf("  %s %s\n", labelStyle.Render("target ="),
		func() string {
			if step.target[foco] == 1 {
				return successStyle.Render("+1")
			}
			return dimStyle.Render("−1")
		}()))

	if step.erros[foco] {
		delta := ALFA * float64(step.target[foco]-step.y[foco])
		sb.WriteString("\n" + warnStyle.Render("  Regra Delta:") + "\n")
		sb.WriteString(fmt.Sprintf("  δ = %.2f×(%+d−%+d) = %s\n",
			ALFA, step.target[foco], step.y[foco],
			warnStyle.Render(fmt.Sprintf("%+.4f", delta))))
		sb.WriteString(dimStyle.Render("  wᵢ ← wᵢ + δ·xᵢ\n  bias ← bias + δ"))
	}
	return sb.String()
}

func renderArchSaida(step TrainingStep) string {
	var sb strings.Builder
	sb.WriteString(labelStyle.Bold(true).Render("SAÍDA") + "\n\n")
	best := argmax(step.yIn[:])

	sb.WriteString(dimStyle.Render("  ○ ○ … ○\n   ╲ │ ╱\n  argmax\n    │\n    ▼") + "\n\n")

	sb.WriteString(lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).BorderForeground(neonGreen).
		Foreground(neonGreen).Bold(true).Padding(1, 2).
		Render(nomesLetras[best]))
	sb.WriteString("\n\n")

	sb.WriteString(labelStyle.Render("Top 3:") + "\n")
	idxs := make([]int, N_LETRAS)
	for i := range idxs {
		idxs[i] = i
	}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < N_LETRAS; j++ {
			if step.yIn[idxs[j]] > step.yIn[idxs[i]] {
				idxs[i], idxs[j] = idxs[j], idxs[i]
			}
		}
	}
	medals := []string{"①", "②", "③"}
	for i := 0; i < 3; i++ {
		j := idxs[i]
		st := dimStyle
		if j == best {
			st = successStyle
		}
		sb.WriteString(fmt.Sprintf("  %s %s %s\n",
			st.Render(medals[i]),
			st.Render(nomesLetras[j]),
			st.Render(fmt.Sprintf("%+.3f", step.yIn[j])),
		))
	}

	erroCount := 0
	for _, e := range step.erros {
		if e {
			erroCount++
		}
	}
	sb.WriteString("\n")
	if erroCount > 0 {
		sb.WriteString(errorStyle.Render(fmt.Sprintf("✦ %d corrigidas", erroCount)))
	} else {
		sb.WriteString(successStyle.Render("✓ sem erros"))
	}
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

	infoBox := fmt.Sprintf("%s\n\n  %s %s\n  %s %s\n  %s %s",
		labelStyle.Bold(true).Render("Resumo do Treinamento"),
		labelStyle.Render("Convergiu:    "), func() string {
			if r.convergiu {
				return successStyle.Render("SIM")
			}
			return errorStyle.Render("NÃO")
		}(),
		labelStyle.Render("Ciclos:       "), weightStyle.Render(fmt.Sprintf("%d", r.ciclos)),
		labelStyle.Render("Steps c/ erro:"), weightStyle.Render(fmt.Sprintf("%d", len(r.steps))),
	)
	sb.WriteString(boxStyle.Render(infoBox))
	sb.WriteString("\n\n")
	sb.WriteString(hintStyle.Render("  Enter → menu"))
	return sb.String()
}

// =============================================================================
// Render: Slides MADALINE
// =============================================================================

// totalSlides retorna o número total de slides.
func totalSlides() int { return 7 }

// slideMaxSteps controla quantos elementos são revelados progressivamente por slide.
func slideMaxSteps(idx int) int {
	steps := []int{8, 10, 12, 14, 12, 10, 8}
	if idx < len(steps) {
		return steps[idx]
	}
	return 8
}

func renderSlide(m model) string {
	nav := fmt.Sprintf("  Slide %d/%d", m.slideIdx+1, totalSlides())
	navBar := lipgloss.JoinHorizontal(lipgloss.Center,
		hintStyle.Render("  ← → navegar • Espaço/Enter avançar • q sair"),
		strings.Repeat(" ", maxInt(0, m.winW-80)),
		infoStyle.Render(nav),
	)

	var content string
	s := m.slideTotalStep
	step := m.slideStep

	switch m.slideIdx {
	case 0:
		content = slideArquitetura(s)
	case 1:
		content = slideEntrada(s, step)
	case 2:
		content = slideADALINEUnica(s, step, m.resultado)
	case 3:
		content = slideSomatorio(s, step, m.resultado)
	case 4:
		content = slideAtivacao(s, step)
	case 5:
		content = slideRegraDelta(s, step, m.resultado)
	case 6:
		content = slideSaida(s, step)
	}

	return content + "\n\n" + navBar
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 0: Visão geral da arquitetura MADALINE
// ─────────────────────────────────────────────────────────────────────────────
func slideArquitetura(s int) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonMagenta).
		Padding(0, 4).Render("  ARQUITETURA MADALINE  ")
	sb.WriteString(title + "\n\n")

	if s < 1 {
		return sb.String()
	}

	// Diagrama de camadas
	inputCol := buildInputColumn(s)
	midCol := buildMiddleArrows(s)
	adalineCol := buildADALINEColumn(s)
	outArrow := buildOutputArrow(s)
	outputCol := buildOutputColumn(s)

	diagram := lipgloss.JoinHorizontal(lipgloss.Center,
		inputCol, midCol, adalineCol, outArrow, outputCol,
	)
	sb.WriteString(diagram)

	if s >= 7 {
		sb.WriteString("\n\n")
		sb.WriteString(slideBox.Render(
			boldWhite.Render("Resumo da Arquitetura:\n\n") +
				fmt.Sprintf("  %s  35 pixels (grade 5×7) em bipolar −1/+1\n", infoStyle.Render("●")) +
				fmt.Sprintf("  %s  13 unidades ADALINE — uma por letra A–M\n", labelStyle.Render("●")) +
				fmt.Sprintf("  %s  Saída: argmax dos y_in (vencedor)\n", successStyle.Render("●")) +
				fmt.Sprintf("  %s  Codificação One-of-N\n", orangeStyle.Render("●")),
		))
	}

	return sb.String()
}

func buildInputColumn(s int) string {
	var sb strings.Builder
	sb.WriteString(infoStyle.Bold(true).Render("  ENTRADAS") + "\n")
	sb.WriteString(dimStyle.Render("  (35 pixels)") + "\n\n")

	pixels := []string{"x₁", "x₂", "x₃", "⋮", "x₃₅"}
	for i, p := range pixels {
		if s < i+2 {
			break
		}
		if p == "⋮" {
			sb.WriteString(dimStyle.Render("    " + p) + "\n")
		} else {
			sb.WriteString(infoStyle.Render(fmt.Sprintf("  ○ %s = ±1", p)) + "\n")
		}
	}
	return sb.String()
}

func buildMiddleArrows(s int) string {
	if s < 3 {
		return "     "
	}
	var sb strings.Builder
	sb.WriteString("        \n\n\n")
	lines := []string{"  ──▶", "  ──▶", "  ──▶", "   ⋮ ", "  ──▶"}
	for i, l := range lines {
		if s < i+3 {
			sb.WriteString("      \n")
		} else {
			sb.WriteString(dimStyle.Render(l) + "\n")
		}
	}
	return sb.String()
}

func buildADALINEColumn(s int) string {
	if s < 3 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString(labelStyle.Bold(true).Render("  ADALINE (camada 1)") + "\n")
	sb.WriteString(dimStyle.Render("  (13 unidades)") + "\n\n")

	letters := []string{"A", "B", "C", "⋮", "M"}
	for i, l := range letters {
		if s < i+3 {
			break
		}
		if l == "⋮" {
			sb.WriteString(dimStyle.Render("     " + l) + "\n")
		} else {
			color := neonMagenta
			if l == "A" {
				color = neonGreen
			}
			sb.WriteString(lipgloss.NewStyle().Foreground(color).
				Render(fmt.Sprintf("  ◉ ADALINE-%s", l)) + "\n")
		}
	}
	return sb.String()
}

func buildOutputArrow(s int) string {
	if s < 6 {
		return ""
	}
	return dimStyle.Render("\n\n\n  ──argmax──▶")
}

func buildOutputColumn(s int) string {
	if s < 6 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString(successStyle.Bold(true).Render("  SAÍDA") + "\n")
	sb.WriteString(dimStyle.Render("  (camada 2)") + "\n\n")
	sb.WriteString(lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(neonGreen).
		Foreground(neonGreen).Bold(true).
		Padding(0, 2).Render("letra\nvencedora"))
	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 1: Codificação da entrada
// ─────────────────────────────────────────────────────────────────────────────
func slideEntrada(s int, step *TrainingStep) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonCyan).
		Padding(0, 4).Render("  SLIDE 2 — CODIFICAÇÃO DA ENTRADA  ")
	sb.WriteString(title + "\n\n")

	if s < 1 {
		return sb.String()
	}

	dataset := letrasDataset()
	letraIdx := 0
	if step != nil {
		letraIdx = step.letraIdx
	}
	entrada := dataset[letraIdx]

	// Coluna esquerda: grade visual
	left := buildEntradaGrade(entrada, s)
	// Coluna direita: explicação
	right := buildEntradaExplicacao(entrada, letraIdx, s)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	if s >= 9 {
		sb.WriteString("\n\n")
		sb.WriteString(slideBox.Render(
			warnStyle.Render("Por que bipolar?\n\n") +
				dimStyle.Render("  Com entrada binária 0/1: se xᵢ=0, então wᵢ·xᵢ=0\n") +
				dimStyle.Render("  → o peso wᵢ NUNCA é corrigido pelo pixel apagado.\n\n") +
				successStyle.Render("  Com −1/+1: pixel apagado contribui −wᵢ\n") +
				successStyle.Render("  → todos os pesos participam do aprendizado!"),
		))
	}

	return sb.String()
}

func buildEntradaGrade(entrada [N_ENTRADAS]int, s int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Grade 5×7") + " " + dimStyle.Render("(representação visual)") + "\n\n")

	reveal := s * 5 // revela 5 pixels por step
	count := 0
	for i := 0; i < N_LINHAS; i++ {
		sb.WriteString("  ")
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			idx := i*N_COLUNAS + j
			if count < reveal {
				if entrada[idx] == 1 {
					sb.WriteString(lipgloss.NewStyle().Foreground(neonPink).Render("█"))
				} else {
					sb.WriteString(dimStyle.Render("·"))
				}
			} else {
				sb.WriteString(dimStyle.Render("?"))
			}
			count++
		}
		sb.WriteString("\n")
	}

	if s >= 8 {
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render("  ") + infoStyle.Render("█") + dimStyle.Render(" = +1  ") +
			dimStyle.Render("· = −1"))
	}
	return sb.String()
}

func buildEntradaExplicacao(entrada [N_ENTRADAS]int, letraIdx int, s int) string {
	var sb strings.Builder
	sb.WriteString(boldWhite.Render("Vetor de entrada x") + "\n\n")

	if s < 3 {
		sb.WriteString(dimStyle.Render("  aguardando..."))
		return sb.String()
	}

	sb.WriteString(fmt.Sprintf("  %s Letra: %s\n\n",
		labelStyle.Render("→"),
		infoStyle.Bold(true).Render(nomesLetras[letraIdx]),
	))

	// Mostra primeiros pixels como vetor
	sb.WriteString(dimStyle.Render("  x = [") + "\n")
	count := 0
	ativos := 0
	inativos := 0
	for i := 0; i < N_ENTRADAS; i++ {
		if entrada[i] == 1 {
			ativos++
		} else {
			inativos++
		}
		if s >= 4 && count < 10 {
			val := entrada[i]
			var valStr string
			if val == 1 {
				valStr = lipgloss.NewStyle().Foreground(neonPink).Render("+1")
			} else {
				valStr = dimStyle.Render("−1")
			}
			sb.WriteString(fmt.Sprintf("    x%-2d = %s\n", i, valStr))
			count++
		}
	}
	if s >= 4 {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("    … +%d valores\n", N_ENTRADAS-10)))
		sb.WriteString(dimStyle.Render("  ]") + "\n")
	}

	if s >= 6 {
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf("  %s pixels ativos  %s = +1\n",
			lipgloss.NewStyle().Foreground(neonPink).Bold(true).Render(fmt.Sprintf("%d", ativos)),
			lipgloss.NewStyle().Foreground(neonPink).Render("(█)"),
		))
		sb.WriteString(fmt.Sprintf("  %s pixels inativos %s = −1\n",
			dimStyle.Render(fmt.Sprintf("%d", inativos)),
			dimStyle.Render("(·)"),
		))
	}

	if s >= 8 {
		sb.WriteString("\n")
		sb.WriteString(warnStyle.Render(fmt.Sprintf("  Target one-of-N:\n")))
		sb.WriteString(dimStyle.Render("  t = [ −1, −1, "))
		sb.WriteString(successStyle.Render("+1"))
		sb.WriteString(dimStyle.Render(fmt.Sprintf(", … ]  ← +1 na pos %d", letraIdx)))
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 2: Uma ADALINE em detalhe
// ─────────────────────────────────────────────────────────────────────────────
func slideADALINEUnica(s int, step *TrainingStep, r *ResultadoTreino) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonMagenta).
		Padding(0, 4).Render("  SLIDE 3 — ESTRUTURA DA ADALINE  ")
	sb.WriteString(title + "\n\n")

	if s < 1 {
		return sb.String()
	}

	foco := 0
	if step != nil {
		for j, e := range step.erros {
			if e {
				foco = j
				break
			}
		}
	}

	left := buildADALINEStructure(foco, s, r)
	right := buildADALINEFormula(foco, s, step, r)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	return sb.String()
}

func buildADALINEStructure(foco, s int, r *ResultadoTreino) string {
	var sb strings.Builder

	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).
		Render(fmt.Sprintf("ADALINE-%s", nomesLetras[foco])) + "\n\n")

	if s < 2 {
		return sb.String()
	}

	// Diagrama da ADALINE com entradas, pesos e somador
	inputs := []string{"x₀", "x₁", "x₂", "⋮", "x₃₄"}
	weights := []string{"w₀", "w₁", "w₂", "⋮", "w₃₄"}

	var u *ADALINEUnit
	if r != nil {
		u2 := r.rede.unidades[foco]
		u = &u2
	}

	for i, inp := range inputs {
		if s < i+2 {
			break
		}
		if inp == "⋮" {
			sb.WriteString(dimStyle.Render("  ⋮      ⋮\n"))
			continue
		}

		var wStr string
		if u != nil && i < N_ENTRADAS {
			wStr = weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i]))
		} else {
			wStr = dimStyle.Render("  w?  ")
		}

		line := fmt.Sprintf("  %s ──[%s]──┐\n",
			infoStyle.Render(inp),
			wStr,
		)
		if i == 2 {
			line = fmt.Sprintf("  %s ──[%s]──┤──[ Σ+b ]──[ f ]──▶ y\n",
				infoStyle.Render(inp),
				wStr,
			)
		} else if i > 2 && inp != "⋮" {
			line = fmt.Sprintf("  %s ──[%s]──┘\n",
				infoStyle.Render(inp),
				wStr,
			)
		}
		sb.WriteString(line)
		_ = weights[i]
	}

	if s >= 6 {
		sb.WriteString("\n")
		if u != nil {
			sb.WriteString(fmt.Sprintf("  %s %s\n",
				labelStyle.Render("bias ="),
				weightStyle.Render(fmt.Sprintf("%+.4f", u.bias)),
			))
		}
	}

	if s >= 8 {
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render("  Pesos: 35 + 1 bias = 36 params"))
	}

	return sb.String()
}

func buildADALINEFormula(foco, s int, step *TrainingStep, r *ResultadoTreino) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("Parâmetros e operação") + "\n\n")

	if s < 2 {
		return sb.String()
	}

	if s >= 2 {
		sb.WriteString(labelStyle.Bold(true).Render("Pesos sinápticos:") + "\n")
		sb.WriteString(dimStyle.Render("  (aprendidos pela Regra Delta)\n\n"))
	}

	if r != nil && s >= 3 {
		u := r.rede.unidades[foco]
		for i := 0; i < minInt(8, N_ENTRADAS); i++ {
			if s < i+3 {
				break
			}
			bar := ""
			barLen := int(math.Abs(u.pesos[i]) / 0.5 * 8)
			if barLen < 1 {
				barLen = 1
			}
			if barLen > 8 {
				barLen = 8
			}
			if u.pesos[i] > 0 {
				bar = lipgloss.NewStyle().Foreground(neonPink).Render(strings.Repeat("▶", barLen))
			} else {
				bar = dimStyle.Render(strings.Repeat("◀", barLen))
			}
			sb.WriteString(fmt.Sprintf("  w%-2d %s %s\n",
				i,
				weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i])),
				bar,
			))
		}
		sb.WriteString(dimStyle.Render("  …\n"))
	}

	if s >= 10 {
		sb.WriteString("\n" + labelStyle.Bold(true).Render("Interpretação:") + "\n")
		sb.WriteString(dimStyle.Render("  Peso alto positivo → pixel ativo\n"))
		sb.WriteString(dimStyle.Render("  favorece reconhecer esta letra\n"))
		sb.WriteString(dimStyle.Render("  Peso alto negativo → pixel ativo\n"))
		sb.WriteString(dimStyle.Render("  prejudica o reconhecimento"))
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 3: Somatório — cálculo completo do y_in
// ─────────────────────────────────────────────────────────────────────────────
func slideSomatorio(s int, step *TrainingStep, r *ResultadoTreino) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonCyan).
		Padding(0, 4).Render("  SLIDE 4 — CÁLCULO DO POTENCIAL y_in  ")
	sb.WriteString(title + "\n\n")

	if step == nil || r == nil {
		sb.WriteString(dimStyle.Render("(sem dados de treinamento)"))
		return sb.String()
	}

	foco := 0
	for j, e := range step.erros {
		if e {
			foco = j
			break
		}
	}

	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]
	u := r.rede.unidades[foco]

	left := buildSomatorioFormula(foco, s, entrada, u, step)
	right := buildSomatorioVisualizacao(foco, s, entrada, u, step)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	return sb.String()
}

func buildSomatorioFormula(foco, s int, entrada [N_ENTRADAS]int, u ADALINEUnit, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(lipgloss.NewStyle().Foreground(neonMagenta).Bold(true).
		Render(fmt.Sprintf("ADALINE-%s — fórmula", nomesLetras[foco])) + "\n\n")

	if s >= 1 {
		sb.WriteString(boldWhite.Render("  y_in = b + Σᵢ(wᵢ · xᵢ)") + "\n\n")
	}

	if s >= 2 {
		sb.WriteString(fmt.Sprintf("  %s  %s\n\n",
			labelStyle.Render("b   ="),
			weightStyle.Render(fmt.Sprintf("%+.5f", u.bias)),
		))
	}

	soma := u.bias
	shown := 0
	for i := 0; i < N_ENTRADAS; i++ {
		contrib := u.pesos[i] * float64(entrada[i])
		soma += contrib

		if s >= 3+shown/2 && shown < 12 {
			xSign := "+1"
			xColor := neonPink
			if entrada[i] == -1 {
				xSign = "−1"
				xColor = dimGray
			}
			contribStyle := dimStyle
			if contrib > 0 {
				contribStyle = lipgloss.NewStyle().Foreground(neonGreen)
			} else if contrib < 0 {
				contribStyle = errorStyle
			}
			sb.WriteString(fmt.Sprintf("  w%-2d(%s) × %s = %s\n",
				i,
				weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i])),
				lipgloss.NewStyle().Foreground(xColor).Render(xSign),
				contribStyle.Render(fmt.Sprintf("%+.4f", contrib)),
			))
			shown++
		}
	}

	if s >= 9 {
		sb.WriteString(dimStyle.Render(fmt.Sprintf("  … +%d termos\n", N_ENTRADAS-shown)))
	}

	if s >= 11 {
		sb.WriteString("\n" + dimStyle.Render(strings.Repeat("─", 36)) + "\n")
		sb.WriteString(fmt.Sprintf("  %s %s\n",
			labelStyle.Render("y_in ="),
			weightStyle.Render(fmt.Sprintf("%+.6f", step.yIn[foco])),
		))
	}

	return sb.String()
}

func buildSomatorioVisualizacao(foco, s int, entrada [N_ENTRADAS]int, u ADALINEUnit, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("Visualização das contribuições") + "\n\n")

	if s < 2 {
		return sb.String()
	}

	// Mapa de calor dos pesos × entrada
	sb.WriteString(dimStyle.Render("  wᵢ·xᵢ por pixel (grade 5×7):\n\n"))

	maxContrib := 0.001
	contribs := make([]float64, N_ENTRADAS)
	for i := 0; i < N_ENTRADAS; i++ {
		contribs[i] = u.pesos[i] * float64(entrada[i])
		if a := math.Abs(contribs[i]); a > maxContrib {
			maxContrib = a
		}
	}

	for i := 0; i < N_LINHAS; i++ {
		sb.WriteString("  ")
		for j := 0; j < N_COLUNAS; j++ {
			idx := i*N_COLUNAS + j
			c := contribs[idx]
			norm := c / maxContrib // −1..+1

			var ch string
			if s < 3 {
				ch = dimStyle.Render("?")
			} else if norm > 0.6 {
				ch = lipgloss.NewStyle().Foreground(neonGreen).Bold(true).Render("▓")
			} else if norm > 0.2 {
				ch = lipgloss.NewStyle().Foreground(neonGreen).Render("▒")
			} else if norm > -0.2 {
				ch = dimStyle.Render("░")
			} else if norm > -0.6 {
				ch = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF3030")).Render("▒")
			} else {
				ch = errorStyle.Render("▓")
			}
			sb.WriteString(ch + " ")
		}
		sb.WriteString("\n")
	}

	if s >= 6 {
		sb.WriteString("\n")
		sb.WriteString(successStyle.Render("  ▓▒") + dimStyle.Render(" contribuição positiva\n"))
		sb.WriteString(errorStyle.Render("  ▒▓") + dimStyle.Render(" contribuição negativa\n"))
	}

	if s >= 10 {
		soma := step.yIn[foco]
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf("  Σ todas = %s\n",
			weightStyle.Render(fmt.Sprintf("%+.6f", soma)),
		))
		sb.WriteString(dimStyle.Render("  (+ bias incluído)"))
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 4: Função de ativação degrau bipolar
// ─────────────────────────────────────────────────────────────────────────────
func slideAtivacao(s int, step *TrainingStep) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonMagenta).
		Padding(0, 4).Render("  SLIDE 5 — FUNÇÃO DE ATIVAÇÃO  ")
	sb.WriteString(title + "\n\n")

	if s < 1 {
		return sb.String()
	}

	left := buildAtivacaoGrafico(s)
	right := buildAtivacaoExemplos(s, step)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	return sb.String()
}

func buildAtivacaoGrafico(s int) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("Degrau Bipolar  f(y_in)") + "\n\n")

	if s >= 1 {
		sb.WriteString(dimStyle.Render("   +1 ") + successStyle.Render("┄┄┄┄┄┄┄┄┄┄") + "\n")
		sb.WriteString(dimStyle.Render("      ") + dimStyle.Render("          ") + "\n")
	}

	if s >= 2 {
		sb.WriteString(dimStyle.Render("    0 ") + dimStyle.Render("────┬─────────") + "\n")
		sb.WriteString(dimStyle.Render("      ") + dimStyle.Render("    │") + "\n")
	}

	if s >= 3 {
		sb.WriteString(dimStyle.Render("   −1 ") + errorStyle.Render("┄┄┄┄") + "\n")
		sb.WriteString(dimStyle.Render("      ") + "\n")
		sb.WriteString(dimStyle.Render("      −∞   0   +∞  →y_in") + "\n")
	}

	if s >= 4 {
		sb.WriteString("\n")
		sb.WriteString(boldWhite.Render("Definição:") + "\n\n")
	}

	if s >= 5 {
		sb.WriteString(fmt.Sprintf("  %s y_in ≥ 0  →  f = %s\n",
			dimStyle.Render("se"),
			successStyle.Render("+1"),
		))
	}

	if s >= 6 {
		sb.WriteString(fmt.Sprintf("  %s y_in < 0  →  f = %s\n",
			dimStyle.Render("se"),
			errorStyle.Render("−1"),
		))
	}

	if s >= 8 {
		sb.WriteString("\n")
		sb.WriteString(warnStyle.Render("Por que bipolar e não 0/1?\n"))
		sb.WriteString(dimStyle.Render("  Derivada simétrica → gradiente\n"))
		sb.WriteString(dimStyle.Render("  mais estável na Regra Delta"))
	}

	return sb.String()
}

func buildAtivacaoExemplos(s int, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("Exemplos neste step") + "\n\n")

	if step == nil || s < 2 {
		sb.WriteString(dimStyle.Render("  (aguardando dados...)"))
		return sb.String()
	}

	shown := 0
	for j := 0; j < N_LETRAS && shown < 8; j++ {
		if s < shown+2 {
			break
		}
		yIn := step.yIn[j]
		yAct := step.y[j]
		target := step.target[j]

		var fStr, tStr string
		if yAct == 1 {
			fStr = successStyle.Render("+1")
		} else {
			fStr = errorStyle.Render("−1")
		}
		if target == 1 {
			tStr = successStyle.Render("+1")
		} else {
			tStr = dimStyle.Render("−1")
		}

		match := "✓"
		matchStyle := successStyle
		if yAct != target {
			match = "✗"
			matchStyle = errorStyle
		}

		sb.WriteString(fmt.Sprintf("  %s y_in=%s → f=%s t=%s %s\n",
			infoStyle.Render(fmt.Sprintf("%-2s", nomesLetras[j])),
			weightStyle.Render(fmt.Sprintf("%+.3f", yIn)),
			fStr, tStr,
			matchStyle.Render(match),
		))
		shown++
	}

	if s >= 10 {
		sb.WriteString("\n")
		erros := 0
		for _, e := range step.erros {
			if e {
				erros++
			}
		}
		if erros > 0 {
			sb.WriteString(errorStyle.Render(fmt.Sprintf("  %d ADALINEs precisam ser\n  corrigidas neste step!", erros)))
		} else {
			sb.WriteString(successStyle.Render("  Todas as ADALINEs corretas!"))
		}
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 5: Regra Delta — atualização dos pesos
// ─────────────────────────────────────────────────────────────────────────────
func slideRegraDelta(s int, step *TrainingStep, r *ResultadoTreino) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonYellow).
		Padding(0, 4).Foreground(darkBg).Render("  SLIDE 6 — REGRA DELTA (APRENDIZADO)  ")
	sb.WriteString(title + "\n\n")

	if s < 1 {
		return sb.String()
	}

	foco := 0
	if step != nil {
		for j, e := range step.erros {
			if e {
				foco = j
				break
			}
		}
	}

	left := buildDeltaFormula(foco, s, step)
	right := buildDeltaNumerical(foco, s, step, r)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	return sb.String()
}

func buildDeltaFormula(foco, s int, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(warnStyle.Bold(true).Render("Regra Delta") + "\n\n")

	if s >= 1 {
		sb.WriteString(dimStyle.Render("  Atualizar SOMENTE quando:\n"))
		sb.WriteString(fmt.Sprintf("    f(y_in) %s target\n\n", errorStyle.Render("≠")))
	}

	if s >= 2 {
		sb.WriteString(boldWhite.Render("  Passos:\n\n"))
	}

	steps := []struct {
		step int
		text string
	}{
		{3, "① Calcular erro:\n     e = target − y\n"},
		{4, "② Calcular delta:\n     δ = α × e\n     δ = α × (target − y)\n"},
		{5, "③ Atualizar pesos:\n     wᵢ ← wᵢ + δ · xᵢ\n"},
		{6, "④ Atualizar bias:\n     b ← b + δ\n"},
	}

	for _, st := range steps {
		if s >= st.step {
			sb.WriteString(infoStyle.Render("  "+st.text) + "\n")
		}
	}

	if s >= 7 {
		sb.WriteString("\n")
		sb.WriteString(labelStyle.Bold(true).Render("  Parâmetros:\n"))
		sb.WriteString(fmt.Sprintf("    α (alfa) = %s\n", weightStyle.Render(fmt.Sprintf("%.2f", ALFA))))
		sb.WriteString(dimStyle.Render("    (taxa de aprendizagem)\n"))
	}

	if s >= 9 {
		sb.WriteString("\n")
		sb.WriteString(dimStyle.Render("  Convergência:\n"))
		sb.WriteString(dimStyle.Render("  Loop até NENHUMA ADALINE\n"))
		sb.WriteString(dimStyle.Render("  errar em ciclo completo."))
	}

	return sb.String()
}

func buildDeltaNumerical(foco, s int, step *TrainingStep, r *ResultadoTreino) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render(fmt.Sprintf("Exemplo: ADALINE-%s", nomesLetras[foco])) + "\n\n")

	if step == nil || r == nil || s < 2 {
		sb.WriteString(dimStyle.Render("  (aguardando dados...)"))
		return sb.String()
	}

	if !step.erros[foco] {
		sb.WriteString(successStyle.Render("  Esta ADALINE não teve erro\n"))
		sb.WriteString(dimStyle.Render("  → sem atualização de pesos"))
		return sb.String()
	}

	u := r.rede.unidades[foco]
	dataset := letrasDataset()
	entrada := dataset[step.letraIdx]
	target := step.target[foco]
	yAct := step.y[foco]
	delta := ALFA * float64(target-yAct)

	if s >= 2 {
		sb.WriteString(fmt.Sprintf("  y_in  = %s\n", weightStyle.Render(fmt.Sprintf("%+.5f", step.yIn[foco]))))
		sb.WriteString(fmt.Sprintf("  f     = %s\n", func() string {
			if yAct == 1 {
				return successStyle.Render("+1")
			}
			return errorStyle.Render("−1")
		}()))
		sb.WriteString(fmt.Sprintf("  target= %s\n\n", func() string {
			if target == 1 {
				return successStyle.Render("+1")
			}
			return errorStyle.Render("−1")
		}()))
	}

	if s >= 4 {
		erro := target - yAct
		sb.WriteString(fmt.Sprintf("  e = %+d − (%+d) = %s\n",
			target, yAct,
			warnStyle.Render(fmt.Sprintf("%+d", erro)),
		))
	}

	if s >= 5 {
		sb.WriteString(fmt.Sprintf("  δ = %.2f × (%+d) = %s\n",
			ALFA, target-yAct,
			warnStyle.Render(fmt.Sprintf("%+.4f", delta)),
		))
	}

	if s >= 6 {
		sb.WriteString("\n" + labelStyle.Bold(true).Render("  Δ pesos:") + "\n")
		for i := 0; i < minInt(6, N_ENTRADAS); i++ {
			if s < i+6 {
				break
			}
			dw := delta * float64(entrada[i])
			wNovo := u.pesos[i] + dw
			sb.WriteString(fmt.Sprintf("  w%-2d: %s %s %s = %s\n",
				i,
				weightStyle.Render(fmt.Sprintf("%+.4f", u.pesos[i])),
				warnStyle.Render(fmt.Sprintf("%+.4f", dw)),
				dimStyle.Render("→"),
				infoStyle.Render(fmt.Sprintf("%+.4f", wNovo)),
			))
		}
	}

	if s >= 9 {
		sb.WriteString("\n")
		db := delta
		sb.WriteString(fmt.Sprintf("  bias: %s %s → %s\n",
			weightStyle.Render(fmt.Sprintf("%+.4f", u.bias)),
			warnStyle.Render(fmt.Sprintf("%+.4f", db)),
			infoStyle.Render(fmt.Sprintf("%+.4f", u.bias+db)),
		))
	}

	return sb.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide 6: Saída — argmax e decisão final
// ─────────────────────────────────────────────────────────────────────────────
func slideSaida(s int, step *TrainingStep) string {
	var sb strings.Builder

	title := lipgloss.NewStyle().Bold(true).Foreground(darkBg).Background(neonGreen).
		Padding(0, 4).Render("  SLIDE 7 — CAMADA DE SAÍDA (ARGMAX)  ")
	sb.WriteString(title + "\n\n")

	if step == nil || s < 1 {
		sb.WriteString(dimStyle.Render("(aguardando dados...)"))
		return sb.String()
	}

	left := buildSaidaBarras(s, step)
	right := buildSaidaDecisao(s, step)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
		thinBox.Render(left), "    ", thinBox.Render(right),
	))

	return sb.String()
}

func buildSaidaBarras(s int, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("y_in de cada ADALINE") + "\n\n")

	if s < 1 {
		return sb.String()
	}

	// Encontra max abs para normalizar
	maxAbs := 0.01
	for _, v := range step.yIn {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}

	best := argmax(step.yIn[:])

	for j := 0; j < N_LETRAS; j++ {
		if s < j+1 {
			break
		}
		yIn := step.yIn[j]

		// Barra proporcional centrada em 0
		barNeg := 10
		barPos := 10
		if yIn > 0 {
			barPos = int(yIn / maxAbs * 10)
			if barPos < 1 {
				barPos = 1
			}
		} else {
			barNeg = int(-yIn / maxAbs * 10)
			if barNeg < 1 {
				barNeg = 1
			}
		}

		negBar := strings.Repeat("◀", barNeg)
		posBar := strings.Repeat("▶", barPos)

		var letterStyle lipgloss.Style
		var negStyle, posStyle lipgloss.Style
		if j == best {
			letterStyle = successStyle
			negStyle = dimStyle
			posStyle = successStyle
		} else if step.erros[j] {
			letterStyle = infoStyle
			negStyle = dimStyle
			posStyle = infoStyle
		} else {
			letterStyle = midStyle
			negStyle = dimStyle
			posStyle = dimStyle
		}

		if yIn < 0 {
			sb.WriteString(fmt.Sprintf("  %s %s%s│%s  %s\n",
				letterStyle.Render(fmt.Sprintf("%-2s", nomesLetras[j])),
				dimStyle.Render(strings.Repeat(" ", 10-barNeg)),
				negStyle.Render(negBar),
				dimStyle.Render(strings.Repeat(" ", 10)),
				weightStyle.Render(fmt.Sprintf("%+.3f", yIn)),
			))
		} else {
			sb.WriteString(fmt.Sprintf("  %s %s│%s%s  %s\n",
				letterStyle.Render(fmt.Sprintf("%-2s", nomesLetras[j])),
				dimStyle.Render(strings.Repeat(" ", 10)),
				posStyle.Render(posBar),
				dimStyle.Render(strings.Repeat(" ", 10-barPos)),
				weightStyle.Render(fmt.Sprintf("%+.3f", yIn)),
			))
		}
	}

	return sb.String()
}

func buildSaidaDecisao(s int, step *TrainingStep) string {
	var sb strings.Builder

	sb.WriteString(boldWhite.Render("Decisão final") + "\n\n")

	best := argmax(step.yIn[:])

	if s >= 2 {
		sb.WriteString(labelStyle.Bold(true).Render("argmax(y_in):") + "\n\n")
		sb.WriteString(fmt.Sprintf("  → maior y_in = %s\n",
			weightStyle.Render(fmt.Sprintf("%+.4f", step.yIn[best])),
		))
		sb.WriteString(fmt.Sprintf("  → unidade %s\n\n",
			successStyle.Render(fmt.Sprintf("ADALINE-%s", nomesLetras[best])),
		))
	}

	if s >= 4 {
		sb.WriteString(boldWhite.Render("  Saída da rede:") + "\n\n")
		sb.WriteString("  " + lipgloss.NewStyle().
			Border(lipgloss.ThickBorder()).
			BorderForeground(neonGreen).
			Foreground(neonGreen).Bold(true).
			Padding(1, 4).
			Render(nomesLetras[best]) + "\n\n")
	}

	if s >= 6 {
		target := step.letraIdx
		isCorrect := best == target
		if isCorrect {
			sb.WriteString(successStyle.Render("  ✓ Correto!\n"))
			sb.WriteString(dimStyle.Render(fmt.Sprintf("  Reconheceu %s = %s\n",
				nomesLetras[step.letraIdx],
				nomesLetras[best],
			)))
		} else {
			sb.WriteString(errorStyle.Render(fmt.Sprintf("  ✗ Errou! Esperado: %s\n",
				nomesLetras[step.letraIdx],
			)))
			sb.WriteString(warnStyle.Render("  → Regra Delta corrigirá\n"))
			sb.WriteString(warnStyle.Render("    os pesos no próximo step"))
		}
	}

	if s >= 8 {
		sb.WriteString("\n\n")
		sb.WriteString(dimStyle.Render("  Após convergência:\n"))
		sb.WriteString(dimStyle.Render("  cada letra tem sua ADALINE\n"))
		sb.WriteString(dimStyle.Render("  com o maior y_in para\n"))
		sb.WriteString(dimStyle.Render("  seu padrão específico."))
	}

	return sb.String()
}

// =============================================================================
// Render: Desenho de Letra
// =============================================================================

func renderDrawLetter(m model) string {
	var sb strings.Builder

	sb.WriteString(labelStyle.Bold(true).Render("Desenhe uma letra") + "\n\n")

	cursorStyle := lipgloss.NewStyle().Background(neonCyan).Foreground(darkBg).Bold(true)
	cursorActiveStyle := lipgloss.NewStyle().Background(neonYellow).Foreground(darkBg).Bold(true)
	pixelOn := lipgloss.NewStyle().Foreground(neonPink)
	pixelOff := dimStyle

	for i := 0; i < N_LINHAS; i++ {
		sb.WriteString("  ")
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			idx := i*N_COLUNAS + j
			isCursor := (i == m.cursorLin && j == m.cursorCol)
			isOn := m.grade[idx] == 1

			switch {
			case isCursor && isOn:
				sb.WriteString(cursorActiveStyle.Render("█"))
			case isCursor:
				sb.WriteString(cursorStyle.Render("·"))
			case isOn:
				sb.WriteString(pixelOn.Render("█"))
			default:
				sb.WriteString(pixelOff.Render("·"))
			}
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  ↑↓←→ mover  •  Espaço toggle  •  Click/Drag pintar  •  R reset  •  Enter reconhecer  •  Esc menu"))
	return sb.String()
}

// =============================================================================
// Render: Reconhecimento
// =============================================================================

func renderRecognition(m model) string {
	var sb strings.Builder

	sb.WriteString(labelStyle.Bold(true).Render("Resultado do Reconhecimento") + "\n\n")

	dataset := letrasDataset()
	letraRec := dataset[m.letraReconhecida]

	leftBox := boxStyle.Render(
		boldWhite.Render("Sua entrada") + "\n\n" + formataLetraGrid(m.grade),
	)
	rightBox := boxStyle.Render(
		successStyle.Render("Reconhecida: "+nomesLetras[m.letraReconhecida]) + "\n\n" + formataLetraGrid(letraRec),
	)

	sb.WriteString(lipgloss.JoinHorizontal(lipgloss.Top, leftBox, "    ", rightBox))
	sb.WriteString("\n\n")

	sb.WriteString(labelStyle.Bold(true).Render("Ativações y_in:") + "\n\n")

	maxAbs := 0.01
	for _, v := range m.yInsReconhecidos {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}

	for j := 0; j < N_LETRAS; j++ {
		barLen := int((m.yInsReconhecidos[j]+maxAbs) / (2 * maxAbs) * 20)
		if barLen < 0 {
			barLen = 0
		}
		if barLen > 20 {
			barLen = 20
		}
		bar := fmt.Sprintf("%-20s", strings.Repeat("█", barLen))
		yStr := fmt.Sprintf("%+7.3f", m.yInsReconhecidos[j])

		if j == m.letraReconhecida {
			sb.WriteString(fmt.Sprintf("  %s %s %s  %s\n",
				successStyle.Render(nomesLetras[j]),
				successStyle.Render(bar),
				weightStyle.Render(yStr),
				successStyle.Render("← VENCEDOR"),
			))
		} else {
			sb.WriteString(fmt.Sprintf("  %s %s %s\n",
				midStyle.Render(nomesLetras[j]),
				dimStyle.Render(bar),
				dimStyle.Render(yStr),
			))
		}
	}

	sb.WriteString("\n")
	sb.WriteString(hintStyle.Render("  Enter → desenhar novamente  •  Esc → menu"))
	return sb.String()
}

// =============================================================================
// Utilitários
// =============================================================================

func argmax(v []float64) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// =============================================================================
// iniciarTUI
// =============================================================================

func iniciarTUI() {
	p := tea.NewProgram(initialModel(),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)
	if _, err := p.Run(); err != nil {
		fmt.Printf("Erro: %v\n", err)
		os.Exit(1)
	}
}
