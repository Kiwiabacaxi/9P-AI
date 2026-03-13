package main

import (
	"math/rand"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// TRABALHO 03 - PARTE 01: MADALINE para Reconhecimento de Letras (A–M)
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// ARQUITETURA MADALINE:
//   - Entradas: grade 5×7 = 35 pixels em bipolar (-1/+1)
//   - Camada 1: 13 unidades ADALINE (uma por letra)
//   - Camada 2: saída única = argmax dos y_in (vencedor)
//   - Codificação: One-of-N — letra correta = +1, todas as outras = -1
//   - Regra de aprendizado: Regra Delta — w += α*(t-y_in)*x (atualiza em erro)
//   - Ativação: degrau bipolar (>= 0 → +1, < 0 → -1)
//   - Pesos iniciais: rand [-0.5, +0.5]
//   - α = 0.01, MAX_CICLOS = 10000
// =============================================================================

const (
	N_LINHAS   = 7
	N_COLUNAS  = 5
	N_ENTRADAS = 35
	N_LETRAS   = 13
	ALFA       = 0.01
	MAX_CICLOS = 10000
)

var nomesLetras = [N_LETRAS]string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"}

type ADALINEUnit struct {
	pesos [N_ENTRADAS]float64
	bias  float64
}

type MADALINE struct {
	unidades [N_LETRAS]ADALINEUnit
}

// TrainingStep: estado da rede após processar UMA letra em UM ciclo
type TrainingStep struct {
	ciclo    int
	letraIdx int
	yIn      [N_LETRAS]float64
	y        [N_LETRAS]int
	target   [N_LETRAS]int
	erros    [N_LETRAS]bool
}

type ResultadoTreino struct {
	convergiu bool
	ciclos    int
	steps     []TrainingStep
	rede      MADALINE
}

// letrasDataset retorna os 13 padrões 5×7 em bipolar.
// Grade: 5 colunas × 7 linhas → 35 pixels por letra.
func letrasDataset() [N_LETRAS][N_ENTRADAS]int {
	grade := [N_LETRAS][N_LINHAS][N_COLUNAS]int{
		// A: . █ █ █ .
		{
			{-1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// B: █ █ █ █ .
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
		},
		// C: . █ █ █ █
		{
			{-1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{-1, 1, 1, 1, 1},
		},
		// D: █ █ █ █ .
		{
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, -1},
		},
		// E: █ █ █ █ █
		{
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// F: █ █ █ █ █
		{
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
		},
		// G: . █ █ █ █
		{
			{-1, 1, 1, 1, 1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, 1},
		},
		// H: █ . . . █
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, 1, 1, 1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
		// I: █ █ █ █ █
		{
			{1, 1, 1, 1, 1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{-1, -1, 1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// J: . . . . █
		{
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{-1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{-1, 1, 1, 1, -1},
		},
		// K: █ . . . █
		{
			{1, -1, -1, -1, 1},
			{1, -1, -1, 1, -1},
			{1, -1, 1, -1, -1},
			{1, 1, -1, -1, -1},
			{1, -1, 1, -1, -1},
			{1, -1, -1, 1, -1},
			{1, -1, -1, -1, 1},
		},
		// L: █ . . . .
		{
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, -1, -1, -1, -1},
			{1, 1, 1, 1, 1},
		},
		// M: █ . . . █
		{
			{1, -1, -1, -1, 1},
			{1, 1, -1, 1, 1},
			{1, -1, 1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
			{1, -1, -1, -1, 1},
		},
	}

	var dataset [N_LETRAS][N_ENTRADAS]int
	for l := 0; l < N_LETRAS; l++ {
		idx := 0
		for i := 0; i < N_LINHAS; i++ {
			for j := 0; j < N_COLUNAS; j++ {
				dataset[l][idx] = grade[l][i][j]
				idx++
			}
		}
	}
	return dataset
}

// targetVetor retorna o vetor one-of-N: +1 na posição idx, -1 demais.
func targetVetor(idx int) [N_LETRAS]int {
	var t [N_LETRAS]int
	for i := range t {
		t[i] = -1
	}
	t[idx] = 1
	return t
}

// bipolarStep aplica o degrau bipolar.
func bipolarStep(x float64) int {
	if x >= 0 {
		return 1
	}
	return -1
}

// calcularYIn calcula a soma ponderada + bias de uma unidade ADALINE.
func calcularYIn(u ADALINEUnit, entrada [N_ENTRADAS]int) float64 {
	soma := u.bias
	for i := 0; i < N_ENTRADAS; i++ {
		soma += u.pesos[i] * float64(entrada[i])
	}
	return soma
}

// treinarMADALINE executa o loop completo de treinamento.
// Salva apenas steps onde alguma ADALINE foi corrigida.
func treinarMADALINE() ResultadoTreino {
	var rede MADALINE

	// Inicialização aleatória [-0.5, +0.5]
	for j := 0; j < N_LETRAS; j++ {
		for i := 0; i < N_ENTRADAS; i++ {
			rede.unidades[j].pesos[i] = rand.Float64() - 0.5
		}
		rede.unidades[j].bias = rand.Float64() - 0.5
	}

	dataset := letrasDataset()
	var steps []TrainingStep
	convergiu := false
	ciclosReais := MAX_CICLOS

	for ciclo := 1; ciclo <= MAX_CICLOS; ciclo++ {
		erroNoCiclo := false

		for letraIdx := 0; letraIdx < N_LETRAS; letraIdx++ {
			entrada := dataset[letraIdx]
			target := targetVetor(letraIdx)

			var yIn [N_LETRAS]float64
			var y [N_LETRAS]int
			var erros [N_LETRAS]bool

			for j := 0; j < N_LETRAS; j++ {
				yIn[j] = calcularYIn(rede.unidades[j], entrada)
				y[j] = bipolarStep(yIn[j])

				if y[j] != target[j] {
					delta := ALFA * float64(target[j]-y[j])
					for i := 0; i < N_ENTRADAS; i++ {
						rede.unidades[j].pesos[i] += delta * float64(entrada[i])
					}
					rede.unidades[j].bias += delta
					erros[j] = true
					erroNoCiclo = true
				}
			}

			// Salvar apenas steps com erros para economizar memória
			hasErro := false
			for _, e := range erros {
				if e {
					hasErro = true
					break
				}
			}
			if hasErro {
				steps = append(steps, TrainingStep{
					ciclo:    ciclo,
					letraIdx: letraIdx,
					yIn:      yIn,
					y:        y,
					target:   target,
					erros:    erros,
				})
			}
		}

		if !erroNoCiclo {
			convergiu = true
			ciclosReais = ciclo
			break
		}
	}

	return ResultadoTreino{
		convergiu: convergiu,
		ciclos:    ciclosReais,
		steps:     steps,
		rede:      rede,
	}
}

// reconhecer aplica a rede treinada em uma entrada e retorna
// o índice da letra reconhecida + todos os y_in.
func reconhecer(rede MADALINE, entrada [N_ENTRADAS]int) (int, [N_LETRAS]float64) {
	var yIns [N_LETRAS]float64
	for j := 0; j < N_LETRAS; j++ {
		yIns[j] = calcularYIn(rede.unidades[j], entrada)
	}

	// argmax
	best := 0
	for j := 1; j < N_LETRAS; j++ {
		if yIns[j] > yIns[best] {
			best = j
		}
	}
	return best, yIns
}

// formataLetraGrid retorna a representação visual 5×7 de uma entrada.
func formataLetraGrid(entrada [N_ENTRADAS]int) string {
	pixelAtivo := lipgloss.NewStyle().Foreground(lipgloss.Color("#FF6EC7")).Render("█")
	pixelInativo := lipgloss.NewStyle().Foreground(lipgloss.Color("#555555")).Render("·")

	var sb strings.Builder
	for i := 0; i < N_LINHAS; i++ {
		for j := 0; j < N_COLUNAS; j++ {
			if j > 0 {
				sb.WriteString(" ")
			}
			if entrada[i*N_COLUNAS+j] == 1 {
				sb.WriteString(pixelAtivo)
			} else {
				sb.WriteString(pixelInativo)
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	iniciarTUI()
}
