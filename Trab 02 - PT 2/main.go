package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// =============================================================================
// TRABALHO 02 - PARTE 02: Perceptron para Portas Lógicas
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// OBJETIVO: Treinar um Perceptron para cada porta lógica (AND, OR, NAND, NOR, XOR)
// e mostrar os pesos finais e bias após o treinamento.
//
// DIFERENÇA EM RELAÇÃO AO TRABALHO 01 (Hebb):
//   - Hebb atualiza pesos em TODA amostra: w += x * target
//   - Perceptron só atualiza quando ERRA:  w += alfa * (target - y) * x
//   - Perceptron repete ciclos até não errar mais (convergência)
//
// REPRESENTAÇÃO BIPOLAR:
//   Entradas e saídas usam -1 e +1 (não binário 0/1).
//   Com 0 na entrada, x*w = 0 e o peso nunca seria corrigido.
//
// ATENÇÃO COM O XOR:
//   O XOR não é linearmente separável — o Perceptron simples
//   NÃO consegue aprender. O programa detecta isso e avisa o usuário.
// =============================================================================

const (
	ALFA       = 0.01 // taxa de aprendizagem
	MAX_CICLOS = 1000 // limite pra não travar no XOR
)

// Porta agrupa nome, entradas e saídas esperadas de uma porta lógica.
type Porta struct {
	nome    string
	inputs  [4][2]int // 4 amostras com 2 entradas cada
	targets [4]int    // saída esperada pra cada amostra
}

// todasAsPortas retorna as 5 portas lógicas em formato bipolar.
func todasAsPortas() []Porta {
	return []Porta{
		{
			nome:    "AND",
			inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets: [4]int{1, -1, -1, -1},
		},
		{
			nome:    "OR",
			inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets: [4]int{1, 1, 1, -1},
		},
		{
			nome:    "NAND",
			inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets: [4]int{-1, 1, 1, 1},
		},
		{
			nome:    "NOR",
			inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets: [4]int{-1, -1, -1, 1},
		},
		{
			nome:    "XOR",
			inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets: [4]int{-1, 1, 1, -1},
		},
	}
}

// ativacao aplica o degrau bipolar.
//
//	soma >= 0 → 1
//	soma <  0 → -1
func ativacao(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

// stepLog guarda dados de um passo do treinamento pra TUI exibir.
type stepLog struct {
	ciclo    int
	amostra  int // índice 0-3
	x1, x2   int
	target   int
	yLiq     float64
	y        int
	teveErro bool
}

// resultadoTreino guarda o resultado final do treinamento de uma porta.
type resultadoTreino struct {
	porta     Porta
	w1, w2    float64
	bias      float64
	ciclos    int
	convergiu bool
	steps     []stepLog
	testes    []testResult
}

// testResult guarda o teste de uma amostra com os pesos finais.
type testResult struct {
	x1, x2   int
	target   int
	predicao int
	yLiq     float64
	acertou  bool
}

// treinarPorta executa o Perceptron numa porta lógica.
// Retorna todos os detalhes pra TUI renderizar.
func treinarPorta(p Porta) resultadoTreino {
	// Pesos iniciais aleatórios em [-0.5, 0.5] (convenção do professor)
	w1 := rand.Float64() - 0.5
	w2 := rand.Float64() - 0.5
	bias := rand.Float64() - 0.5

	var steps []stepLog
	convergiu := false
	ciclo := 0

	for ciclo < MAX_CICLOS {
		ciclo++
		errou := false

		for i := 0; i < 4; i++ {
			x1 := p.inputs[i][0]
			x2 := p.inputs[i][1]

			// Potencial de ativação: y_in = w1*x1 + w2*x2 + bias
			yLiq := w1*float64(x1) + w2*float64(x2) + bias
			y := ativacao(yLiq)

			teveErro := false

			// Regra do Perceptron: só atualiza se errou
			if y != p.targets[i] {
				errou = true
				teveErro = true
				delta := ALFA * float64(p.targets[i]-y)
				w1 += delta * float64(x1)
				w2 += delta * float64(x2)
				bias += delta
			}

			steps = append(steps, stepLog{
				ciclo: ciclo, amostra: i,
				x1: x1, x2: x2,
				target: p.targets[i],
				yLiq:   yLiq, y: y,
				teveErro: teveErro,
			})
		}

		if !errou {
			convergiu = true
			break
		}
	}

	// Testa com os pesos finais
	var testes []testResult
	acertos := 0
	for i := 0; i < 4; i++ {
		x1, x2 := p.inputs[i][0], p.inputs[i][1]
		yLiq := w1*float64(x1) + w2*float64(x2) + bias
		pred := ativacao(yLiq)
		ok := pred == p.targets[i]
		if ok {
			acertos++
		}
		testes = append(testes, testResult{
			x1: x1, x2: x2, target: p.targets[i],
			predicao: pred, yLiq: yLiq, acertou: ok,
		})
	}

	return resultadoTreino{
		porta: p, w1: w1, w2: w2, bias: bias,
		ciclos: ciclo, convergiu: convergiu,
		steps: steps, testes: testes,
	}
}

// formataTabela renderiza a tabela verdade de uma porta pra TUI.
func formataTabela(p Porta) string {
	var sb strings.Builder
	sb.WriteString(lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#64B5F6")).
		Render("Tabela Verdade (bipolar)"))
	sb.WriteString("\n\n")
	sb.WriteString("  X1  │ X2  │ Saída\n")
	sb.WriteString("  ────┼─────┼──────\n")
	for i := 0; i < 4; i++ {
		sb.WriteString(fmt.Sprintf("  %3d │ %3d │  %3d\n",
			p.inputs[i][0], p.inputs[i][1], p.targets[i]))
	}
	return sb.String()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	iniciarTUI()
}
