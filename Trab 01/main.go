package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// =============================================================================
// TRABALHO 01 - Regra de Hebb: Teste para Portas Lógicas
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// O neurônio artificial recebe entradas, pondera pelos pesos e passa pela
// função de ativação para produzir uma saída.
//
// CÁLCULO DO POTENCIAL DE ATIVAÇÃO:
//   y_in = (x1 * w1) + (x2 * w2) + bias
//
// REGRA DE HEBB (atualização dos pesos a cada amostra):
//   w1   = w1   + (x1 * y)   // y = saída desejada (target)
//   w2   = w2   + (x2 * y)
//   bias = bias + y           // bias tem entrada fixa = 1
//
// FUNÇÃO DE ATIVAÇÃO - Degrau Bipolar:
//   soma >= 0  →  saída =  1
//   soma <  0  →  saída = -1
//
// Por que bipolar (-1 e +1) em vez de binário (0 e 1)?
//   Com valor 0 na entrada, o produto x * y seria sempre 0
//   e os pesos nunca seriam atualizados para aquela amostra.
//   O bipolar garante que toda amostra contribui para o aprendizado.
// =============================================================================

// ativacao aplica o degrau bipolar ao potencial calculado.
// Decide se o neurônio "dispara" (1) ou "inibe" (-1).
func ativacao(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

// separando inputs (entradas) e targets (saídas desejadas)
type Porta struct {
	nome      string
	descricao string
	inputs    [4][2]int
	targets   [4]int
}

// todasAsPortas == tabelas verdade de cada porta
// AND, OR, NAND e NOR são linearmente separáveis → a rede aprende.
// XOR não é linearmente separável → a rede não converge
func todasAsPortas() []Porta {
	return []Porta{
		{
			nome:      "AND",
			descricao: "Retorna 1 apenas quando ambas entradas sao 1",
			inputs:    [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets:   [4]int{1, -1, -1, -1},
		},
		{
			nome:      "OR",
			descricao: "Retorna 1 quando pelo menos uma entrada e 1",
			inputs:    [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets:   [4]int{1, 1, 1, -1},
		},
		{
			nome:      "NAND",
			descricao: "Negacao do AND - retorna -1 apenas quando ambas sao 1",
			inputs:    [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets:   [4]int{-1, 1, 1, 1},
		},
		{
			nome:      "NOR",
			descricao: "Negacao do OR - retorna 1 apenas quando ambas sao -1",
			inputs:    [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets:   [4]int{-1, -1, -1, 1},
		},
		{
			nome:      "XOR",
			descricao: "Retorna 1 quando as entradas sao diferentes (nao converge com Hebb simples)",
			inputs:    [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			targets:   [4]int{-1, 1, 1, -1},
		},
	}
}

// Treinamento + teste da porta lógica,
func executarPorta(p Porta) {
	limpar()
	cabecalho()
	fmt.Println(sep())
	fmt.Printf("  PORTA: %s\n", p.nome)
	fmt.Printf("  %s\n", p.descricao)
	fmt.Println(sep())
	fmt.Println()

	// Exibe a tabela verdade da porta antes do treinamento
	fmt.Println("Tabela Verdade (bipolar):")
	fmt.Println("  X1  | X2  | Saida")
	fmt.Println("  ----|-----|------")
	for i := 0; i < 4; i++ {
		fmt.Printf("  %3d | %3d | %3d\n", p.inputs[i][0], p.inputs[i][1], p.targets[i])
	}
	fmt.Println()

	// --- TREINAMENTO ---
	// Pesos iniciam em zero. A cada amostra aplicamos a Regra de Hebb:
	// o peso cresce quando entrada e saída têm o mesmo sinal,
	// e decresce quando têm sinais opostos — reforço por correlação.
	fmt.Println("--- Iniciando Treinamento (Regra de Hebb) ---")

	w1, w2, bias := 0.0, 0.0, 0.0

	for i := 0; i < 4; i++ {
		x1 := p.inputs[i][0]
		x2 := p.inputs[i][1]
		y  := p.targets[i]

		w1   = w1   + float64(x1*y)
		w2   = w2   + float64(x2*y)
		bias = bias + float64(y)

		fmt.Printf("Amostra %d: Pesos atualizados -> W1: %.1f, W2: %.1f, Bias: %.1f\n",
			i+1, w1, w2, bias)
	}

	fmt.Println()
	fmt.Printf("Pesos Finais: W1=%.1f, W2=%.1f, Bias=%.1f\n", w1, w2, bias)

	// --- TESTE ---
	// Usa os pesos aprendidos para calcular y_in de cada amostra.
	// A função de ativação converte y_in em -1 ou 1 para comparar com o alvo.
	fmt.Println()
	fmt.Println("--- Teste Final ---")
	fmt.Println(strings.Repeat("-", 50))

	acertos := 0
	// for i := range 4 {
	for i := 0; i < 4; i++ {
		x1 := p.inputs[i][0]
		x2 := p.inputs[i][1]
		y  := p.targets[i]

		soma     := float64(x1)*w1 + float64(x2)*w2 + bias
		predicao := ativacao(soma)

		status := "✓"
		if predicao == y {
			acertos++
		} else {
			status = "✗"
		}

		fmt.Printf("Entrada: [%2d, %2d] | Alvo: %2d | Predicao: %2d %s\n",
			x1, x2, y, predicao, status)
	}

	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("Acuracia: %d/4 (%.0f%%)\n", acertos, float64(acertos)/4*100)
	fmt.Println()
	aguardar()
}

func executarTodas() {
	for _, p := range todasAsPortas() {
		executarPorta(p)
	}
}

// --- UTILITÁRIOS DE TELA ---

func sep() string { return strings.Repeat("=", 60) }

func cabecalho() {
	fmt.Println(sep())
	fmt.Println("  TRABALHO 01 - Regra de Hebb: Portas Logicas")
	fmt.Println(sep())
}

func limpar() {
	fmt.Print(strings.Repeat("\n", 4))
}

func aguardar() {
	fmt.Print("  Pressione ENTER para voltar ao menu...")
	bufio.NewReader(os.Stdin).ReadString('\n')
}

func lerOpcao() string {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("  Opcao: ")
	entrada, _ := reader.ReadString('\n')
	return strings.TrimSpace(entrada)
}

// --- MENU PRINCIPAL ---

func menu() {
	portas := todasAsPortas()

	for {
		limpar()
		cabecalho()
		fmt.Println()
		fmt.Println("  Selecione uma opcao:")
		fmt.Println()

		for i, p := range portas {
			fmt.Printf("  [%d] Porta %s\n", i+1, p.nome)
		}

		fmt.Println()
		fmt.Printf("  [%d] Executar TODAS as portas\n", len(portas)+1)
		fmt.Println("  [0] Sair")
		fmt.Println()

		opcao := lerOpcao()

		switch opcao {
		case "0":
			limpar()
			fmt.Println("  Encerrando. Ate mais!")
			fmt.Println()
			return
		default:
			encontrou := false
			for i, p := range portas {
				if opcao == fmt.Sprintf("%d", i+1) {
					executarPorta(p)
					encontrou = true
					break
				}
			}
			if !encontrou && opcao == fmt.Sprintf("%d", len(portas)+1) {
				executarTodas()
				encontrou = true
			}
			if !encontrou {
				fmt.Println()
				fmt.Println("  Opcao invalida. Tente novamente.")
				aguardar()
			}
		}
	}
}

func main() {
	menu()
}