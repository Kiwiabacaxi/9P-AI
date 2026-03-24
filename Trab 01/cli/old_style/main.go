package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// =============================================================================
// REGRA DE HEBB - Teste para Portas Lógicas (Trabalho 01)
// =============================================================================
//
// NEURÔNIO:
//   y_in = (x1*w1) + (x2*w2) + b
//
// FUNÇÃO DE ATIVAÇÃO (degrau bipolar):
//   y_in > 0  →  saída = +1
//   y_in < 0  →  saída = -1
//   y_in == 0 →  saída =  0
//
// REGRA DE HEBB (atualização de pesos a cada amostra):
//   w_novo = w_antigo + (η * x * desejado)
//   b_novo = b_antigo + (η * 1 * desejado)
// =============================================================================

const taxa = 1.0

type Amostra struct {
	x1, x2   float64
	desejado float64
}

func ativacao(yIn float64) float64 {
	if yIn > 0 {
		return 1
	} else if yIn < 0 {
		return -1
	}
	return 0
}

// Treina e testa para uma porta, mostrando tudo na tela
func executarPorta(nome string, descricao string, amostras []Amostra) {
	fmt.Println()
	fmt.Println(sep())
	fmt.Printf("  PORTA: %s \n", nome)
	fmt.Printf("  %s\n", descricao)
	fmt.Println(sep())
	fmt.Println()

	fmt.Println("Tabela Verdade (bipolar):")
	fmt.Println("  X1  | X2  | Saída")
	fmt.Println("  ----|-----|------")
	for _, a := range amostras {
		fmt.Printf("  %3.0f | %3.0f | %3.0f\n", a.x1, a.x2, a.desejado)
	}
	fmt.Println()

	fmt.Println("--- Iniciando Treinamento (Regra de Hebb) ---")

	w1, w2, b := 0.0, 0.0, 0.0

	for i, a := range amostras {
		w1 += taxa * a.x1 * a.desejado
		w2 += taxa * a.x2 * a.desejado
		b += taxa * 1 * a.desejado

		fmt.Printf("Amostra %d: Pesos atualizados -> W1: %.1f, W2: %.1f, Bias: %.1f\n", i+1, w1, w2, b)
	}

	// --- TESTE ---
	fmt.Println()
	fmt.Printf("Pesos Finais: W1=%.1f, W2=%.1f, Bias=%.1f\n", w1, w2, b)
	fmt.Println()
	fmt.Println("--- Teste Final ---")
	fmt.Println(strings.Repeat("-", 50))

	acertos := 0
	for _, a := range amostras {
		yIn := a.x1*w1 + a.x2*w2 + b
		saida := ativacao(yIn)
		ok := "✓"
		if saida == a.desejado {
			acertos++
		} else {
			ok = "✗"
		}
		fmt.Printf("Entrada: [%2.0f, %2.0f] | Alvo: %2.0f | Predição: %2.0f %s\n",
			a.x1, a.x2, a.desejado, saida, ok)
	}

	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("Acurácia: %d/%d (%.0f%%)\n", acertos, len(amostras), float64(acertos)/float64(len(amostras))*100)
	fmt.Println()
	aguardar()
}

// Executa todas as portas em sequência
func executarTodas() {
	limpar()
	cabecalho()
	for _, p := range todasAsPortas() {
		executarPorta(" "+p.nome, p.descricao, p.amostras)
	}
}

// --- DADOS DAS PORTAS (representação bipolar: 0 → -1, 1 → +1) ---

type Porta struct {
	nome      string
	descricao string
	amostras  []Amostra
}

func todasAsPortas() []Porta {
	return []Porta{
		{"AND", "Retorna 1 apenas quando ambas entradas são 1", []Amostra{{-1, -1, -1}, {-1, +1, -1}, {+1, -1, -1}, {+1, +1, +1}}},
		{"OR", "Retorna 1 quando pelo menos uma entrada é 1", []Amostra{{-1, -1, -1}, {-1, +1, +1}, {+1, -1, +1}, {+1, +1, +1}}},
		{"NAND", "Negação do AND - retorna -1 apenas quando ambas são 1", []Amostra{{-1, -1, +1}, {-1, +1, +1}, {+1, -1, +1}, {+1, +1, -1}}},
		{"NOR", "Negação do OR - retorna 1 apenas quando ambas são -1", []Amostra{{-1, -1, +1}, {-1, +1, -1}, {+1, -1, -1}, {+1, +1, -1}}},
		{"XOR", "Retorna 1 apenas quando as entradas são diferentes (Ou Exclusivo)", []Amostra{{-1, -1, -1}, {-1, +1, +1}, {+1, -1, +1}, {+1, +1, -1}}},
	}
}

// --- UTILITÁRIOS ---

func sep() string { return strings.Repeat("=", 60) }

func cabecalho() {
	fmt.Println(sep())
	fmt.Println("  TRABALHO 01 - Regra de Hebb: Portas Lógicas")
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
			// Verifica portas individuais (1 a 5)
			for i, p := range portas {
				if opcao == fmt.Sprintf("%d", i+1) {
					limpar()
					cabecalho()
					executarPorta(" "+p.nome, p.descricao, p.amostras)
					encontrou = true
					break
				}
			}
			// Opção "todas" (6)
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
