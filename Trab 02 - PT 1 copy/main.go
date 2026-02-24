package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// =============================================================================
// TRABALHO 02 - Parte 01: Perceptron para Reconhecimento de Letras A e B
// Disciplina: Redes Neurais Artificiais
// =============================================================================
//
// Adaptação do Trabalho 01 (Regra de Hebb) para usar o Perceptron.
//
// DIFERENÇA PRINCIPAL: Hebb vs Perceptron
//
//   HEBB (Trabalho 01):
//     Atualiza os pesos para TODA amostra, sempre:
//       w_i = w_i + (x_i * target)
//
//   PERCEPTRON (Trabalho 02):
//     Atualiza os pesos APENAS quando a predição está errada:
//       y_in  = soma(x_i * w_i) + bias
//       y     = ativacao(y_in)
//       SE y != target:
//         w_i  = w_i  + alfa * (target - y) * x_i
//         bias = bias + alfa * (target - y)
//
//     O treinamento repete por CICLOS até que nenhum erro ocorra.
//
// ENTRADAS: Letras A e B representadas em matrizes 7x7 (49 pixels).
//   Pixel "ligado"  =  1  (parte da letra)
//   Pixel "apagado" = -1  (fundo)
//
// SAÍDAS (bipolar):
//   A → target = -1
//   B → target =  1
//
// PARÂMETROS (seguindo o código C do Prof. Jefferson):
//   Pesos iniciais: aleatórios no intervalo [-0.5, +0.5]
//   Bias inicial:    aleatório em [-0.5, +0.5]
//   Taxa de aprendizagem (alfa): 0.01
//   Limiar de ativação: 0 (degrau bipolar)
// =============================================================================

const ALFA = 0.01
const LINHAS = 7
const COLUNAS = 7
const N_ENTRADAS = LINHAS * COLUNAS // 49 pixels

// ativacao aplica o degrau bipolar.
//   y_in >= 0 → saída =  1  (reconheceu como B)
//   y_in <  0 → saída = -1  (reconheceu como A)
func ativacao(soma float64) int {
	if soma >= 0 {
		return 1
	}
	return -1
}

// letraA define a letra A em uma grade 7x7.
// 1  = pixel ativo (parte da letra)
// -1 = pixel inativo (fundo)
//
//   . # # # # # .
//   # . . . . . #
//   # . . . . . #
//   # # # # # # #
//   # . . . . . #
//   # . . . . . #
//   # . . . . . #
func letraA() [N_ENTRADAS]float64 {
	grade := [LINHAS][COLUNAS]int{
		{-1,  1,  1,  1,  1,  1, -1},
		{ 1, -1, -1, -1, -1, -1,  1},
		{ 1, -1, -1, -1, -1, -1,  1},
		{ 1,  1,  1,  1,  1,  1,  1},
		{ 1, -1, -1, -1, -1, -1,  1},
		{ 1, -1, -1, -1, -1, -1,  1},
		{ 1, -1, -1, -1, -1, -1,  1},
	}
	var entrada [N_ENTRADAS]float64
	idx := 0
	for _, linha := range grade {
		for _, pixel := range linha {
			entrada[idx] = float64(pixel)
			idx++
		}
	}
	return entrada
}

// letraB define a letra B em uma grade 7x7.
//
//   # # # # # . .
//   # . . . . # .
//   # . . . . # .
//   # # # # # . .
//   # . . . . # .
//   # . . . . # .
//   # # # # # . .
func letraB() [N_ENTRADAS]float64 {
	grade := [LINHAS][COLUNAS]int{
		{ 1,  1,  1,  1,  1, -1, -1},
		{ 1, -1, -1, -1, -1,  1, -1},
		{ 1, -1, -1, -1, -1,  1, -1},
		{ 1,  1,  1,  1,  1, -1, -1},
		{ 1, -1, -1, -1, -1,  1, -1},
		{ 1, -1, -1, -1, -1,  1, -1},
		{ 1,  1,  1,  1,  1, -1, -1},
	}
	var entrada [N_ENTRADAS]float64
	idx := 0
	for _, linha := range grade {
		for _, pixel := range linha {
			entrada[idx] = float64(pixel)
			idx++
		}
	}
	return entrada
}

// imprimirLetra exibe a grade 7x7 de uma letra no terminal.
// '#' representa pixel ativo, '.' representa fundo.
func imprimirLetra(entrada [N_ENTRADAS]float64) {
	for i := 0; i < LINHAS; i++ {
		fmt.Print("  ")
		for j := 0; j < COLUNAS; j++ {
			if entrada[i*COLUNAS+j] == 1 {
				fmt.Print("# ")
			} else {
				fmt.Print(". ")
			}
		}
		fmt.Println()
	}
}

// treinarPerceptron executa o treinamento da rede Perceptron.
//
// O algoritmo Perceptron (diferença chave em relação a Hebb):
//   1. Calcula y_in = soma(x_i * w_i) + bias
//   2. Aplica ativação: y = degrau(y_in)
//   3. SÓ atualiza pesos se houve ERRO (y != target):
//        delta = alfa * (target - y)
//        w_i   = w_i   + delta * x_i
//        bias  = bias  + delta
//   4. Repete por ciclos até convergência (zero erros em um ciclo completo)
func treinarPerceptron(
	amostras [][N_ENTRADAS]float64,
	targets []int,
	verbose bool,
) ([N_ENTRADAS]float64, float64, int) {

	// Pesos inicializados aleatoriamente em [-0.5, +0.5]
	// (padrão da literatura, conforme código do professor)
	var pesos [N_ENTRADAS]float64
	for i := range pesos {
		pesos[i] = rand.Float64() - 0.5
	}
	bias := rand.Float64() - 0.5

	ciclo := 0
	for {
		ciclo++
		condErro := false // sem erros neste ciclo → convergiu

		for s, entrada := range amostras {
			// Calcula potencial de ativação
			yIn := bias
			for i := 0; i < N_ENTRADAS; i++ {
				yIn += entrada[i] * pesos[i]
			}

			// Aplica função de ativação
			y := ativacao(yIn)

			// Atualiza pesos SOMENTE se errou (diferença do Hebb!)
			if verbose {
				nomeLetra := "B"
				if targets[s] == -1 {
					nomeLetra = "A"
				}
				fmt.Println(strings.Repeat("-", 60))
				fmt.Printf("  Ciclo %d | Amostra: Letra %s\n", ciclo, nomeLetra)
				fmt.Printf("  y_in (potencial) = %.4f\n", yIn)
				fmt.Printf("  y    (ativacao)  = %d\n", y)
				fmt.Printf("  target           = %d\n", targets[s])
			}

			if y != targets[s] {
				condErro = true
				delta := ALFA * float64(targets[s]-y)
				for i := 0; i < N_ENTRADAS; i++ {
					pesos[i] += delta * entrada[i]
				}
				bias += delta

				if verbose {
					fmt.Printf("  Resultado: ERRO  → atualizando pesos\n")
					fmt.Printf("  delta = alfa * (target - y) = %.2f * (%d - %d) = %.4f\n",
						ALFA, targets[s], y, delta)
					fmt.Printf("  Novo bias = %.4f\n", bias)
				}
			} else if verbose {
				fmt.Printf("  Resultado: OK    → pesos mantidos\n")
			}
		}

		// Sem erros neste ciclo → convergência atingida
		if !condErro {
			if verbose {
				fmt.Println(strings.Repeat("=", 60))
				fmt.Printf("  Ciclo %d completo — nenhum erro. Convergência!\n", ciclo)
				fmt.Println(strings.Repeat("=", 60))
			}
			break
		}

		// Proteção contra loop infinito (problema não separável)
		if ciclo >= 10000 {
			fmt.Println("  AVISO: Limite de 10000 ciclos atingido sem convergência.")
			break
		}
	}

	return pesos, bias, ciclo
}

// testarPerceptron usa os pesos treinados para classificar cada amostra.
func testarPerceptron(
	amostras [][N_ENTRADAS]float64,
	targets []int,
	nomes []string,
	pesos [N_ENTRADAS]float64,
	bias float64,
) {
	fmt.Println()
	fmt.Println("--- Teste Final ---")
	fmt.Println(strings.Repeat("-", 50))

	acertos := 0
	for s, entrada := range amostras {
		yIn := bias
		for i := 0; i < N_ENTRADAS; i++ {
			yIn += entrada[i] * pesos[i]
		}

		predicao := ativacao(yIn)
		status := "✓"
		if predicao == targets[s] {
			acertos++
		} else {
			status = "✗"
		}

		nomePredicao := "B"
		if predicao == -1 {
			nomePredicao = "A"
		}

		fmt.Printf("  Letra: %s | Alvo: %2d | Predição: %2d (%s) %s\n",
			nomes[s], targets[s], predicao, nomePredicao, status)
	}

	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("  Acurácia: %d/%d (%.0f%%)\n",
		acertos, len(amostras),
		float64(acertos)/float64(len(amostras))*100)
}

// --- ESTADO GLOBAL (pesos salvos após treinamento) ---

var pesosTreinados [N_ENTRADAS]float64
var biasTreinado float64
var redeTreinada bool

// executarTreinamento treina a rede e salva os pesos para uso na operação.
func executarTreinamento() {
	limpar()
	cabecalho()
	fmt.Println(sep())
	fmt.Println("  TREINANDO A REDE - Letras A e B (Matrizes 7x7)")
	fmt.Println("  Algoritmo: Perceptron  |  Taxa de aprendizagem: 0.01")
	fmt.Println(sep())
	fmt.Println()

	entradaA := letraA()
	entradaB := letraB()

	amostras := [][N_ENTRADAS]float64{entradaA, entradaB}
	targets := []int{-1, 1}

	fmt.Println("Padrões de entrada:")
	fmt.Println()
	fmt.Println("  Letra A (target = -1):")
	imprimirLetra(entradaA)
	fmt.Println()
	fmt.Println("  Letra B (target =  1):")
	imprimirLetra(entradaB)
	fmt.Println()

	fmt.Println("--- Iniciando Treinamento (Perceptron) ---")
	fmt.Println()
	fmt.Println("  Regra de atualização (só quando erra):")
	fmt.Println("    delta   = alfa * (target - y)")
	fmt.Println("    w_i     = w_i + delta * x_i")
	fmt.Println("    bias    = bias + delta")
	fmt.Println()

	pesos, bias, ciclos := treinarPerceptron(amostras, targets, true)

	pesosTreinados = pesos
	biasTreinado = bias
	redeTreinada = true

	fmt.Println()
	fmt.Printf("  Convergência em %d ciclo(s)!\n", ciclos)
	fmt.Printf("  Bias final: %.4f\n", biasTreinado)
	fmt.Println()
	fmt.Println("  Rede treinada! Use a opcao [2] para operar.")
	fmt.Println()
	aguardar()
}

// executarOperacao usa os pesos treinados para classificar A e B.
func executarOperacao() {
	limpar()
	cabecalho()
	fmt.Println(sep())
	fmt.Println("  OPERANDO A REDE - Classificação com Pesos Treinados")
	fmt.Println(sep())
	fmt.Println()

	if !redeTreinada {
		fmt.Println("  ATENÇÃO: A rede ainda não foi treinada!")
		fmt.Println("  Execute a opcao [1] primeiro.")
		fmt.Println()
		aguardar()
		return
	}

	entradaA := letraA()
	entradaB := letraB()

	amostras := [][N_ENTRADAS]float64{entradaA, entradaB}
	targets := []int{-1, 1}
	nomes := []string{"A", "B"}

	fmt.Printf("  Bias utilizado: %.4f\n", biasTreinado)

	testarPerceptron(amostras, targets, nomes, pesosTreinados, biasTreinado)

	fmt.Println()
	aguardar()
}

// --- UTILITÁRIOS (mesmos do Trabalho 01) ---

func sep() string { return strings.Repeat("=", 60) }

func cabecalho() {
	fmt.Println(sep())
	fmt.Println("  TRABALHO 02 - Perceptron: Reconhecimento de Letras")
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

// executarTreinamentoEOperacao treina a rede e em seguida opera sem voltar ao menu.
func executarTreinamentoEOperacao() {
	executarTreinamento()
	executarOperacao()
}

// --- MENU PRINCIPAL ---

func menu() {
	for {
		limpar()
		cabecalho()
		fmt.Println()
		fmt.Println("  Selecione uma opcao:")
		fmt.Println()
		fmt.Println("  [1] Treinar a rede")
		fmt.Println("  [2] Operar")
		fmt.Println("  [3] Treinar e Operar")
		fmt.Println("  [0] Sair")
		fmt.Println()

		opcao := lerOpcao()

		switch opcao {
		case "1":
			executarTreinamento()
		case "2":
			executarOperacao()
		case "3":
			executarTreinamentoEOperacao()
		case "0":
			limpar()
			fmt.Println("  Encerrando. Ate mais!")
			fmt.Println()
			return
		default:
			fmt.Println()
			fmt.Println("  Opcao invalida. Tente novamente.")
			aguardar()
		}
	}
}

func main() {
	menu()
}