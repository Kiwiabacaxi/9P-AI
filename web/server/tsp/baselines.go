package tsp

import (
	"math"
)

// =============================================================================
// Baselines clássicos pra TSP — algoritmos determinísticos pra comparar com o AG.
//
// O ponto pedagógico: o AG não está sozinho. Existem heurísticas conhecidas
// que resolvem o TSP rapidinho e dão bons resultados. Comparar com elas
// mostra quando o AG vale a pena (geometria complicada, muitas cidades) e
// quando uma heurística simples já chega perto.
//
//   1) Nearest Neighbor — greedy puro. Escolhe sempre a cidade mais próxima
//      ainda não visitada. Tipicamente 25% pior que o ótimo, mas é instantâneo.
//
//   2) 2-opt — busca local. Pega um tour qualquer e tenta inverter sub-segmentos
//      (movimentos 2-opt). Se algum movimento reduz a distância, aceita.
//      Repete até não conseguir melhorar. Tipicamente 5–10% pior que o ótimo.
//
//   3) NN + 2-opt — combo clássico. NN gera o tour inicial, 2-opt poliquí.
// =============================================================================

// NearestNeighbor — começa em `inicio` e sempre vai pra cidade mais próxima
// ainda não visitada. Retorna um tour fechado (volta a inicio).
func NearestNeighbor(matriz [][]float64, inicio int) []int {
	n := len(matriz)
	if n == 0 {
		return nil
	}
	if inicio < 0 || inicio >= n {
		inicio = 0
	}

	tour := make([]int, 0, n)
	visitado := make([]bool, n)

	atual := inicio
	tour = append(tour, atual)
	visitado[atual] = true

	for i := 1; i < n; i++ {
		melhor := -1
		melhorDist := math.Inf(1)
		for j := 0; j < n; j++ {
			if visitado[j] {
				continue
			}
			d := matriz[atual][j]
			if d < melhorDist {
				melhorDist = d
				melhor = j
			}
		}
		if melhor == -1 {
			break
		}
		tour = append(tour, melhor)
		visitado[melhor] = true
		atual = melhor
	}
	return tour
}

// TwoOpt — busca local com movimentos 2-opt. Inverte sub-segmentos do tour
// até não haver mais melhoria possível.
//
// Cada "movimento 2-opt": escolher dois índices i < j, reverter tour[i:j+1].
// Se essa reversão produz tour mais curto, aceita. Continua até passada
// completa sem melhoria (convergência local).
//
// Vantagem sobre swap aleatório: a inversão remove "cruzamentos" no tour,
// que são quase sempre o que torna um tour ruim ruim.
func TwoOpt(tour []int, matriz [][]float64) []int {
	n := len(tour)
	if n < 4 {
		return tour
	}
	atual := make([]int, n)
	copy(atual, tour)

	melhorou := true
	for melhorou {
		melhorou = false
		for i := 0; i < n-1; i++ {
			for j := i + 1; j < n; j++ {
				// Calcula delta sem precisar recomputar o tour inteiro.
				// Aresta antes: (i-1)→i  e  j→(j+1)
				// Aresta depois (após inverter [i:j+1]): (i-1)→j  e  i→(j+1)
				prev := (i - 1 + n) % n
				next := (j + 1) % n
				if prev == j || next == i {
					continue // segmentos colados — reverter é noop
				}
				a, b := atual[prev], atual[i]
				c, d := atual[j], atual[next]
				delta := matriz[a][c] + matriz[b][d] - matriz[a][b] - matriz[c][d]
				if delta < -1e-9 {
					// inverte tour[i..j]
					for x, y := i, j; x < y; x, y = x+1, y-1 {
						atual[x], atual[y] = atual[y], atual[x]
					}
					melhorou = true
				}
			}
		}
	}
	return atual
}
