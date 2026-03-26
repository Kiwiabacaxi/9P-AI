package mlport

import "math/rand"

// =============================================================================
// Dataset de letras A-Z (grade 5x7 bipolar) + Vetores Bipolares Ortogonais
//
// O dataset de letras eh o mesmo usado no MLP Letras (package letras),
// mas os targets sao vetores bipolares ortogonais em vez de one-hot.
//
// Vetores bipolares ortogonais (Fausett 1994, Manzan 2016):
// - Comecamos com 2 vetores base de 2 dimensoes: [1,1] e [1,-1]
// - Expansao recursiva: cada vetor v gera (v,v) e (v,-v)
// - 2 vetores -> 4 vetores de 4 dims -> 8 de 8 dims -> 16 de 16 dims -> 32 de 32 dims
// - Para 26 letras, usamos os primeiros 26 de 32 vetores ortogonais
// =============================================================================

const NLinhas = 7
const NColunas = 5
const NIn = 35  // 5*7 pixels
const NOrt = 32 // dimensao dos vetores ortogonais (2^5)
const NClasses = 26

var Nomes = [NClasses]string{
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
	"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
}

// Dataset retorna as 26 letras como vetores de 35 floats bipolares.
func Dataset() [NClasses][NIn]float64 {
	grade := [NClasses][NLinhas][NColunas]int{
		// A
		{{-1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// B
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}},
		// C
		{{-1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {-1, 1, 1, 1, 1}},
		// D
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}},
		// E
		{{1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}},
		// F
		{{1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}},
		// G
		{{-1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, 1}},
		// H
		{{1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// I
		{{1, 1, 1, 1, 1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {1, 1, 1, 1, 1}},
		// J
		{{-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, -1}},
		// K
		{{1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1, 1, -1, -1}, {1, 1, -1, -1, -1}, {1, -1, 1, -1, -1}, {1, -1, -1, 1, -1}, {1, -1, -1, -1, 1}},
		// L
		{{1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}},
		// M
		{{1, -1, -1, -1, 1}, {1, 1, -1, 1, 1}, {1, -1, 1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// N
		{{1, -1, -1, -1, 1}, {1, 1, -1, -1, 1}, {1, -1, 1, -1, 1}, {1, -1, -1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// O
		{{-1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, -1}},
		// P
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}},
		// Q
		{{-1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, 1, -1, 1}, {1, -1, -1, 1, 1}, {-1, 1, 1, 1, -1}},
		// R
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}, {1, -1, 1, -1, -1}, {1, -1, -1, 1, -1}, {1, -1, -1, -1, 1}},
		// S
		{{-1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {-1, 1, 1, 1, -1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}},
		// T
		{{1, 1, 1, 1, 1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}},
		// U
		{{1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, -1}},
		// V
		{{1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, 1, -1, 1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}},
		// W
		{{1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, 1, -1, 1}, {1, -1, 1, -1, 1}, {1, 1, -1, 1, 1}, {1, 1, -1, 1, 1}, {-1, 1, -1, 1, -1}},
		// X
		{{1, -1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, 1, -1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// Y
		{{1, -1, -1, -1, 1}, {-1, 1, -1, 1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}},
		// Z
		{{1, 1, 1, 1, 1}, {-1, -1, -1, 1, -1}, {-1, -1, 1, -1, -1}, {-1, 1, -1, -1, -1}, {-1, 1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}},
	}

	var dataset [NClasses][NIn]float64
	for l := 0; l < NClasses; l++ {
		idx := 0
		for i := 0; i < NLinhas; i++ {
			for j := 0; j < NColunas; j++ {
				dataset[l][idx] = float64(grade[l][i][j])
				idx++
			}
		}
	}
	return dataset
}

// GerarVetoresOrtogonais gera 32 vetores bipolares ortogonais de 32 dimensoes.
//
// Algoritmo (Fausett 1994):
// 1. Comecar com 2 vetores base de 2 dims: a=[1,1], b=[1,-1]
// 2. Para cada vetor v, gerar 2 novos: (v,v) e (v,-v) — concatenacao
// 3. Repetir ate ter 2^5 = 32 vetores de 32 dimensoes
//
// Exemplo do slide:
//   Base: (a)=[1,1] (b)=[1,-1]
//   Passo 1: (a,a)=[1,1,1,1] (a,-a)=[1,1,-1,-1] (b,b)=[1,-1,1,-1] (b,-b)=[1,-1,-1,1]
//   Passo 2: 8 vetores de 8 dimensoes
//   ...
//   Passo 4: 32 vetores de 32 dimensoes
func GerarVetoresOrtogonais() [NOrt][NOrt]float64 {
	// Comecar com 2 vetores base
	vetores := [][]float64{
		{1, 1},
		{1, -1},
	}

	// Expandir 4 vezes: 2 -> 4 -> 8 -> 16 -> 32
	for passo := 0; passo < 4; passo++ {
		novos := make([][]float64, 0, len(vetores)*2)
		for _, v := range vetores {
			// (v, v) — concatena v consigo mesmo
			vv := make([]float64, len(v)*2)
			copy(vv[:len(v)], v)
			copy(vv[len(v):], v)
			novos = append(novos, vv)

			// (v, -v) — concatena v com -v
			vmv := make([]float64, len(v)*2)
			copy(vmv[:len(v)], v)
			for i, val := range v {
				vmv[len(v)+i] = -val
			}
			novos = append(novos, vmv)
		}
		vetores = novos
	}

	// Converter para array fixo
	var result [NOrt][NOrt]float64
	for i := 0; i < NOrt; i++ {
		for j := 0; j < NOrt; j++ {
			result[i][j] = vetores[i][j]
		}
	}
	return result
}

// Inicializar cria a rede com pesos aleatorios em [-0.5, +0.5]
func Inicializar(nHid int) OrtMLP {
	rng := rand.New(rand.NewSource(42))
	m := OrtMLP{
		nHid: nHid,
		V:    make([][]float64, NIn),
		V0:   make([]float64, nHid),
		W:    make([][]float64, nHid),
		W0:   make([]float64, NOrt),
	}
	for i := 0; i < NIn; i++ {
		m.V[i] = make([]float64, nHid)
		for j := 0; j < nHid; j++ {
			m.V[i][j] = rng.Float64() - 0.5
		}
	}
	for j := 0; j < nHid; j++ {
		m.V0[j] = rng.Float64() - 0.5
		m.W[j] = make([]float64, NOrt)
		for k := 0; k < NOrt; k++ {
			m.W[j][k] = rng.Float64() - 0.5
		}
	}
	for k := 0; k < NOrt; k++ {
		m.W0[k] = rng.Float64() - 0.5
	}
	return m
}
