package tsp

import (
	"math"
	"testing"
)

// ringMatriz — n cidades igualmente espaçadas num círculo de raio 1.
// Matriz euclidiana simétrica e determinística, boa pra testar o GA: o tour
// ótimo é simplesmente seguir o círculo na ordem.
func ringMatriz(n int) [][]float64 {
	pts := make([][2]float64, n)
	for i := 0; i < n; i++ {
		ang := 2 * math.Pi * float64(i) / float64(n)
		pts[i] = [2]float64{math.Cos(ang), math.Sin(ang)}
	}
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			dx := pts[i][0] - pts[j][0]
			dy := pts[i][1] - pts[j][1]
			m[i][j] = math.Hypot(dx, dy)
		}
	}
	return m
}

// TestTreinarDeterministico — caracteriza o comportamento do AG de população
// única (Trabalho 11). Serve de regressão: o refactor que extrai
// EvoluirUmaGeracao NÃO pode mudar o resultado com a mesma seed.
func TestTreinarDeterministico(t *testing.T) {
	mat := ringMatriz(8)
	cfg := DefaultConfig()
	cfg.PopSize = 20
	cfg.MaxGeracoes = 40
	cfg.Seed = 42

	r1 := Treinar(nil, cfg, mat, nil)
	r2 := Treinar(nil, cfg, mat, nil)

	if r1.MelhorDist != r2.MelhorDist {
		t.Fatalf("não-determinístico: r1=%v r2=%v", r1.MelhorDist, r2.MelhorDist)
	}

	// Golden — capturado do código pré-refactor. Trava o comportamento.
	const golden = 6.122934917525586
	if math.Abs(r1.MelhorDist-golden) > 1e-9 {
		t.Fatalf("regressão de comportamento: melhorDist=%.15f, golden=%.15f", r1.MelhorDist, golden)
	}
}
