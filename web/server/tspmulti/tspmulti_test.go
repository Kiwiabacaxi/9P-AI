package tspmulti

import (
	"math"
	"testing"

	"mlp-server/tsp"
)

// ringMatriz — n cidades num círculo de raio 1 (matriz euclidiana determinística).
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

func baseCfg() MultiConfig {
	cfg := DefaultMultiConfig()
	cfg.NumIlhas = 3
	cfg.TamIlha = 20
	cfg.MaxGeracoes = 50
	cfg.IntervaloMigracao = 10
	cfg.NumMigrantes = 1
	cfg.Seed = 42
	return cfg
}

// Teste 1 + 2: migração em anel move o melhor da ilha i pra (i+1) e substitui
// o pior; e a coleta dos melhores ocorre ANTES das inserções (simultânea).
func TestMigracaoAnelSimultanea(t *testing.T) {
	n := 8
	mat := ringMatriz(n)
	ga := tsp.Sanitizar(tsp.DefaultConfig(), n)

	// Três ilhas pequenas com custos conhecidos: tour identidade.
	mk := func(seedTour []int) tsp.Individuo {
		d, ml, ts, c := tsp.Avaliar(seedTour, mat, nil, ga)
		return tsp.Individuo{Tour: append([]int(nil), seedTour...), Distancia: d, MaxLeg: ml, TempoSec: ts, Custo: c}
	}
	// Cada ilha: 2 indivíduos (melhor + pior distinto). Usamos permutações.
	bom := []int{0, 1, 2, 3, 4, 5, 6, 7}  // tour do círculo em ordem (ótimo) → menor custo
	ruim := []int{0, 4, 1, 5, 2, 6, 3, 7} // ziguezague → custo alto
	ilhas := [][]tsp.Individuo{
		{mk(bom), mk(ruim)},
		{mk(ruim), mk(ruim)},
		{mk(ruim), mk(ruim)},
	}
	numIlhas := 3

	bestTourIlha0 := append([]int(nil), ilhas[0][melhorIdx(ilhas[0])].Tour...)

	// Replica a lógica de migração do orquestrador (1 migrante, anel).
	bests := make([][]tsp.Individuo, numIlhas)
	for i := 0; i < numIlhas; i++ {
		bests[i] = topMigrantes(ilhas[i], 1)
	}
	for i := 0; i < numIlhas; i++ {
		dest := (i + 1) % numIlhas
		substituirPiores(ilhas[dest], bests[i])
	}

	// O melhor da ilha 0 (bom) deve ter migrado pra ilha 1.
	achou := false
	for _, ind := range ilhas[1] {
		if sameTour(ind.Tour, bestTourIlha0) {
			achou = true
			break
		}
	}
	if !achou {
		t.Fatalf("migrante da ilha 0 não chegou na ilha 1")
	}

	// Simultaneidade: a ilha 0 NÃO deve ter recebido o migrante da ilha 1
	// "encadeado" — só o que a ilha 2 mandou (que era 'ruim'), preservando o
	// seu próprio melhor 'bom'. Garante que a ilha 0 ainda contém 'bom'.
	temBom := false
	for _, ind := range ilhas[0] {
		if sameTour(ind.Tour, bestTourIlha0) {
			temBom = true
			break
		}
	}
	if !temBom {
		t.Fatalf("ilha 0 perdeu seu melhor após migração (deveria manter cópia)")
	}
}

func sameTour(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Teste 3: o melhor global nunca piora ao longo das gerações (monotonicidade
// garantida pelo elitismo + tracking de best-so-far).
func TestMonotonicidadeGlobal(t *testing.T) {
	mat := ringMatriz(12)
	res := Treinar(nil, baseCfg(), mat, nil)
	for i := 1; i < len(res.HistGlobal); i++ {
		if res.HistGlobal[i] > res.HistGlobal[i-1]+1e-12 {
			t.Fatalf("global piorou na geração %d: %.6f → %.6f", i, res.HistGlobal[i-1], res.HistGlobal[i])
		}
	}
}

// Teste 4: mesma seed ⇒ mesmo resultado, independente do escalonamento das
// goroutines (cada ilha tem rng próprio).
func TestDeterminismo(t *testing.T) {
	mat := ringMatriz(12)
	r1 := Treinar(nil, baseCfg(), mat, nil)
	r2 := Treinar(nil, baseCfg(), mat, nil)
	if r1.MelhorGlobalDist != r2.MelhorGlobalDist {
		t.Fatalf("não-determinístico: %.9f vs %.9f", r1.MelhorGlobalDist, r2.MelhorGlobalDist)
	}
	if !sameTour(r1.MelhorGlobalTour, r2.MelhorGlobalTour) {
		t.Fatalf("tours globais diferentes entre execuções")
	}
}

// Teste 5: com CompararPopUnica, o baseline tem o mesmo comprimento do histórico
// global e usa NumIlhas × TamIlha indivíduos.
func TestBaselineComparativo(t *testing.T) {
	mat := ringMatriz(12)
	cfg := baseCfg()
	cfg.CompararPopUnica = true
	res := Treinar(nil, cfg, mat, nil)
	if len(res.HistRefUnica) != len(res.HistGlobal) {
		t.Fatalf("HistRefUnica (%d) != HistGlobal (%d)", len(res.HistRefUnica), len(res.HistGlobal))
	}
	if res.MelhorRefUnicaDist <= 0 || math.IsInf(res.MelhorRefUnicaDist, 1) {
		t.Fatalf("MelhorRefUnicaDist inválido: %v", res.MelhorRefUnicaDist)
	}
}

// Teste extra: número de gerações de migração = MaxGeracoes / IntervaloMigracao.
func TestQuantidadeMigracoes(t *testing.T) {
	mat := ringMatriz(10)
	cfg := baseCfg() // 50 ger, intervalo 10 → migra em 10,20,30,40,50 = 5 vezes
	res := Treinar(nil, cfg, mat, nil)
	esperado := cfg.MaxGeracoes / cfg.IntervaloMigracao
	if len(res.GeracoesMigracao) != esperado {
		t.Fatalf("migrações: esperado %d, obtido %d (%v)", esperado, len(res.GeracoesMigracao), res.GeracoesMigracao)
	}
}
