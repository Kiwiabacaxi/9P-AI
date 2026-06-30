package tspranking

import (
	"math"
	"math/rand"
	"testing"
)

func aprox(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

func ehPermutacaoValida(t []int, n int) bool {
	if len(t) != n {
		return false
	}
	visto := make([]bool, n)
	for _, c := range t {
		if c < 0 || c >= n || visto[c] {
			return false
		}
		visto[c] = true
	}
	return true
}

// Aula 16 (slide "Passo 05"): N=5, η_max=1.5 → P = [0.30, 0.25, 0.20, 0.15, 0.10].
func TestRankingLinearSlide(t *testing.T) {
	got := ProbsRankingLinear(5, 1.5)
	want := []float64{0.30, 0.25, 0.20, 0.15, 0.10}
	if len(got) != len(want) {
		t.Fatalf("tamanho: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if !aprox(got[i], want[i], 1e-9) {
			t.Errorf("rank %d: got %.6f want %.6f", i+1, got[i], want[i])
		}
	}
}

// Aula 16 (slide exponencial): N=5, c=2 → pesos [16,8,4,2,1], soma 31,
// P ≈ [0.516, 0.258, 0.129, 0.064, 0.032].
func TestRankingExpSlide(t *testing.T) {
	got := ProbsRankingExp(5, 2.0)
	want := []float64{16.0 / 31, 8.0 / 31, 4.0 / 31, 2.0 / 31, 1.0 / 31}
	if len(got) != len(want) {
		t.Fatalf("tamanho: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if !aprox(got[i], want[i], 1e-9) {
			t.Errorf("rank %d: got %.6f want %.6f", i+1, got[i], want[i])
		}
	}
}

func TestRankingProbsSomaUmEMonotonas(t *testing.T) {
	casos := []struct {
		nome  string
		probs []float64
	}{
		{"linear η=1.5", ProbsRankingLinear(20, 1.5)},
		{"linear η=2.0", ProbsRankingLinear(20, 2.0)},
		{"exp c=1.07", ProbsRankingExp(20, 1.07)},
		{"exp c=2.0", ProbsRankingExp(20, 2.0)},
	}
	for _, c := range casos {
		soma := 0.0
		for _, p := range c.probs {
			if p < 0 {
				t.Errorf("%s: probabilidade negativa %.6f", c.nome, p)
			}
			soma += p
		}
		if !aprox(soma, 1.0, 1e-9) {
			t.Errorf("%s: soma %.9f != 1", c.nome, soma)
		}
		for i := 1; i < len(c.probs); i++ {
			if c.probs[i] > c.probs[i-1]+1e-12 {
				t.Errorf("%s: não-monótona em rank %d (%.6f > %.6f)", c.nome, i+1, c.probs[i], c.probs[i-1])
			}
		}
	}
}

func TestOXProduzPermutacaoValida(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	n := 10
	for iter := 0; iter < 300; iter++ {
		pa := rng.Perm(n)
		pb := rng.Perm(n)
		f1, f2 := CruzamentoOX(pa, pb, rng)
		if !ehPermutacaoValida(f1, n) || !ehPermutacaoValida(f2, n) {
			t.Fatalf("OX inválido: f1=%v f2=%v (pa=%v pb=%v)", f1, f2, pa, pb)
		}
	}
}

func TestPMXProduzPermutacaoValida(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	n := 10
	for iter := 0; iter < 300; iter++ {
		pa := rng.Perm(n)
		pb := rng.Perm(n)
		f1, f2 := CruzamentoPMX(pa, pb, rng)
		if !ehPermutacaoValida(f1, n) || !ehPermutacaoValida(f2, n) {
			t.Fatalf("PMX inválido: f1=%v f2=%v (pa=%v pb=%v)", f1, f2, pa, pb)
		}
	}
}

func TestMutacoesProduzemPermutacaoValida(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	n := 10
	for iter := 0; iter < 300; iter++ {
		s := rng.Perm(n)
		MutacaoSwap(s, rng)
		if !ehPermutacaoValida(s, n) {
			t.Fatalf("swap inválido: %v", s)
		}
		inv := rng.Perm(n)
		MutacaoInversao(inv, rng)
		if !ehPermutacaoValida(inv, n) {
			t.Fatalf("inversão inválida: %v", inv)
		}
	}
}

func TestCalcularDistanciaTour(t *testing.T) {
	mat := [][]float64{
		{0, 1, 2},
		{1, 0, 3},
		{2, 3, 0},
	}
	// ciclo 0→1→2→0 = 1 + 3 + 2 = 6
	if d := CalcularDistanciaTour([]int{0, 1, 2}, mat); !aprox(d, 6, 1e-9) {
		t.Errorf("got %.3f want 6", d)
	}
	// rotação não muda o custo do ciclo
	if d := CalcularDistanciaTour([]int{1, 2, 0}, mat); !aprox(d, 6, 1e-9) {
		t.Errorf("rotacionado got %.3f want 6", d)
	}
}

func TestMapaSimetricoEConsistente(t *testing.T) {
	m := ConstruirMapa()
	n := len(m.Cidades)
	if n != 10 {
		t.Fatalf("esperado 10 cidades, got %d", n)
	}
	if len(m.Matriz) != n || len(m.Fonte) != n {
		t.Fatalf("matriz/fonte com dimensão errada")
	}
	for i := 0; i < n; i++ {
		if m.Matriz[i][i] != 0 {
			t.Errorf("diagonal [%d][%d] = %.2f, esperado 0", i, i, m.Matriz[i][i])
		}
		for j := 0; j < n; j++ {
			if !aprox(m.Matriz[i][j], m.Matriz[j][i], 1e-6) {
				t.Errorf("assimetria [%d][%d]=%.3f vs [%d][%d]=%.3f", i, j, m.Matriz[i][j], j, i, m.Matriz[j][i])
			}
			if i != j && m.Matriz[i][j] <= 0 {
				t.Errorf("distância não-positiva [%d][%d] = %.3f", i, j, m.Matriz[i][j])
			}
		}
	}
}

// Tabela da Aula 13 (slide 9): Uberaba(0) → Uberlândia(1) = 106 km, e deve vir
// marcada como dado de tabela (não preenchida por Haversine).
func TestMapaUsaTabelaDaAula13(t *testing.T) {
	m := ConstruirMapa()
	if !aprox(m.Matriz[0][1], 106, 1e-9) {
		t.Errorf("Uberaba→Uberlândia esperado 106 (tabela), got %.3f", m.Matriz[0][1])
	}
	if !m.Fonte[0][1] {
		t.Error("Uberaba→Uberlândia devia estar marcado como vindo da tabela")
	}
	// Uberaba(0) → Araguari(2) é "—" na tabela → preenchido por Haversine.
	if m.Fonte[0][2] {
		t.Error("Uberaba→Araguari devia estar marcado como preenchido (Haversine), não tabela")
	}
}

func TestTreinarMelhoraDistancia(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxGeracoes = 150
	cfg.Seed = 7
	res := Treinar(nil, cfg)
	if len(res.HistMelhor) == 0 {
		t.Fatal("sem histórico de melhor")
	}
	inicial := res.HistMelhor[0]
	if res.MelhorDist > inicial+1e-9 {
		t.Errorf("distância piorou: inicial %.2f, final %.2f", inicial, res.MelhorDist)
	}
	if !ehPermutacaoValida(res.MelhorTour, 10) {
		t.Errorf("melhor tour inválido: %v", res.MelhorTour)
	}
}

func TestTreinarRankingExpFunciona(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Selecao = SelRankingExp
	cfg.CExp = 1.1
	cfg.MaxGeracoes = 100
	cfg.Seed = 11
	res := Treinar(nil, cfg)
	if !ehPermutacaoValida(res.MelhorTour, 10) {
		t.Errorf("ranking exp: tour inválido %v", res.MelhorTour)
	}
}
