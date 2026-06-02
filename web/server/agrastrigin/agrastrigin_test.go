package agrastrigin

import (
	"math"
	"math/rand"
	"testing"
)

// TestRastriginMinimoNoZero — f(0,0,0) deve ser 0 (mínimo global conhecido).
func TestRastriginMinimoNoZero(t *testing.T) {
	got := Rastrigin([]float64{0, 0, 0})
	if math.Abs(got) > 1e-9 {
		t.Fatalf("Rastrigin(0,0,0) = %v; esperado 0", got)
	}
}

// TestRastriginValorConhecido — pontos não-zero devem dar > 0.
func TestRastriginValorPositivo(t *testing.T) {
	cases := [][]float64{
		{1, 0, 0},
		{0.5, 0.5, 0},
		{-2.1, 1.3, 0.8},
	}
	for _, x := range cases {
		got := Rastrigin(x)
		if got <= 0 {
			t.Fatalf("Rastrigin(%v) = %v; esperado > 0", x, got)
		}
	}
}

// TestRadcliffFormulasConvexas — confere a soma dos genes preserva o "centro
// de massa" dos pais (propriedade da combinação convexa).
//
//	c1 + c2 = β(xa+xb) + (1−β)(xa+xb) = xa + xb
func TestRadcliffFormulas(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pa := []float64{1, 2, 3}
	pb := []float64{-1, 0, 5}
	c1, c2 := CruzamentoRadcliff(pa, pb, rng)
	for i := 0; i < 3; i++ {
		soma := c1[i] + c2[i]
		esperado := pa[i] + pb[i]
		if math.Abs(soma-esperado) > 1e-9 {
			t.Fatalf("RADCLIFF gene %d: c1+c2=%v, pa+pb=%v", i, soma, esperado)
		}
		// cada filho dentro do intervalo dos pais (convexidade)
		lo := math.Min(pa[i], pb[i])
		hi := math.Max(pa[i], pb[i])
		for _, c := range []float64{c1[i], c2[i]} {
			if c < lo-1e-9 || c > hi+1e-9 {
				t.Fatalf("RADCLIFF gene %d: filho %v fora de [%v,%v]", i, c, lo, hi)
			}
		}
	}
}

// TestWrightFormulas — os 3 candidatos devem corresponder às fórmulas;
// validados/melhores devolvidos.
func TestWrightFormulas(t *testing.T) {
	pa := []float64{1, 2, 0}
	pb := []float64{2, 1, 1}
	dMin, dMax := -5.12, 5.12
	c1, c2 := CruzamentoWright(pa, pb, dMin, dMax, Rastrigin)
	// Calcula os 3 esperados
	wa := []float64{0.5*pa[0] + 0.5*pb[0], 0.5*pa[1] + 0.5*pb[1], 0.5*pa[2] + 0.5*pb[2]}
	wb := []float64{1.5*pa[0] - 0.5*pb[0], 1.5*pa[1] - 0.5*pb[1], 1.5*pa[2] - 0.5*pb[2]}
	wc := []float64{-0.5*pa[0] + 1.5*pb[0], -0.5*pa[1] + 1.5*pb[1], -0.5*pa[2] + 1.5*pb[2]}
	cands := [][]float64{wa, wb, wc}
	contem := func(c []float64) bool {
		for _, w := range cands {
			ok := true
			for i := range c {
				if math.Abs(c[i]-w[i]) > 1e-9 {
					ok = false
					break
				}
			}
			if ok {
				return true
			}
		}
		return false
	}
	if !contem(c1) || !contem(c2) {
		t.Fatalf("WRIGHT filhos %v / %v não correspondem a nenhuma das fórmulas", c1, c2)
	}
}

// TestDeterminismo — mesma seed ⇒ mesmo resultado.
func TestDeterminismo(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxGeracoes = 50
	cfg.Seed = 42
	r1 := Treinar(nil, cfg)
	r2 := Treinar(nil, cfg)
	if r1.MelhorFx != r2.MelhorFx {
		t.Fatalf("não determinístico: %v vs %v", r1.MelhorFx, r2.MelhorFx)
	}
}

// TestConvergenciaRadcliff — em 200 gerações com RADCLIFF e pop razoável,
// o GA deve achar um valor BAIXO de Rastrigin (próximo a 0).
func TestConvergenciaRadcliff(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Cruzamento = CrossRadcliff
	cfg.MaxGeracoes = 300
	cfg.Seed = 42
	r := Treinar(nil, cfg)
	// Não exigimos zero perfeito (Rastrigin é difícil), mas deve ser modesto.
	// Threshold conservador — qualquer valor < 5 já indica convergência razoável.
	if r.MelhorFx > 5 {
		t.Fatalf("RADCLIFF não convergiu: melhorFx=%v esperado < 5", r.MelhorFx)
	}
	// Dentro do domínio
	for i, v := range r.MelhorX {
		if v < cfg.DominioMin-1e-9 || v > cfg.DominioMax+1e-9 {
			t.Fatalf("melhorX[%d]=%v fora do domínio [%v,%v]", i, v, cfg.DominioMin, cfg.DominioMax)
		}
	}
}

// TestConvergenciaWright — mesma coisa pro WRIGHT.
func TestConvergenciaWright(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Cruzamento = CrossWright
	cfg.MaxGeracoes = 300
	cfg.Seed = 42
	r := Treinar(nil, cfg)
	if r.MelhorFx > 5 {
		t.Fatalf("WRIGHT não convergiu: melhorFx=%v esperado < 5", r.MelhorFx)
	}
}

// TestMonotonicidadeMelhorGlobal — o melhor (rastreado por geração via Step)
// deve ser não-crescente.
func TestMonotonicidadeGlobal(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxGeracoes = 50
	cfg.Seed = 7
	ch := make(chan Step, 100)
	go func() {
		Treinar(ch, cfg)
		close(ch)
	}()
	var prev float64 = math.Inf(1)
	for s := range ch {
		if s.MelhorGlobalFx > prev+1e-9 {
			t.Fatalf("melhor global piorou na geração %d: %v → %v", s.Geracao, prev, s.MelhorGlobalFx)
		}
		prev = s.MelhorGlobalFx
	}
}

// TestDominioRespeitado — todos os indivíduos devem estar dentro do domínio.
func TestDominioRespeitado(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxGeracoes = 30
	cfg.Seed = 1
	ch := make(chan Step, 50)
	go func() {
		Treinar(ch, cfg)
		close(ch)
	}()
	for s := range ch {
		for _, ind := range s.Populacao {
			for j, v := range ind.X {
				if v < cfg.DominioMin-1e-9 || v > cfg.DominioMax+1e-9 {
					t.Fatalf("ger %d ind X[%d]=%v fora de [%v,%v]", s.Geracao, j, v, cfg.DominioMin, cfg.DominioMax)
				}
			}
		}
	}
}
