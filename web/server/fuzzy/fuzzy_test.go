package fuzzy

import (
	"math"
	"testing"
)

// O exemplo canônico da apostila (seção 2.7.3): cor 15 UH, pH 7, turbidez 0 UT
// → Q = 0.6, qualidade ADEQUADA. É o teste de regressão de todo o sistema.
func TestExemploApostila(t *testing.T) {
	r := Avaliar(Entrada{Cor: 15, PH: 7, Turbidez: 0})
	if math.Abs(r.Centroide-0.6) > 0.01 {
		t.Fatalf("centroide = %.4f, esperado 0.60 ± 0.01", r.Centroide)
	}
	if r.Classe != QAdequada {
		t.Fatalf("classe = %q, esperado %q", r.Classe, QAdequada)
	}
	// cor 15 está no cruzamento adequada/inadequada → exatamente 2 regras ativas
	// (aparência adequada + aparência inadequada, ambas com pH bom e turbidez boa).
	if r.RegrasAtivas != 2 {
		t.Fatalf("regras ativas = %d, esperado 2", r.RegrasAtivas)
	}
	for _, ra := range r.Regras {
		if ra.Forca > 0 && math.Abs(ra.Forca-0.5) > 1e-9 {
			t.Fatalf("força da regra ativa = %.4f, esperado 0.5", ra.Forca)
		}
		if ra.Forca > 0 && ra.Saida != QAdequada {
			t.Fatalf("regra ativa aponta pra %q, esperado %q", ra.Saida, QAdequada)
		}
	}
}

func TestAguaPerfeita(t *testing.T) {
	r := Avaliar(Entrada{Cor: 0, PH: 7.5, Turbidez: 0})
	if r.Classe != QBoa {
		t.Fatalf("classe = %q, esperado boa", r.Classe)
	}
	if r.Centroide < 0.8 {
		t.Fatalf("centroide = %.4f, esperado > 0.8", r.Centroide)
	}
}

func TestEsgoto(t *testing.T) {
	r := Avaliar(Entrada{Cor: 30, PH: 12, Turbidez: 10})
	if r.Classe != QInadequada {
		t.Fatalf("classe = %q, esperado inadequada", r.Classe)
	}
	if r.Centroide > 0.3 {
		t.Fatalf("centroide = %.4f, esperado < 0.3", r.Centroide)
	}
}

// pH fora da faixa segura derruba a qualidade mesmo com cor e turbidez ótimas.
func TestPhInadequadoDominaTudo(t *testing.T) {
	r := Avaliar(Entrada{Cor: 0, PH: 4, Turbidez: 0})
	if r.Classe != QInadequada {
		t.Fatalf("classe = %q, esperado inadequada (pH 4)", r.Classe)
	}
}

// Turbidez inadequada → inadequada em TODAS as 15 combinações (coluna 3 das tabelas).
func TestTurbidezInadequadaDominaTudo(t *testing.T) {
	r := Avaliar(Entrada{Cor: 0, PH: 7.5, Turbidez: 9})
	if r.Classe != QInadequada {
		t.Fatalf("classe = %q, esperado inadequada (turbidez 9)", r.Classe)
	}
}

func TestTrapezio(t *testing.T) {
	tr := Trapezio{4, 6, 14, 16}
	casos := []struct{ x, mu float64 }{
		{3, 0}, {4, 0}, {5, 0.5}, {6, 1}, {10, 1}, {14, 1}, {15, 0.5}, {16, 0}, {20, 0},
	}
	for _, c := range casos {
		if got := tr.Mu(c.x); math.Abs(got-c.mu) > 1e-9 {
			t.Errorf("Mu(%.1f) = %.4f, esperado %.4f", c.x, got, c.mu)
		}
	}
	// ombros nas bordas do domínio (a == b e c == d) valem 1 na borda.
	esq := Trapezio{0, 0, 4, 6}
	if esq.Mu(0) != 1 {
		t.Errorf("ombro esquerdo: Mu(0) = %.4f, esperado 1", esq.Mu(0))
	}
	dir := Trapezio{14, 16, 30, 30}
	if dir.Mu(30) != 1 {
		t.Errorf("ombro direito: Mu(30) = %.4f, esperado 1", dir.Mu(30))
	}
}

// As 45 regras cobrem todas as combinações de antecedentes, sem repetição.
func TestCoberturaRegras(t *testing.T) {
	if len(Regras) != 45 {
		t.Fatalf("len(Regras) = %d, esperado 45 (3×5×3)", len(Regras))
	}
	vistas := map[[3]string]bool{}
	for _, r := range Regras {
		k := [3]string{r.Aparencia, r.PH, r.Turbidez}
		if vistas[k] {
			t.Fatalf("regra duplicada: %v", k)
		}
		vistas[k] = true
		if r.Saida != QBoa && r.Saida != QAdequada && r.Saida != QInadequada {
			t.Fatalf("consequente inválido: %q", r.Saida)
		}
	}
}

// Em qualquer ponto do domínio de cada variável, algum termo tem μ > 0 —
// garante que sempre existe regra ativa (e o centroide nunca divide por zero).
func TestCoberturaDominio(t *testing.T) {
	for _, v := range []Variavel{VarCor, VarPH, VarTurbidez, VarQualidade} {
		for i := 0; i <= 200; i++ {
			x := v.Min + (v.Max-v.Min)*float64(i)/200
			max := 0.0
			for _, termo := range v.Termos {
				if mu := termo.Trap.Mu(x); mu > max {
					max = mu
				}
			}
			if max <= 0 {
				t.Fatalf("%s: nenhum termo cobre x = %.3f", v.ID, x)
			}
		}
	}
}

func TestSuperficie(t *testing.T) {
	s, err := GerarSuperficie("ph", "turbidez", Entrada{Cor: 15, PH: 7, Turbidez: 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Xs) != nSurf || len(s.Ys) != nSurf || len(s.Z) != nSurf {
		t.Fatalf("dimensões erradas: %d×%d (esperado %d)", len(s.Xs), len(s.Z), nSurf)
	}
	for i := range s.Z {
		if len(s.Z[i]) != nSurf {
			t.Fatalf("linha %d com %d colunas", i, len(s.Z[i]))
		}
		for j, q := range s.Z[i] {
			if q < 0 || q > 1 || math.IsNaN(q) {
				t.Fatalf("Z[%d][%d] = %.4f fora de [0,1]", i, j, q)
			}
		}
	}
	// eixos inválidos são rejeitados.
	if _, err := GerarSuperficie("ph", "ph", Entrada{}); err == nil {
		t.Fatal("eixos iguais deveriam dar erro")
	}
	if _, err := GerarSuperficie("banana", "ph", Entrada{}); err == nil {
		t.Fatal("eixo inexistente deveria dar erro")
	}
}

// Entradas fora do domínio são clampadas, não explodem.
func TestClampEntradas(t *testing.T) {
	r := Avaliar(Entrada{Cor: -5, PH: 99, Turbidez: -1})
	if r.Entrada.Cor != 0 || r.Entrada.PH != 14 || r.Entrada.Turbidez != 0 {
		t.Fatalf("clamp falhou: %+v", r.Entrada)
	}
}
