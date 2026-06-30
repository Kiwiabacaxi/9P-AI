package rnaga

import (
	"math"
	"math/rand"
	"testing"
)

func TestDecodeDentroDasFaixas(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 2000; i++ {
		c := CromossomoAleatorio(rng)
		if n := c.Neuronios(); n < 2 || n > 15 {
			t.Fatalf("neuronios fora da faixa: %d", n)
		}
		if l := c.Camadas(); l < 2 || l > 5 {
			t.Fatalf("camadas fora da faixa: %d", l)
		}
		if a := c.TaxaAprend(); a < 1e-5 || a > 0.1 {
			t.Fatalf("taxa fora da faixa: %g", a)
		}
		if e := c.MaxEpocas(); e < 20 || e > 1000 {
			t.Fatalf("epocas fora da faixa: %d", e)
		}
	}
}

func TestNovoValorGeneSempreValido(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	base := CromossomoAleatorio(rng)
	for pos := 0; pos < 6; pos++ {
		for i := 0; i < 500; i++ {
			c := base
			c.Genes[pos] = NovoValorGene(pos, rng)
			// todos os decoders devem continuar válidos
			if c.Neuronios() < 2 || c.Neuronios() > 15 ||
				c.Camadas() < 2 || c.Camadas() > 5 ||
				c.TaxaAprend() < 1e-5 || c.TaxaAprend() > 0.1 ||
				c.MaxEpocas() < 20 || c.MaxEpocas() > 1000 {
				t.Fatalf("pos %d gerou cromossomo inválido: %s", pos, c.String())
			}
		}
	}
}

func TestMutacaoForcadaContinuaValida(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 1000; i++ {
		c := CromossomoAleatorio(rng)
		Mutacao(&c, 1.0, rng) // prob 1 → sempre muta uma posição
		if c.Neuronios() < 2 || c.Neuronios() > 15 ||
			c.Camadas() < 2 || c.Camadas() > 5 ||
			c.TaxaAprend() < 1e-5 || c.TaxaAprend() > 0.1 ||
			c.MaxEpocas() < 20 || c.MaxEpocas() > 1000 {
			t.Fatalf("mutação gerou inválido: %s", c.String())
		}
	}
}

func TestCruzamentoUmPontoComplementar(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for i := 0; i < 500; i++ {
		pa := CromossomoAleatorio(rng)
		pb := CromossomoAleatorio(rng)
		fa, fb := CruzamentoUmPonto(pa, pb, rng)
		// cada gene do filho vem de um dos pais; os dois filhos são complementares
		for k := 0; k < 6; k++ {
			okA := fa.Genes[k] == pa.Genes[k] || fa.Genes[k] == pb.Genes[k]
			okB := fb.Genes[k] == pa.Genes[k] || fb.Genes[k] == pb.Genes[k]
			if !okA || !okB {
				t.Fatalf("gene %d não veio de nenhum pai", k)
			}
			// complementaridade: se fa pegou de pa, fb pegou de pb (e vice-versa)
			if fa.Genes[k] == pa.Genes[k] && fb.Genes[k] != pb.Genes[k] && pa.Genes[k] != pb.Genes[k] {
				t.Fatalf("gene %d não é complementar", k)
			}
		}
	}
}

func TestDatasetShapeEFaixas(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ds := GerarDataset(rng)
	if len(ds.X) != 100 || len(ds.Y) != 100 {
		t.Fatalf("esperado 100 padrões, got X=%d Y=%d", len(ds.X), len(ds.Y))
	}
	for i := 0; i < 100; i++ {
		if len(ds.X[i]) != 15 || len(ds.Y[i]) != 13 {
			t.Fatalf("padrão %d: dims X=%d Y=%d", i, len(ds.X[i]), len(ds.Y[i]))
		}
		for _, x := range ds.X[i] {
			if x < 3 || x > 1457 {
				t.Fatalf("entrada fora da faixa: %g", x)
			}
		}
		for _, y := range ds.Y[i] {
			if y < 58 || y > 312 {
				t.Fatalf("saída fora da faixa: %g", y)
			}
		}
	}
}

func TestMSEDeterministico(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	ds := GerarDataset(rng)
	c := CromossomoAleatorio(rng)
	a := AvaliarMSE(c, ds, 80, 42)
	b := AvaliarMSE(c, ds, 80, 42)
	if a != b {
		t.Fatalf("MSE não determinístico: %g != %g", a, b)
	}
	if math.IsNaN(a) || math.IsInf(a, 0) {
		t.Fatalf("MSE inválido: %g", a)
	}
}

func TestNormalizacaoMelhoraMSE(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	ds := GerarDataset(rng)
	// arquitetura decente, variando só a normalização (gene 5: 1=norm, 2=sem)
	base := Cromossomo{Genes: [6]float64{10, 3, 0.05, 400, 2 /*offline*/, 1 /*norm*/}}
	semNorm := base
	semNorm.Genes[5] = 2
	mseNorm := AvaliarMSE(base, ds, 400, 99)
	mseSem := AvaliarMSE(semNorm, ds, 400, 99)
	if !(mseNorm < mseSem) {
		t.Fatalf("normalizar devia reduzir MSE: norm=%g sem=%g", mseNorm, mseSem)
	}
}

// O ponto central da honestidade do benchmark: as otimizações NÃO mudam o que é
// computado, só a velocidade — então o MSE final é idêntico nos 4 modos.
func TestBenchmarkModosMesmoMSE(t *testing.T) {
	cfg := Config{PopSize: 10, MaxGeracoes: 5, ProbMutacao: 0.05, TetoEpocas: 60, Seed: 7}
	res := RodarBenchmark(nil, cfg)
	if len(res.Modos) != 5 {
		t.Fatalf("esperado 5 modos, got %d", len(res.Modos))
	}
	atual := res.Modos[4].MelhorMSE
	// modos 1..4 usam gonum no offline → devem ser BIT-idênticos entre si.
	for i := 1; i <= 4; i++ {
		if math.Abs(res.Modos[i].MelhorMSE-atual) > 1e-6 {
			t.Errorf("modo %q (gonum): MSE difere de bit-idêntico (%g vs %g)", res.Modos[i].Nome, res.Modos[i].MelhorMSE, atual)
		}
	}
	// modo 0 usa matmul de laços → praticamente idêntico (diferença ínfima de FP).
	if rel := math.Abs(res.Modos[0].MelhorMSE-atual) / atual; rel > 0.05 {
		t.Errorf("modo ingênuo (laços): MSE divergiu demais (%.4f vs %.4f, rel=%.3f)", res.Modos[0].MelhorMSE, atual, rel)
	}
	if res.Modos[4].CacheHits <= 0 {
		t.Errorf("o modo com memoização deveria ter cache hits, got %d", res.Modos[4].CacheHits)
	}
}

func TestTreinarReduzMelhorMSE(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PopSize = 12
	cfg.MaxGeracoes = 8
	cfg.TetoEpocas = 60
	cfg.Seed = 123
	res := Treinar(nil, cfg)
	if len(res.HistMelhor) == 0 {
		t.Fatal("sem histórico")
	}
	if res.MelhorMSE > res.HistMelhor[0]+1e-9 {
		t.Fatalf("melhor MSE piorou: inicial %g final %g", res.HistMelhor[0], res.MelhorMSE)
	}
	// vencedor decodifica para faixas válidas
	v := res.MelhorCromossomo
	if v.Neuronios() < 2 || v.Neuronios() > 15 || v.Camadas() < 2 || v.Camadas() > 5 {
		t.Fatalf("vencedor inválido: %s", v.String())
	}
}
