package matching

import (
	"math"
	"testing"
)

func TestHaversineKnownPair(t *testing.T) {
	// Rondonópolis (-16.47, -54.64) → Santos (-23.96, -46.33). ~1100 km.
	d := HaversineKm(-16.47, -54.64, -23.96, -46.33)
	if d < 1000 || d > 1300 {
		t.Errorf("expected ~1100km, got %.1f", d)
	}
}

func TestPremioProteina(t *testing.T) {
	cases := []struct {
		prot float64
		want float64
	}{
		{35.0, 0},
		{36.0, 0},
		{37.0, 1.0},
		{40.0, 4.0},
		{45.0, 4.0}, // capped
	}
	for _, c := range cases {
		got := premioProteina(c.prot)
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("premioProteina(%.1f) = %.3f, want %.3f", c.prot, got, c.want)
		}
	}
}

func TestEvaluateAllUnmatched(t *testing.T) {
	s, _ := BuildScenario(ScenarioBalanceado, 42)
	cfg := DefaultConfig()
	c := make(Chromosome, len(s.Lots))
	for i := range c {
		c[i] = -1
	}
	br := Evaluate(s, c, cfg)
	if br.NumMatched != 0 {
		t.Errorf("expected 0 matched, got %d", br.NumMatched)
	}
	if br.Violacoes != 0 {
		t.Errorf("expected 0 violations when all unmatched, got %d", br.Violacoes)
	}
	if br.Fitness != 0 {
		t.Errorf("expected fitness=0, got %.3f", br.Fitness)
	}
}

func TestEvaluateAllToTrader0OverCapacity(t *testing.T) {
	s, _ := BuildScenario(ScenarioBalanceado, 42)
	cfg := DefaultConfig()
	c := make(Chromosome, len(s.Lots))
	for i := range c {
		c[i] = 0
	}
	br := Evaluate(s, c, cfg)
	if br.NumMatched != len(s.Lots) {
		t.Errorf("expected %d matched, got %d", len(s.Lots), br.NumMatched)
	}
	// Volume total de 6 lotes (2500..5000 t cada) = 15k..30k > capacidade 9k → overflow
	if !br.TraderStats[0].OverCapacity {
		t.Errorf("expected trader 0 to be over capacity")
	}
	if br.Violacoes < 1 {
		t.Errorf("expected at least 1 violation")
	}
}
