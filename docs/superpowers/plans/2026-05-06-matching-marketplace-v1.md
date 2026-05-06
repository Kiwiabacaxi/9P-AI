# Matching Marketplace — Etapa 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar uma nova aba "Matching" ao 9P-AI que modela escoamento de soja MT→Santos como problema de matching marketplace (N produtores × M traders) resolvido por GA, substituindo conceitualmente o TSP. Mantém o TSP atual intacto.

**Architecture:** Backend Go isolado em `web/server/matching/` (mesmo padrão de `tsp/`, `genetico/`). Frontend React em `views/MatchingView.tsx` com visualização Leaflet em `components/viz/MatchingMap.tsx`. Comunicação via REST + SSE pra progresso por geração.

**Tech Stack:** Go (stdlib + math/rand), React 19, react-leaflet 5, recharts, SSE.

**Etapa atual:** **1 (single-objective + 1 baseline + 2 cenários sintéticos)**.
**Etapas futuras (não cobertas aqui):** Hungarian baseline, modo 60×6, NSGA-II multi-objetivo, calibração com dados reais (Comex/CONAB/CEPEA), índice IOSCO, modal switching. Ver [docs/brainstorm-GA.md](../../brainstorm-GA.md).

---

## Out of Scope (Etapa 1)

Pra evitar drift de escopo durante execução, estes itens **NÃO devem ser implementados** nesta etapa:

- Hungarian baseline (etapa 2)
- NSGA-II / multi-objetivo (etapa 3)
- Calibração com dados reais (etapa 3)
- Modo 60×6 produtores (etapa 2 — só 6×4 nesta etapa)
- Cenários "Comprador Dominante" e "Preço Alto" (etapa 2 — só "Balanceado" e "Crise de Qualidade")
- Modal switching (etapa 4+)
- IOSCO (etapa 4+)
- Robustez estocástica (etapa 4+)
- Integração OSRM real para rotas matching (etapa 2 — usar haversine nesta etapa pra simplificar; OSRM já existe pro TSP, podemos reusar depois)

---

## File Structure

### Backend (Go)
- **Create** `web/server/matching/types.go` — Domain types: `Producer`, `Trader`, `Lot`, `Match`, `Config`, `Step`, `Result`, `Scenario`
- **Create** `web/server/matching/scenarios.go` — Scenario presets (Balanceado, Crise de Qualidade) — geração determinística com seed
- **Create** `web/server/matching/fitness.go` — Cálculo de preço, superávit, penalidades hard
- **Create** `web/server/matching/fitness_test.go` — Testes unitários da fitness (cromossomo hardcoded → fitness conhecida)
- **Create** `web/server/matching/genetic.go` — Loop GA: população, torneio, crossover uniforme + repair, mutação, elitismo, função `Treinar`
- **Create** `web/server/matching/baselines.go` — Greedy by reserve price
- **Modify** `web/server/main.go` — Adicionar handlers `/api/matching/*`, registrar mux, status flags

### Frontend (React)
- **Create** `web/frontend/src/views/MatchingView.tsx` — View principal (controles, run, painel lateral, mapa, gráficos)
- **Create** `web/frontend/src/components/viz/MatchingMap.tsx` — Mapa Leaflet com produtores/traders/porto + rotas multi-cor
- **Create** `web/frontend/src/api/matching.ts` (opcional, pode ficar inline no view) — Funções de fetch tipadas
- **Modify** `web/frontend/src/api/types.ts` — Adicionar tipos do matching e `'matching'` ao `ViewId`
- **Modify** `web/frontend/src/App.tsx` — Importar `MatchingView`, registrar em `viewComponents`
- **Modify** `web/frontend/src/components/layout/Sidebar.tsx` — Adicionar entrada "Matching · Soja" sob seção "Algoritmo Genético"

### Docs
- **Modify** `docs/brainstorm-GA.md` — Adicionar nota no topo apontando pro plano e marcando Etapa 1 como em execução

---

## Task 1: Branch e estrutura de pastas

**Files:**
- Branch já criada: `feat/matching-marketplace` (verificar)
- Criar diretório: `web/server/matching/`

- [ ] **Step 1: Confirmar branch correta**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `feat/matching-marketplace`

- [ ] **Step 2: Criar diretório do package**

```bash
mkdir -p web/server/matching
```

- [ ] **Step 3: Commit do plano (se ainda não feito)**

```bash
git add docs/superpowers/plans/2026-05-06-matching-marketplace-v1.md
git status
git commit -m "docs: add matching marketplace v1 plan"
```

---

## Task 2: Domain types

**Files:**
- Create: `web/server/matching/types.go`

- [ ] **Step 1: Escrever types.go**

```go
package matching

// Producer representa um produtor de soja em um município.
type Producer struct {
	ID        int     `json:"id"`
	Nome      string  `json:"nome"`
	Municipio string  `json:"municipio"`
	UF        string  `json:"uf"`
	Lat       float64 `json:"lat"`
	Lng       float64 `json:"lng"`
}

// Lot é uma oferta de venda de um produtor com qualidade e preço de reserva.
// Volume em toneladas. Proteína/umidade/impurezas em percentual (0-100).
// Janela em "semanas" simbólicas (0..N).
type Lot struct {
	ID            int     `json:"id"`
	ProducerID    int     `json:"producerId"`
	VolumeT       float64 `json:"volumeT"`
	Proteina      float64 `json:"proteina"`
	Umidade       float64 `json:"umidade"`
	Impurezas     float64 `json:"impurezas"`
	PrecoReserva  float64 `json:"precoReserva"`  // R$/saca60
	JanelaSemana  int     `json:"janelaSemana"`
}

// Trader é um comprador (Cargill, Bunge, etc.) com hub geográfico.
type Trader struct {
	ID             int     `json:"id"`
	Nome           string  `json:"nome"`
	Cor            string  `json:"cor"`            // hex pra UI
	HubMunicipio   string  `json:"hubMunicipio"`
	HubLat         float64 `json:"hubLat"`
	HubLng         float64 `json:"hubLng"`
	CapacidadeT    float64 `json:"capacidadeT"`
	ProteinaMin    float64 `json:"proteinaMin"`    // spec mínima blend ponderado
	UmidadeMax     float64 `json:"umidadeMax"`
	ImpurezasMax   float64 `json:"impurezasMax"`
	PrecoMaximo    float64 `json:"precoMaximo"`    // R$/saca60
	JanelaSemana   int     `json:"janelaSemana"`   // semana do navio em Santos
}

// Scenario agrega um setup completo do problema.
type Scenario struct {
	ID          string     `json:"id"`
	Nome        string     `json:"nome"`
	Descricao   string     `json:"descricao"`
	Producers   []Producer `json:"producers"`
	Lots        []Lot      `json:"lots"`
	Traders     []Trader   `json:"traders"`
	PrecoBase   float64    `json:"precoBase"`     // R$/saca60 dia (Santos FOB)
	PortLat     float64    `json:"portLat"`       // Porto Santos
	PortLng     float64    `json:"portLng"`
}

// Match é o resultado de uma atribuição: cromossomo[i] = trader index, ou -1.
type Chromosome []int

// Config controla o GA.
type Config struct {
	PopSize         int     `json:"popSize"`
	MaxGeracoes     int     `json:"maxGeracoes"`
	ProbCruzamento  float64 `json:"probCruzamento"`
	ProbMutacao     float64 `json:"probMutacao"`
	TamanhoTorneio  int     `json:"tamanhoTorneio"`
	Elitismo        int     `json:"elitismo"`
	LambdaLog       float64 `json:"lambdaLog"`     // peso custo logístico
	LambdaQual      float64 `json:"lambdaQual"`    // peso penalidade qualidade
	MBig            float64 `json:"mBig"`          // penalidade hard
	Seed            int64   `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		PopSize:        80,
		MaxGeracoes:    200,
		ProbCruzamento: 0.85,
		ProbMutacao:    0.20,
		TamanhoTorneio: 4,
		Elitismo:       2,
		LambdaLog:      0.05,
		LambdaQual:     50.0,
		MBig:           1e6,
	}
}

// TraderStats por trader pra UI/painel lateral.
type TraderStats struct {
	TraderID         int     `json:"traderId"`
	VolumeAlocadoT   float64 `json:"volumeAlocadoT"`
	NumLotes         int     `json:"numLotes"`
	BlendProteina    float64 `json:"blendProteina"`
	BlendUmidade     float64 `json:"blendUmidade"`
	BlendImpurezas   float64 `json:"blendImpurezas"`
	OverCapacity     bool    `json:"overCapacity"`
	UnderSpec        bool    `json:"underSpec"`
}

// Step é um snapshot por geração emitido via SSE.
type Step struct {
	Geracao         int           `json:"geracao"`
	MelhorCrom      Chromosome    `json:"melhorCrom"`
	MelhorFitness   float64       `json:"melhorFitness"`
	MediaFitness    float64       `json:"mediaFitness"`
	PiorFitness     float64       `json:"piorFitness"`
	MelhorSuperavit float64       `json:"melhorSuperavit"`
	MelhorViolacoes int           `json:"melhorViolacoes"`
	TraderStats     []TraderStats `json:"traderStats"`
	NumMatched      int           `json:"numMatched"`
}

// Result é o estado final de um treino.
type Result struct {
	Geracoes      int        `json:"geracoes"`
	MelhorCrom    Chromosome `json:"melhorCrom"`
	MelhorFitness float64    `json:"melhorFitness"`
	HistMelhor    []float64  `json:"histMelhor"`
	HistMedia     []float64  `json:"histMedia"`
	Cfg           Config     `json:"cfg"`
	ScenarioID    string     `json:"scenarioId"`
}
```

- [ ] **Step 2: Verificar compila**

```bash
cd web/server && go build ./matching/
```
Expected: nenhuma saída (sucesso)

- [ ] **Step 3: Commit**

```bash
git add web/server/matching/types.go
git commit -m "feat(matching): domain types — Producer, Trader, Lot, Chromosome, Config"
```

---

## Task 3: Cenários sintéticos

**Files:**
- Create: `web/server/matching/scenarios.go`

Cenários nesta etapa: **Balanceado** (capacidades simétricas, qualidades razoáveis) e **Crise de Qualidade** (40% dos lotes com proteína < 35).

Coordenadas dos hubs reais do MT-GO (do brainstorm doc):
- Cargill — Rondonópolis MT (-16.47, -54.64)
- Bunge — Cuiabá MT (-15.60, -56.10)
- ADM — Alto Araguaia MT (-17.31, -53.21)
- COFCO — Rio Verde GO (-17.79, -50.93)

Porto Santos: (-23.96, -46.33).

Produtores (6) — municípios reais com produção de soja:
1. Sorriso MT (-12.55, -55.72)
2. Sapezal MT (-13.55, -58.81)
3. Primavera do Leste MT (-15.56, -54.30)
4. Campo Verde MT (-15.55, -55.16)
5. Jataí GO (-17.88, -51.71)
6. Mineiros GO (-17.57, -52.55)

- [ ] **Step 1: Escrever scenarios.go**

```go
package matching

import "math/rand"

const (
	ScenarioBalanceado     = "balanceado"
	ScenarioCriseQualidade = "crise-qualidade"
)

// portCoords para Santos.
const (
	PortSantosLat = -23.96
	PortSantosLng = -46.33
)

func baseProducers() []Producer {
	return []Producer{
		{ID: 0, Nome: "Fazenda Sorriso", Municipio: "Sorriso", UF: "MT", Lat: -12.55, Lng: -55.72},
		{ID: 1, Nome: "Fazenda Sapezal", Municipio: "Sapezal", UF: "MT", Lat: -13.55, Lng: -58.81},
		{ID: 2, Nome: "Fazenda Primavera", Municipio: "Primavera do Leste", UF: "MT", Lat: -15.56, Lng: -54.30},
		{ID: 3, Nome: "Fazenda Campo Verde", Municipio: "Campo Verde", UF: "MT", Lat: -15.55, Lng: -55.16},
		{ID: 4, Nome: "Fazenda Jataí", Municipio: "Jataí", UF: "GO", Lat: -17.88, Lng: -51.71},
		{ID: 5, Nome: "Fazenda Mineiros", Municipio: "Mineiros", UF: "GO", Lat: -17.57, Lng: -52.55},
	}
}

func baseTraders() []Trader {
	return []Trader{
		{
			ID: 0, Nome: "Cargill", Cor: "#E63946",
			HubMunicipio: "Rondonópolis", HubLat: -16.47, HubLng: -54.64,
			CapacidadeT: 9000, ProteinaMin: 36.0, UmidadeMax: 14.0, ImpurezasMax: 1.0,
			PrecoMaximo: 145, JanelaSemana: 1,
		},
		{
			ID: 1, Nome: "Bunge", Cor: "#2A9D8F",
			HubMunicipio: "Cuiabá", HubLat: -15.60, HubLng: -56.10,
			CapacidadeT: 7500, ProteinaMin: 36.0, UmidadeMax: 14.0, ImpurezasMax: 1.0,
			PrecoMaximo: 142, JanelaSemana: 1,
		},
		{
			ID: 2, Nome: "ADM", Cor: "#F4A261",
			HubMunicipio: "Alto Araguaia", HubLat: -17.31, HubLng: -53.21,
			CapacidadeT: 8000, ProteinaMin: 36.5, UmidadeMax: 13.5, ImpurezasMax: 1.0,
			PrecoMaximo: 148, JanelaSemana: 1,
		},
		{
			ID: 3, Nome: "COFCO", Cor: "#264653",
			HubMunicipio: "Rio Verde", HubLat: -17.79, HubLng: -50.93,
			CapacidadeT: 6500, ProteinaMin: 36.0, UmidadeMax: 14.0, ImpurezasMax: 1.0,
			PrecoMaximo: 140, JanelaSemana: 1,
		},
	}
}

// ScenarioBalanceadoBuilder gera 1 lote por produtor, qualidade razoável, volumes simétricos.
func buildBalanceado(seed int64) Scenario {
	rng := rand.New(rand.NewSource(seed))
	prods := baseProducers()
	traders := baseTraders()
	lots := make([]Lot, 0, len(prods))
	for i, p := range prods {
		volume := 2500 + rng.Float64()*2500 // 2500..5000 t
		lots = append(lots, Lot{
			ID:           i,
			ProducerID:   p.ID,
			VolumeT:      volume,
			Proteina:     36.0 + rng.Float64()*3.0, // 36..39
			Umidade:      12.0 + rng.Float64()*2.0, // 12..14
			Impurezas:    0.5 + rng.Float64()*0.5,  // 0.5..1.0
			PrecoReserva: 130 + rng.Float64()*8,    // 130..138
			JanelaSemana: 1,
		})
	}
	return Scenario{
		ID:        ScenarioBalanceado,
		Nome:      "Balanceado",
		Descricao: "Capacidades simétricas, lotes uniformes — caso base de convergência",
		Producers: prods,
		Lots:      lots,
		Traders:   traders,
		PrecoBase: 138,
		PortLat:   PortSantosLat,
		PortLng:   PortSantosLng,
	}
}

// buildCriseQualidade: ~40% dos lotes com proteína < 35 (chuva ruim).
// Força o GA a fazer blends inteligentes.
func buildCriseQualidade(seed int64) Scenario {
	rng := rand.New(rand.NewSource(seed))
	prods := baseProducers()
	traders := baseTraders()
	lots := make([]Lot, 0, len(prods))
	for i, p := range prods {
		volume := 2500 + rng.Float64()*2500
		var proteina float64
		if rng.Float64() < 0.40 {
			proteina = 33.0 + rng.Float64()*2.0 // 33..35 ruim
		} else {
			proteina = 37.0 + rng.Float64()*3.0 // 37..40 excelente
		}
		lots = append(lots, Lot{
			ID:           i,
			ProducerID:   p.ID,
			VolumeT:      volume,
			Proteina:     proteina,
			Umidade:      12.0 + rng.Float64()*2.5,
			Impurezas:    0.5 + rng.Float64()*0.6,
			PrecoReserva: 128 + rng.Float64()*10, // 128..138
			JanelaSemana: 1,
		})
	}
	return Scenario{
		ID:        ScenarioCriseQualidade,
		Nome:      "Crise de Qualidade",
		Descricao: "40% dos lotes com proteína < 35 — força blend inteligente",
		Producers: prods,
		Lots:      lots,
		Traders:   traders,
		PrecoBase: 138,
		PortLat:   PortSantosLat,
		PortLng:   PortSantosLng,
	}
}

// BuildScenario retorna o cenário pelo ID. Seed default = 42.
func BuildScenario(id string, seed int64) (Scenario, bool) {
	if seed == 0 {
		seed = 42
	}
	switch id {
	case ScenarioBalanceado:
		return buildBalanceado(seed), true
	case ScenarioCriseQualidade:
		return buildCriseQualidade(seed), true
	}
	return Scenario{}, false
}

// ScenarioMetadata é um descritor leve pra dropdown.
type ScenarioMetadata struct {
	ID        string `json:"id"`
	Nome      string `json:"nome"`
	Descricao string `json:"descricao"`
}

func ListScenarios() []ScenarioMetadata {
	return []ScenarioMetadata{
		{ID: ScenarioBalanceado, Nome: "Balanceado", Descricao: "Capacidades simétricas, qualidades razoáveis"},
		{ID: ScenarioCriseQualidade, Nome: "Crise de Qualidade", Descricao: "40% dos lotes com proteína baixa"},
	}
}
```

- [ ] **Step 2: Verificar compila**

```bash
cd web/server && go build ./matching/
```
Expected: nenhuma saída.

- [ ] **Step 3: Commit**

```bash
git add web/server/matching/scenarios.go
git commit -m "feat(matching): synthetic scenarios — balanceado, crise-qualidade"
```

---

## Task 4: Fitness function

**Files:**
- Create: `web/server/matching/fitness.go`
- Create: `web/server/matching/fitness_test.go`

Fitness conforme brainstorm doc:
```
fitness(c) = superávit_total(c)
           - λ_log × custo_logístico(c)
           - λ_qual × penalidade_blend(c)
           - M_BIG × violações_hard(c)
```

- 1 saca = 60 kg → tonelada → 16.667 sacas (constante).
- Distância via haversine (etapa 1, sem OSRM).
- `preco_pago(i,j) = precoBase + premio_proteina(lot) - desconto_log(dist) - capped @ trader.PrecoMaximo`.
- Premio: +1 R$/saca por ponto de proteína acima de 36, máx 4 pontos.
- Desconto log: 0.05 R$/saca por km (rough). Distância produtor→hub trader.

- [ ] **Step 1: Escrever fitness.go**

```go
package matching

import "math"

const (
	SacasPorTonelada = 1000.0 / 60.0 // 16.667
)

// HaversineKm calcula distância em km entre dois pontos lat/lng.
func HaversineKm(latA, lngA, latB, lngB float64) float64 {
	const R = 6371.0
	rad := math.Pi / 180.0
	dLat := (latB - latA) * rad
	dLng := (lngB - lngA) * rad
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(latA*rad)*math.Cos(latB*rad)*math.Sin(dLng/2)*math.Sin(dLng/2)
	return 2 * R * math.Asin(math.Sqrt(a))
}

// premioProteina: +R$/saca por ponto de proteína acima de 36, capado em 4.
func premioProteina(proteina float64) float64 {
	delta := proteina - 36.0
	if delta < 0 {
		return 0
	}
	if delta > 4 {
		delta = 4
	}
	return delta * 1.0
}

// descontoLog: R$/saca = 0.05 * dist_km (≈ R$3/saca em 60km, escala ok pra demo).
func descontoLog(distKm float64) float64 {
	return 0.05 * distKm
}

// PrecoPago calcula o preço por saca que trader j pagaria pelo lote i.
// Capado em PrecoMaximo do trader.
func PrecoPago(s Scenario, lotIdx, traderIdx int) float64 {
	lot := s.Lots[lotIdx]
	trader := s.Traders[traderIdx]
	prod := findProducer(s.Producers, lot.ProducerID)
	dist := HaversineKm(prod.Lat, prod.Lng, trader.HubLat, trader.HubLng)
	preco := s.PrecoBase + premioProteina(lot.Proteina) - descontoLog(dist)
	if preco > trader.PrecoMaximo {
		preco = trader.PrecoMaximo
	}
	return preco
}

func findProducer(prods []Producer, id int) Producer {
	for _, p := range prods {
		if p.ID == id {
			return p
		}
	}
	return Producer{}
}

// EvaluateBreakdown avalia um cromossomo e retorna métricas detalhadas.
type FitnessBreakdown struct {
	Fitness         float64
	SuperavitTotal  float64
	CustoLogTotal   float64
	PenalidadeQual  float64
	Violacoes       int
	NumMatched      int
	TraderStats     []TraderStats
}

func Evaluate(s Scenario, c Chromosome, cfg Config) FitnessBreakdown {
	M := len(s.Traders)
	N := len(s.Lots)
	stats := make([]TraderStats, M)
	for j := range stats {
		stats[j].TraderID = s.Traders[j].ID
	}

	// blend acumuladores
	weightedProt := make([]float64, M)
	weightedUmid := make([]float64, M)
	weightedImp := make([]float64, M)

	var superavit, custoLog float64
	var matched int

	for i := 0; i < N; i++ {
		j := c[i]
		if j < 0 || j >= M {
			continue
		}
		matched++
		lot := s.Lots[i]
		trader := s.Traders[j]
		prod := findProducer(s.Producers, lot.ProducerID)
		dist := HaversineKm(prod.Lat, prod.Lng, trader.HubLat, trader.HubLng)
		preco := PrecoPago(s, i, j)
		sacas := lot.VolumeT * SacasPorTonelada
		// superávit: (preco - reserva) * sacas; pode ser negativo se reserva > preço
		superavit += (preco - lot.PrecoReserva) * sacas
		// custo logístico: dist * volume (custo de transporte; quanto maior, pior)
		custoLog += dist * lot.VolumeT
		stats[j].VolumeAlocadoT += lot.VolumeT
		stats[j].NumLotes++
		weightedProt[j] += lot.Proteina * lot.VolumeT
		weightedUmid[j] += lot.Umidade * lot.VolumeT
		weightedImp[j] += lot.Impurezas * lot.VolumeT
	}

	// finaliza blends
	for j := range stats {
		v := stats[j].VolumeAlocadoT
		if v > 0 {
			stats[j].BlendProteina = weightedProt[j] / v
			stats[j].BlendUmidade = weightedUmid[j] / v
			stats[j].BlendImpurezas = weightedImp[j] / v
		}
	}

	// violações hard + penalidade qual
	var penalQual float64
	var violacoes int
	for j, t := range s.Traders {
		st := &stats[j]
		// capacidade
		if st.VolumeAlocadoT > t.CapacidadeT {
			st.OverCapacity = true
			violacoes++
		}
		if st.NumLotes == 0 {
			continue
		}
		// blend mínimo de proteína (penalidade quadrática)
		if st.BlendProteina < t.ProteinaMin {
			st.UnderSpec = true
			violacoes++
			d := t.ProteinaMin - st.BlendProteina
			penalQual += d * d * st.VolumeAlocadoT
		}
		// blend max umidade
		if st.BlendUmidade > t.UmidadeMax {
			st.UnderSpec = true
			violacoes++
			d := st.BlendUmidade - t.UmidadeMax
			penalQual += d * d * st.VolumeAlocadoT
		}
		// impurezas
		if st.BlendImpurezas > t.ImpurezasMax {
			st.UnderSpec = true
			violacoes++
			d := st.BlendImpurezas - t.ImpurezasMax
			penalQual += d * d * st.VolumeAlocadoT
		}
	}

	fit := superavit - cfg.LambdaLog*custoLog - cfg.LambdaQual*penalQual - cfg.MBig*float64(violacoes)
	return FitnessBreakdown{
		Fitness:        fit,
		SuperavitTotal: superavit,
		CustoLogTotal:  custoLog,
		PenalidadeQual: penalQual,
		Violacoes:      violacoes,
		NumMatched:     matched,
		TraderStats:    stats,
	}
}
```

- [ ] **Step 2: Escrever fitness_test.go**

```go
package matching

import (
	"math"
	"testing"
)

func TestHaversineKnownPair(t *testing.T) {
	// Rondonópolis (-16.47, -54.64) → Santos (-23.96, -46.33). ~1100 km.
	d := HaversineKm(-16.47, -54.64, -23.96, -46.33)
	if d < 1000 || d > 1200 {
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
```

- [ ] **Step 3: Rodar testes**

```bash
cd web/server && go test ./matching/ -v
```
Expected: PASS em todos os testes (3 funções de teste, ≥5 sub-asserts).

- [ ] **Step 4: Commit**

```bash
git add web/server/matching/fitness.go web/server/matching/fitness_test.go
git commit -m "feat(matching): fitness function with haversine + quality penalties + tests"
```

---

## Task 5: Genetic algorithm loop

**Files:**
- Create: `web/server/matching/genetic.go`

Operadores conforme brainstorm:
- Torneio k=4
- Crossover uniforme + repair (remove matches menos rentáveis até caber)
- Mutação composta: 50% swap, 30% reassign aleatório, 20% force unmatch (-1)
- Elitismo p=2

- [ ] **Step 1: Escrever genetic.go**

```go
package matching

import (
	"math/rand"
	"sort"
	"time"
)

// initialPopulation gera N cromossomos com matching aleatório (incluindo -1).
func initialPopulation(s Scenario, popSize int, rng *rand.Rand) []Chromosome {
	M := len(s.Traders)
	N := len(s.Lots)
	pop := make([]Chromosome, popSize)
	for i := range pop {
		c := make(Chromosome, N)
		for k := range c {
			r := rng.Intn(M + 1)
			if r == M {
				c[k] = -1
			} else {
				c[k] = r
			}
		}
		pop[i] = c
	}
	return pop
}

// torneio retorna o melhor de k indivíduos sorteados.
func torneio(pop []Chromosome, fits []float64, k int, rng *rand.Rand) Chromosome {
	bestIdx := rng.Intn(len(pop))
	bestFit := fits[bestIdx]
	for i := 1; i < k; i++ {
		idx := rng.Intn(len(pop))
		if fits[idx] > bestFit {
			bestFit = fits[idx]
			bestIdx = idx
		}
	}
	return cloneChrom(pop[bestIdx])
}

func cloneChrom(c Chromosome) Chromosome {
	cp := make(Chromosome, len(c))
	copy(cp, c)
	return cp
}

// crossoverUniforme produz 1 filho com gene[i] do pai aleatoriamente.
func crossoverUniforme(p1, p2 Chromosome, rng *rand.Rand) Chromosome {
	N := len(p1)
	child := make(Chromosome, N)
	for i := 0; i < N; i++ {
		if rng.Intn(2) == 0 {
			child[i] = p1[i]
		} else {
			child[i] = p2[i]
		}
	}
	return child
}

// repair: pra cada trader em overcapacity, remove iterativamente o lote menos rentável até caber.
// "Menos rentável" = menor (preço_pago - preço_reserva) por saca.
func repair(s Scenario, c Chromosome) Chromosome {
	M := len(s.Traders)
	for {
		// soma volumes por trader
		volPorTrader := make([]float64, M)
		for i, j := range c {
			if j < 0 || j >= M {
				continue
			}
			volPorTrader[j] += s.Lots[i].VolumeT
		}
		// acha primeiro trader em overflow
		troubled := -1
		for j := 0; j < M; j++ {
			if volPorTrader[j] > s.Traders[j].CapacidadeT {
				troubled = j
				break
			}
		}
		if troubled == -1 {
			return c
		}
		// remove o lote menos rentável atribuído a este trader
		worstIdx := -1
		worstMargin := 0.0
		first := true
		for i, j := range c {
			if j != troubled {
				continue
			}
			margin := PrecoPago(s, i, j) - s.Lots[i].PrecoReserva
			if first || margin < worstMargin {
				worstMargin = margin
				worstIdx = i
				first = false
			}
		}
		if worstIdx == -1 {
			return c
		}
		c[worstIdx] = -1
	}
}

// mutar aplica mutação composta com probMutacao por gene.
func mutar(c Chromosome, M int, probMutacao float64, rng *rand.Rand) {
	N := len(c)
	for i := 0; i < N; i++ {
		if rng.Float64() >= probMutacao {
			continue
		}
		r := rng.Float64()
		switch {
		case r < 0.50:
			// swap com outro gene
			j := rng.Intn(N)
			c[i], c[j] = c[j], c[i]
		case r < 0.80:
			// reassign aleatório
			c[i] = rng.Intn(M)
		default:
			// force unmatch
			c[i] = -1
		}
	}
}

// elitismo: extrai os top-p indivíduos.
func elites(pop []Chromosome, fits []float64, p int) []Chromosome {
	if p <= 0 {
		return nil
	}
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(a, b int) bool {
		return fits[idxs[a]] > fits[idxs[b]]
	})
	if p > len(idxs) {
		p = len(idxs)
	}
	out := make([]Chromosome, p)
	for i := 0; i < p; i++ {
		out[i] = cloneChrom(pop[idxs[i]])
	}
	return out
}

// Treinar roda o GA. progressCh recebe Steps a cada geração; quando feito, retorna Result.
func Treinar(progressCh chan<- Step, s Scenario, cfg Config) Result {
	cfg = sanitizeCfg(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	M := len(s.Traders)
	pop := initialPopulation(s, cfg.PopSize, rng)
	// reparo inicial
	for i := range pop {
		pop[i] = repair(s, pop[i])
	}

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)

	var melhorGlobal Chromosome
	var melhorFitGlobal float64
	primeiro := true

	for gen := 0; gen < cfg.MaxGeracoes; gen++ {
		// avaliação
		fits := make([]float64, len(pop))
		breakdowns := make([]FitnessBreakdown, len(pop))
		for i, c := range pop {
			br := Evaluate(s, c, cfg)
			fits[i] = br.Fitness
			breakdowns[i] = br
		}
		// stats
		var bestIdx int
		bestFit := fits[0]
		var sum, worst float64 = 0, fits[0]
		for i, f := range fits {
			sum += f
			if f > bestFit {
				bestFit = f
				bestIdx = i
			}
			if f < worst {
				worst = f
			}
		}
		mean := sum / float64(len(fits))
		bestBr := breakdowns[bestIdx]

		if primeiro || bestFit > melhorFitGlobal {
			melhorFitGlobal = bestFit
			melhorGlobal = cloneChrom(pop[bestIdx])
			primeiro = false
		}

		histMelhor = append(histMelhor, bestFit)
		histMedia = append(histMedia, mean)

		if progressCh != nil {
			progressCh <- Step{
				Geracao:         gen,
				MelhorCrom:      cloneChrom(pop[bestIdx]),
				MelhorFitness:   bestFit,
				MediaFitness:    mean,
				PiorFitness:     worst,
				MelhorSuperavit: bestBr.SuperavitTotal,
				MelhorViolacoes: bestBr.Violacoes,
				TraderStats:     bestBr.TraderStats,
				NumMatched:      bestBr.NumMatched,
			}
		}

		// próxima geração
		newPop := elites(pop, fits, cfg.Elitismo)
		for len(newPop) < cfg.PopSize {
			p1 := torneio(pop, fits, cfg.TamanhoTorneio, rng)
			p2 := torneio(pop, fits, cfg.TamanhoTorneio, rng)
			var child Chromosome
			if rng.Float64() < cfg.ProbCruzamento {
				child = crossoverUniforme(p1, p2, rng)
			} else {
				child = cloneChrom(p1)
			}
			mutar(child, M, cfg.ProbMutacao, rng)
			child = repair(s, child)
			newPop = append(newPop, child)
		}
		pop = newPop
	}

	return Result{
		Geracoes:      cfg.MaxGeracoes,
		MelhorCrom:    melhorGlobal,
		MelhorFitness: melhorFitGlobal,
		HistMelhor:    histMelhor,
		HistMedia:     histMedia,
		Cfg:           cfg,
		ScenarioID:    s.ID,
	}
}

func sanitizeCfg(cfg Config) Config {
	if cfg.PopSize <= 1 {
		cfg.PopSize = 80
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 200
	}
	if cfg.TamanhoTorneio <= 1 {
		cfg.TamanhoTorneio = 4
	}
	if cfg.Elitismo < 0 {
		cfg.Elitismo = 0
	}
	if cfg.MBig <= 0 {
		cfg.MBig = 1e6
	}
	return cfg
}
```

- [ ] **Step 2: Verificar compila e testes ainda passam**

```bash
cd web/server && go test ./matching/ -v
```
Expected: testes da Task 4 ainda passam (5+ asserts).

- [ ] **Step 3: Smoke test do GA via go run**

```bash
cd web/server && go test -run XXX -bench . -benchmem ./matching/ 2>&1 | head -3 || true
cd web/server && cat <<'EOF' > /tmp/smoke_matching.go
package main

import (
	"fmt"
	"mlp-server/matching"
)

func main() {
	s, _ := matching.BuildScenario(matching.ScenarioBalanceado, 42)
	cfg := matching.DefaultConfig()
	cfg.MaxGeracoes = 50
	res := matching.Treinar(nil, s, cfg)
	fmt.Printf("best fitness=%.2f, geracoes=%d, hist len=%d\n",
		res.MelhorFitness, res.Geracoes, len(res.HistMelhor))
	fmt.Printf("hist[0]=%.2f, hist[final]=%.2f\n",
		res.HistMelhor[0], res.HistMelhor[len(res.HistMelhor)-1])
}
EOF
go run /tmp/smoke_matching.go && rm /tmp/smoke_matching.go
```
Expected: imprime `best fitness=...` com `hist[final] >= hist[0]` (GA está melhorando).

- [ ] **Step 4: Commit**

```bash
git add web/server/matching/genetic.go
git commit -m "feat(matching): GA loop with tournament, uniform crossover + repair, mutation"
```

---

## Task 6: Greedy baseline

**Files:**
- Create: `web/server/matching/baselines.go`

Greedy: ordena lotes por preço de reserva (descrescente, mais valiosos primeiro), atribui ao trader com maior `preco_pago - reserva` que ainda tem capacidade. Se nenhum cabe, deixa unmatched.

- [ ] **Step 1: Escrever baselines.go**

```go
package matching

import "sort"

// GreedyByReserve atribui lotes em ordem de preço_reserva DESC ao trader que paga mais
// e ainda tem capacidade.
func GreedyByReserve(s Scenario) (Chromosome, FitnessBreakdown) {
	N := len(s.Lots)
	M := len(s.Traders)
	c := make(Chromosome, N)
	for i := range c {
		c[i] = -1
	}

	// indices ordenados por preço_reserva DESC
	idxs := make([]int, N)
	for i := range idxs {
		idxs[i] = i
	}
	sort.SliceStable(idxs, func(a, b int) bool {
		return s.Lots[idxs[a]].PrecoReserva > s.Lots[idxs[b]].PrecoReserva
	})

	usadoT := make([]float64, M)
	for _, i := range idxs {
		bestJ := -1
		bestMargin := 0.0
		first := true
		for j := 0; j < M; j++ {
			if usadoT[j]+s.Lots[i].VolumeT > s.Traders[j].CapacidadeT {
				continue
			}
			margin := PrecoPago(s, i, j) - s.Lots[i].PrecoReserva
			if first || margin > bestMargin {
				bestMargin = margin
				bestJ = j
				first = false
			}
		}
		if bestJ != -1 {
			c[i] = bestJ
			usadoT[bestJ] += s.Lots[i].VolumeT
		}
	}

	br := Evaluate(s, c, DefaultConfig())
	return c, br
}
```

- [ ] **Step 2: Adicionar test no fitness_test.go**

Append:

```go
func TestGreedyBaselineRespectsCapacity(t *testing.T) {
	s, _ := BuildScenario(ScenarioBalanceado, 42)
	_, br := GreedyByReserve(s)
	for j, st := range br.TraderStats {
		if st.OverCapacity {
			t.Errorf("greedy violou capacidade do trader %d (alocado=%.1f, cap=%.1f)",
				j, st.VolumeAlocadoT, s.Traders[j].CapacidadeT)
		}
	}
}
```

- [ ] **Step 3: Rodar testes**

```bash
cd web/server && go test ./matching/ -v
```
Expected: todos passam.

- [ ] **Step 4: Commit**

```bash
git add web/server/matching/baselines.go web/server/matching/fitness_test.go
git commit -m "feat(matching): greedy-by-reserve baseline"
```

---

## Task 7: Backend HTTP endpoints

**Files:**
- Modify: `web/server/main.go`

Endpoints (paralelos aos de tsp):
- `GET  /api/matching/scenarios` → lista de cenários disponíveis
- `POST /api/matching/scenario` (body: `{id, seed?}`) → carrega cenário e retorna Scenario completo
- `GET  /api/matching/config`
- `POST /api/matching/config`
- `GET  /api/matching/train` (SSE) → streaming de Steps + `event: done` com Result
- `POST /api/matching/baseline` (body: `{algoritmo: "greedy"}`) → roda baseline, retorna `{chromosome, breakdown}`
- `POST /api/matching/reset`
- `GET  /api/matching/result` → último Result

Status flags em main.go: `matchingTrained`, `matchingTraining` no AppStatus.

- [ ] **Step 1: Adicionar import no main.go**

Localizar bloco de imports em `web/server/main.go` (próximo da linha 25-30) e adicionar `"mlp-server/matching"` junto com os outros packages internos. Use Edit com contexto suficiente:

```go
// old (find these specific lines)
"mlp-server/genetico"
"mlp-server/genetico2"

// new
"mlp-server/genetico"
"mlp-server/genetico2"
"mlp-server/matching"
```

- [ ] **Step 2: Adicionar globals matching**

Localizar bloco com `gaCfg`, `tspCfg` (próximo da linha 120-140). Adicionar **logo após** o bloco do TSP:

```go
matchingScenario *matching.Scenario
matchingCfg      *matching.Config
matchingRes      *matching.Result
matchingTraining bool
```

- [ ] **Step 3: Adicionar campos no AppStatus**

Localizar `type` anônima com `TspTrained`, `TspTraining` (próximo da linha 212). Adicionar:

```go
MatchingTrained  bool `json:"matchingTrained"`
MatchingTraining bool `json:"matchingTraining"`
```

E no preenchimento do struct (próximo linha 238):

```go
MatchingTrained:  matchingRes != nil,
MatchingTraining: matchingTraining,
```

- [ ] **Step 4: Adicionar handlers no main.go**

Adicionar **antes** de `func main()` ou onde os outros handlers ficam (perto do `handleTspTrain`):

```go
func handleMatchingScenarios(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, matching.ListScenarios())
}

func handleMatchingScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		errJSON(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var body struct {
		ID   string `json:"id"`
		Seed int64  `json:"seed"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		errJSON(w, http.StatusBadRequest, "json invalido")
		return
	}
	s, ok := matching.BuildScenario(body.ID, body.Seed)
	if !ok {
		errJSON(w, http.StatusBadRequest, "scenario id desconhecido")
		return
	}
	matchingScenario = &s
	matchingRes = nil
	writeJSON(w, http.StatusOK, s)
}

func handleMatchingConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		cfg := matching.DefaultConfig()
		if matchingCfg != nil {
			cfg = *matchingCfg
		}
		writeJSON(w, http.StatusOK, cfg)
		return
	}
	if r.Method == http.MethodPost {
		var cfg matching.Config
		if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
			errJSON(w, http.StatusBadRequest, "json invalido")
			return
		}
		matchingCfg = &cfg
		writeJSON(w, http.StatusOK, cfg)
		return
	}
	errJSON(w, http.StatusMethodNotAllowed, "use GET ou POST")
}

func handleMatchingTrain(w http.ResponseWriter, r *http.Request) {
	if matchingScenario == nil {
		errJSON(w, http.StatusBadRequest, "carregue cenario primeiro: POST /api/matching/scenario")
		return
	}
	cfg := matching.DefaultConfig()
	if matchingCfg != nil {
		cfg = *matchingCfg
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		errJSON(w, http.StatusInternalServerError, "streaming nao suportado")
		return
	}

	matchingTraining = true
	defer func() { matchingTraining = false }()

	progressCh := make(chan matching.Step, 32)
	doneCh := make(chan matching.Result, 1)
	go func() {
		res := matching.Treinar(progressCh, *matchingScenario, cfg)
		close(progressCh)
		doneCh <- res
	}()

	for step := range progressCh {
		b, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	res := <-doneCh
	matchingRes = &res
	b, _ := json.Marshal(res)
	fmt.Fprintf(w, "event: done\ndata: %s\n\n", b)
	flusher.Flush()
}

func handleMatchingBaseline(w http.ResponseWriter, r *http.Request) {
	if matchingScenario == nil {
		errJSON(w, http.StatusBadRequest, "carregue cenario primeiro")
		return
	}
	var body struct {
		Algoritmo string `json:"algoritmo"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body.Algoritmo == "" {
		body.Algoritmo = "greedy"
	}
	switch body.Algoritmo {
	case "greedy":
		c, br := matching.GreedyByReserve(*matchingScenario)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"algoritmo":  body.Algoritmo,
			"chromosome": c,
			"breakdown":  br,
		})
	default:
		errJSON(w, http.StatusBadRequest, "algoritmo desconhecido")
	}
}

func handleMatchingReset(w http.ResponseWriter, r *http.Request) {
	matchingScenario = nil
	matchingRes = nil
	matchingCfg = nil
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}

func handleMatchingResult(w http.ResponseWriter, r *http.Request) {
	if matchingRes == nil {
		errJSON(w, http.StatusNotFound, "nenhum resultado ainda")
		return
	}
	writeJSON(w, http.StatusOK, matchingRes)
}
```

**NOTA:** os helpers `writeJSON` e `errJSON` já existem no main.go — confirmar e usar os existentes (mesmas assinaturas que `handleTspBaseline` usa). Se os nomes diferirem (ex: `writeJson`), ajustar.

- [ ] **Step 5: Registrar rotas**

Localizar bloco onde `/api/tsp/*` são registradas (linha ~2256). Adicionar **logo após**:

```go
mux.HandleFunc("/api/matching/scenarios", cors(handleMatchingScenarios))
mux.HandleFunc("/api/matching/scenario",  cors(handleMatchingScenario))
mux.HandleFunc("/api/matching/config",    cors(handleMatchingConfig))
mux.HandleFunc("/api/matching/train",     cors(handleMatchingTrain))
mux.HandleFunc("/api/matching/baseline",  cors(handleMatchingBaseline))
mux.HandleFunc("/api/matching/reset",     cors(handleMatchingReset))
mux.HandleFunc("/api/matching/result",    cors(handleMatchingResult))
```

- [ ] **Step 6: Build server**

```bash
cd web/server && go build -o mlp-server .
```
Expected: nenhuma saída.

- [ ] **Step 7: Smoke test endpoints com curl**

Em terminal separado:

```bash
cd web/server && ./mlp-server &
SERVER_PID=$!
sleep 1
curl -s http://localhost:8080/api/matching/scenarios | head -c 300
curl -s -X POST http://localhost:8080/api/matching/scenario -H 'Content-Type: application/json' -d '{"id":"balanceado","seed":42}' | head -c 300
curl -s -X POST http://localhost:8080/api/matching/baseline -H 'Content-Type: application/json' -d '{"algoritmo":"greedy"}' | head -c 400
kill $SERVER_PID
```
Expected: 3 respostas JSON válidas.

- [ ] **Step 8: Commit**

```bash
git add web/server/main.go
git commit -m "feat(matching): HTTP endpoints — scenarios, config, train (SSE), baseline, reset"
```

---

## Task 8: Frontend types

**Files:**
- Modify: `web/frontend/src/api/types.ts`

- [ ] **Step 1: Adicionar tipos do matching no fim de types.ts**

Append antes da linha `export type ViewId = ...`:

```ts
// Matching Marketplace (Etapa 1)
export interface MatchingProducer {
  id: number;
  nome: string;
  municipio: string;
  uf: string;
  lat: number;
  lng: number;
}

export interface MatchingLot {
  id: number;
  producerId: number;
  volumeT: number;
  proteina: number;
  umidade: number;
  impurezas: number;
  precoReserva: number;
  janelaSemana: number;
}

export interface MatchingTrader {
  id: number;
  nome: string;
  cor: string;
  hubMunicipio: string;
  hubLat: number;
  hubLng: number;
  capacidadeT: number;
  proteinaMin: number;
  umidadeMax: number;
  impurezasMax: number;
  precoMaximo: number;
  janelaSemana: number;
}

export interface MatchingScenario {
  id: string;
  nome: string;
  descricao: string;
  producers: MatchingProducer[];
  lots: MatchingLot[];
  traders: MatchingTrader[];
  precoBase: number;
  portLat: number;
  portLng: number;
}

export interface MatchingScenarioMeta {
  id: string;
  nome: string;
  descricao: string;
}

export interface MatchingConfig {
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  tamanhoTorneio: number;
  elitismo: number;
  lambdaLog: number;
  lambdaQual: number;
  mBig: number;
  seed?: number;
}

export interface MatchingTraderStats {
  traderId: number;
  volumeAlocadoT: number;
  numLotes: number;
  blendProteina: number;
  blendUmidade: number;
  blendImpurezas: number;
  overCapacity: boolean;
  underSpec: boolean;
}

export interface MatchingStep {
  geracao: number;
  melhorCrom: number[];
  melhorFitness: number;
  mediaFitness: number;
  piorFitness: number;
  melhorSuperavit: number;
  melhorViolacoes: number;
  traderStats: MatchingTraderStats[];
  numMatched: number;
}

export interface MatchingResult {
  geracoes: number;
  melhorCrom: number[];
  melhorFitness: number;
  histMelhor: number[];
  histMedia: number[];
  cfg: MatchingConfig;
  scenarioId: string;
}

export interface MatchingFitnessBreakdown {
  Fitness: number;
  SuperavitTotal: number;
  CustoLogTotal: number;
  PenalidadeQual: number;
  Violacoes: number;
  NumMatched: number;
  TraderStats: MatchingTraderStats[];
}

export interface MatchingBaselineResp {
  algoritmo: string;
  chromosome: number[];
  breakdown: MatchingFitnessBreakdown;
}
```

**NOTA:** os campos PascalCase em `MatchingFitnessBreakdown` refletem que a struct Go não tem json tags — Go serializa com nomes de campo originais. Confirmar no runtime; se aparecer minúsculo, adicionar tags na Go struct e ajustar tipos aqui.

- [ ] **Step 2: Adicionar 'matching' ao ViewId**

Localizar `export type ViewId = ...` (final do arquivo). Editar:

```ts
// old
| 'genetico' | 'genetico2' | 'tsp' | 'tsp-compare'
| 'about';

// new
| 'genetico' | 'genetico2' | 'tsp' | 'tsp-compare'
| 'matching'
| 'about';
```

- [ ] **Step 3: Adicionar campos status no AppStatus**

Localizar `export interface AppStatus { ... }` (topo do arquivo). Adicionar:

```ts
matchingTrained: boolean;
matchingTraining: boolean;
```

- [ ] **Step 4: Verificar tipo compila**

```bash
cd web/frontend && npx tsc -b --noEmit
```
Expected: nenhum erro.

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/api/types.ts
git commit -m "feat(matching): frontend types"
```

---

## Task 9: Frontend view skeleton + Leaflet map base

**Files:**
- Create: `web/frontend/src/components/viz/MatchingMap.tsx`
- Create: `web/frontend/src/views/MatchingView.tsx`

Visualização (etapa 1):
- Marker circle por produtor (cor cinza neutro inicialmente; muda pra cor do trader quando matched)
- Marker quadrado/diferente por trader hub (na cor do trader)
- Marker grande pro porto Santos
- Linha haversine produtor→hub trader na cor do trader (matched)
- Linha haversine hub trader→porto na cor do trader, mais grossa, com tracejado vermelho se overCapacity

Como referência, ler [web/frontend/src/components/viz/TspMap.tsx](../../../web/frontend/src/components/viz/TspMap.tsx) — usa react-leaflet 5 com `MapContainer`, `TileLayer`, `Polyline`, `CircleMarker`. Replicar o estilo.

- [ ] **Step 1: Criar MatchingMap.tsx (props mínimas + render base)**

```tsx
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from 'react-leaflet';
import type { MatchingScenario, MatchingTraderStats } from '../../api/types';

interface Props {
  scenario: MatchingScenario | null;
  chromosome: number[] | null;            // gene[i] = traderIdx, ou -1
  traderStats: MatchingTraderStats[] | null;
}

export default function MatchingMap({ scenario, chromosome, traderStats }: Props) {
  if (!scenario) {
    return <div className="map-empty">Carregue um cenário pra começar</div>;
  }
  const center: [number, number] = [-18, -52];

  // overload por trader
  const overloadSet = new Set<number>(
    (traderStats ?? [])
      .filter(s => s.overCapacity)
      .map(s => s.traderId)
  );

  return (
    <MapContainer center={center} zoom={5} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Porto Santos */}
      <CircleMarker
        center={[scenario.portLat, scenario.portLng]}
        radius={10}
        pathOptions={{ color: '#000', fillColor: '#444', fillOpacity: 0.9 }}
      >
        <Tooltip direction="top">Porto de Santos</Tooltip>
      </CircleMarker>

      {/* Trader hubs */}
      {scenario.traders.map(t => (
        <CircleMarker
          key={`trader-${t.id}`}
          center={[t.hubLat, t.hubLng]}
          radius={9}
          pathOptions={{ color: t.cor, fillColor: t.cor, fillOpacity: 0.95, weight: 2 }}
        >
          <Tooltip direction="top">
            <strong>{t.nome}</strong> — {t.hubMunicipio}<br />
            cap: {t.capacidadeT.toFixed(0)} t · prot ≥ {t.proteinaMin}
          </Tooltip>
        </CircleMarker>
      ))}

      {/* Produtores */}
      {scenario.producers.map(p => {
        const lotIdx = scenario.lots.findIndex(l => l.producerId === p.id);
        const matched = chromosome && lotIdx >= 0 ? chromosome[lotIdx] : -1;
        const cor = matched >= 0 ? scenario.traders[matched].cor : '#aaa';
        return (
          <CircleMarker
            key={`prod-${p.id}`}
            center={[p.lat, p.lng]}
            radius={6}
            pathOptions={{ color: cor, fillColor: cor, fillOpacity: 0.85 }}
          >
            <Tooltip direction="top">
              <strong>{p.nome}</strong> — {p.municipio}/{p.uf}
              {lotIdx >= 0 && (
                <>
                  <br />vol: {scenario.lots[lotIdx].volumeT.toFixed(0)} t · prot: {scenario.lots[lotIdx].proteina.toFixed(1)}
                </>
              )}
            </Tooltip>
          </CircleMarker>
        );
      })}

      {/* Linhas produtor→trader (matched) */}
      {chromosome && scenario.producers.map(p => {
        const lotIdx = scenario.lots.findIndex(l => l.producerId === p.id);
        if (lotIdx < 0) return null;
        const j = chromosome[lotIdx];
        if (j < 0 || j >= scenario.traders.length) return null;
        const trader = scenario.traders[j];
        return (
          <Polyline
            key={`pt-${p.id}`}
            positions={[[p.lat, p.lng], [trader.hubLat, trader.hubLng]]}
            pathOptions={{ color: trader.cor, weight: 2, opacity: 0.8 }}
          />
        );
      })}

      {/* Linhas trader→porto (todos os traders, grosso) */}
      {scenario.traders.map(t => {
        const overload = overloadSet.has(t.id);
        return (
          <Polyline
            key={`tp-${t.id}`}
            positions={[[t.hubLat, t.hubLng], [scenario.portLat, scenario.portLng]]}
            pathOptions={{
              color: overload ? '#e63946' : t.cor,
              weight: 4,
              opacity: 0.85,
              dashArray: overload ? '8 6' : undefined,
            }}
          />
        );
      })}
    </MapContainer>
  );
}
```

- [ ] **Step 2: Criar MatchingView.tsx skeleton (sem treino ainda — só carrega cenário e mostra mapa)**

```tsx
import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import type { MatchingScenario, MatchingScenarioMeta } from '../api/types';
import MatchingMap from '../components/viz/MatchingMap';

export default function MatchingView() {
  const [scenarios, setScenarios] = useState<MatchingScenarioMeta[]>([]);
  const [scenarioId, setScenarioId] = useState<string>('');
  const [scenario, setScenario] = useState<MatchingScenario | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<MatchingScenarioMeta[]>('/matching/scenarios')
      .then(s => {
        setScenarios(s);
        if (s.length > 0) setScenarioId(s[0].id);
      })
      .catch(e => setErr(String(e)));
  }, []);

  async function loadScenario() {
    if (!scenarioId) return;
    setLoading(true); setErr(null);
    try {
      const s = await apiPost<MatchingScenario>('/matching/scenario', { id: scenarioId, seed: 42 });
      setScenario(s);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="view matching-view" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16, height: '100%' }}>
      <aside className="matching-sidebar" style={{ padding: 12, overflowY: 'auto' }}>
        <h2>Matching · Soja</h2>
        <p style={{ fontSize: 13, color: '#666' }}>
          Etapa 1: matching marketplace single-objective (6 produtores × 4 traders → Santos).
        </p>

        <div style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontSize: 12, marginBottom: 4 }}>Cenário</label>
          <select
            value={scenarioId}
            onChange={e => setScenarioId(e.target.value)}
            style={{ width: '100%', padding: 6 }}
          >
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>{s.nome}</option>
            ))}
          </select>
          <button
            onClick={loadScenario}
            disabled={loading || !scenarioId}
            style={{ marginTop: 8, width: '100%' }}
          >
            {loading ? 'Carregando…' : 'Carregar cenário'}
          </button>
        </div>

        {err && <div className="error" style={{ marginTop: 8, color: '#e63946' }}>{err}</div>}

        {scenario && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ fontSize: 14 }}>{scenario.nome}</h3>
            <p style={{ fontSize: 12, color: '#666' }}>{scenario.descricao}</p>
            <ul style={{ fontSize: 12, paddingLeft: 16 }}>
              <li>{scenario.producers.length} produtores</li>
              <li>{scenario.lots.length} lotes</li>
              <li>{scenario.traders.length} traders</li>
              <li>preço base: R${scenario.precoBase.toFixed(2)}/saca</li>
            </ul>
          </div>
        )}
      </aside>

      <div className="matching-map-area" style={{ height: '100%', minHeight: 500 }}>
        <MatchingMap scenario={scenario} chromosome={null} traderStats={null} />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verificar tsc compila**

```bash
cd web/frontend && npx tsc -b --noEmit
```
Expected: nenhum erro.

- [ ] **Step 4: Commit**

```bash
git add web/frontend/src/components/viz/MatchingMap.tsx web/frontend/src/views/MatchingView.tsx
git commit -m "feat(matching): MatchingView skeleton + MatchingMap (Leaflet base)"
```

---

## Task 10: Sidebar + App routing — smoke test visual

**Files:**
- Modify: `web/frontend/src/App.tsx`
- Modify: `web/frontend/src/components/layout/Sidebar.tsx`

- [ ] **Step 1: Importar e registrar MatchingView no App.tsx**

Localizar imports e `viewComponents` em `web/frontend/src/App.tsx`. Editar:

```tsx
// old
import GeneticoV2View from './views/GeneticoV2View';
import TspView from './views/TspView';

// new
import GeneticoV2View from './views/GeneticoV2View';
import MatchingView from './views/MatchingView';
import TspView from './views/TspView';
```

E em `viewComponents`:

```tsx
// old
genetico2: GeneticoV2View,
tsp: TspView,

// new
genetico2: GeneticoV2View,
matching: MatchingView,
tsp: TspView,
```

- [ ] **Step 2: Adicionar entrada na Sidebar**

Localizar a seção "Algoritmo Genético" em `Sidebar.tsx` (próximo do final, onde tem `genetico` e `genetico2` como nav items inline). Adicionar **logo após** o item de `genetico2`:

```tsx
<div
  className={`nav-item${active === 'matching' ? ' active' : ''}`}
  onClick={() => onNavigate('matching')}
>
  <span className="nav-icon" style={{ textAlign: 'center', fontSize: '12px' }}>{'⛵'}</span>
  Matching · Soja
</div>
```

(`⛵` = ⛵ — temático "navio"; pode ajustar.)

- [ ] **Step 3: Build + run dev**

Em terminal separado:

```bash
make dev
```

Aguarde 5s. Abra `http://localhost:5173` no browser.

- [ ] **Step 4: Smoke test manual**

Checklist visual (verificar e marcar mentalmente; relate falhas no commit):
1. Sidebar mostra "Matching · Soja" sob seção "Algoritmo Genético"
2. Clicar nele troca pra view do matching
3. Dropdown lista "Balanceado" e "Crise de Qualidade"
4. Botão "Carregar cenário" → mapa renderiza com produtores cinzas, traders coloridos, porto preto, e linhas grossas trader→porto
5. Tooltips funcionam ao passar mouse

Parar dev server (Ctrl+C).

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/App.tsx web/frontend/src/components/layout/Sidebar.tsx
git commit -m "feat(matching): register MatchingView in App + Sidebar nav"
```

---

## Task 11: Treinamento + animação por geração

**Files:**
- Modify: `web/frontend/src/views/MatchingView.tsx`

Conectar SSE: ao clicar "Treinar GA", abrir `apiSSE('/matching/train', ...)` e atualizar estado a cada `Step` recebido. Mapa re-renderiza com `chromosome` atual + `traderStats` atual. Ao receber evento `done`, capturar `Result` final.

- [ ] **Step 1: Adicionar estado e função de treino no MatchingView**

Adicionar imports e estado (no topo do componente):

```tsx
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  MatchingScenario, MatchingScenarioMeta,
  MatchingStep, MatchingResult, MatchingTraderStats,
  MatchingBaselineResp,
} from '../api/types';
```

Adicionar estados:

```tsx
const [training, setTraining] = useState(false);
const [step, setStep] = useState<MatchingStep | null>(null);
const [result, setResult] = useState<MatchingResult | null>(null);
const [baseline, setBaseline] = useState<MatchingBaselineResp | null>(null);
```

Adicionar função:

```tsx
async function startTrain() {
  if (!scenario) return;
  setTraining(true); setStep(null); setResult(null); setErr(null);
  const stop = apiSSE('/matching/train', {
    onMessage: (data: unknown) => setStep(data as MatchingStep),
    onDone: (data: unknown) => {
      setResult(data as MatchingResult);
      setTraining(false);
    },
    onError: () => {
      setErr('erro no streaming');
      setTraining(false);
    },
  });
  void stop;
}

async function runBaseline() {
  if (!scenario) return;
  setErr(null);
  try {
    const r = await apiPost<MatchingBaselineResp>('/matching/baseline', { algoritmo: 'greedy' });
    setBaseline(r);
  } catch (e) {
    setErr(String(e));
  }
}
```

- [ ] **Step 2: Adicionar UI dos botões no aside (depois do bloco que mostra cenário)**

```tsx
{scenario && (
  <div style={{ marginTop: 16 }}>
    <button
      onClick={startTrain}
      disabled={training}
      style={{ width: '100%', marginBottom: 4 }}
    >
      {training ? `Treinando… (gen ${step?.geracao ?? 0})` : 'Treinar GA'}
    </button>
    <button onClick={runBaseline} disabled={training} style={{ width: '100%' }}>
      Rodar baseline (greedy)
    </button>
  </div>
)}

{step && (
  <div style={{ marginTop: 16, fontSize: 12 }}>
    <h3 style={{ fontSize: 13 }}>Geração {step.geracao}</h3>
    <div>fitness: {step.melhorFitness.toFixed(0)}</div>
    <div>superávit: R${step.melhorSuperavit.toFixed(0)}</div>
    <div>matched: {step.numMatched}/{scenario?.lots.length ?? 0}</div>
    <div>violações: {step.melhorViolacoes}</div>
  </div>
)}

{baseline && (
  <div style={{ marginTop: 16, fontSize: 12, padding: 8, background: '#f6f6f6' }}>
    <h3 style={{ fontSize: 13 }}>Baseline (greedy)</h3>
    <div>fitness: {baseline.breakdown.Fitness.toFixed(0)}</div>
    <div>superávit: R${baseline.breakdown.SuperavitTotal.toFixed(0)}</div>
    <div>matched: {baseline.breakdown.NumMatched}</div>
    <div>violações: {baseline.breakdown.Violacoes}</div>
  </div>
)}
```

- [ ] **Step 3: Passar step ao MatchingMap pra animação**

Substituir o componente `<MatchingMap ... />` por:

```tsx
<MatchingMap
  scenario={scenario}
  chromosome={step?.melhorCrom ?? result?.melhorCrom ?? null}
  traderStats={step?.traderStats ?? null}
/>
```

- [ ] **Step 4: tsc check**

```bash
cd web/frontend && npx tsc -b --noEmit
```
Expected: nenhum erro.

- [ ] **Step 5: Smoke test ao vivo**

```bash
make dev
```

Browser http://localhost:5173 → Matching · Soja → Carregar cenário → Treinar GA.

Checklist:
1. Botão fica "Treinando… (gen N)" enquanto roda
2. Mapa anima: produtores trocam de cor a cada geração conforme matching evolui
3. Linha trader→porto fica vermelha tracejada quando há overload, depois normal
4. Stats no painel atualizam (fitness, superávit, matched, violações)
5. Ao terminar, botão volta a "Treinar GA"
6. Rodar baseline mostra valores numéricos no painel

Parar dev server.

- [ ] **Step 6: Commit**

```bash
git add web/frontend/src/views/MatchingView.tsx
git commit -m "feat(matching): live SSE training + greedy baseline UI + animated map"
```

---

## Task 12: Painel lateral por trader + gráfico de evolução

**Files:**
- Modify: `web/frontend/src/views/MatchingView.tsx`

Adicionar:
1. Card por trader mostrando: nome, barra de capacidade alocada/total, blend de proteína, status (OK/over/under)
2. Gráfico recharts com histórico de fitness (melhor + média) — pode capturar `step` em array

- [ ] **Step 1: Adicionar histórico de steps no estado**

```tsx
const [history, setHistory] = useState<MatchingStep[]>([]);
```

E na função `startTrain`, dentro de `onMessage`:

```tsx
onMessage: (data: unknown) => {
  const s = data as MatchingStep;
  setStep(s);
  setHistory(h => [...h, s]);
},
```

E no início de `startTrain` adicionar `setHistory([]);`.

- [ ] **Step 2: Adicionar import recharts e componente de gráfico inline**

Imports:

```tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ResponsiveContainer } from 'recharts';
```

Componente de gráfico (definir dentro do MatchingView, ou fora se preferir):

```tsx
function FitnessChart({ history }: { history: MatchingStep[] }) {
  if (history.length === 0) return null;
  const data = history.map(s => ({
    gen: s.geracao,
    melhor: s.melhorFitness,
    media: s.mediaFitness,
  }));
  return (
    <div style={{ height: 180 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="gen" />
          <YAxis />
          <RTooltip />
          <Line type="monotone" dataKey="melhor" stroke="#2a9d8f" dot={false} />
          <Line type="monotone" dataKey="media" stroke="#e9c46a" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

E renderizar no aside (dentro do bloco `step && ...`, depois das stats):

```tsx
<FitnessChart history={history} />
```

- [ ] **Step 3: Painel por trader (cards)**

Adicionar componente:

```tsx
function TraderCards({ scenario, traderStats }: {
  scenario: MatchingScenario;
  traderStats: MatchingTraderStats[] | null;
}) {
  if (!traderStats) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ fontSize: 13 }}>Traders</h3>
      {scenario.traders.map(t => {
        const st = traderStats.find(s => s.traderId === t.id);
        if (!st) return null;
        const pct = (st.volumeAlocadoT / t.capacidadeT) * 100;
        const status = st.overCapacity ? 'over' : st.underSpec ? 'under' : st.numLotes > 0 ? 'ok' : '—';
        return (
          <div key={t.id} style={{
            padding: 8, marginBottom: 6,
            border: `2px solid ${t.cor}`, borderRadius: 4, fontSize: 12,
            background: st.overCapacity ? '#ffe6e6' : st.underSpec ? '#fff5e6' : 'white',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong style={{ color: t.cor }}>{t.nome}</strong>
              <span>{status}</span>
            </div>
            <div style={{ marginTop: 4, height: 6, background: '#eee', borderRadius: 3 }}>
              <div style={{
                width: `${Math.min(pct, 100)}%`, height: '100%',
                background: t.cor, borderRadius: 3,
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, color: '#666' }}>
              <span>{st.volumeAlocadoT.toFixed(0)}/{t.capacidadeT.toFixed(0)} t</span>
              <span>{st.numLotes} lotes</span>
            </div>
            <div style={{ color: '#666' }}>blend prot: {st.blendProteina.toFixed(2)} (≥ {t.proteinaMin})</div>
          </div>
        );
      })}
    </div>
  );
}
```

E renderizar no aside (após os stats):

```tsx
{scenario && step && <TraderCards scenario={scenario} traderStats={step.traderStats} />}
```

- [ ] **Step 4: tsc + smoke test**

```bash
cd web/frontend && npx tsc -b --noEmit
```

```bash
make dev
```

Checklist:
1. Cards de trader aparecem durante treino
2. Barra de capacidade preenche/esvazia conforme GA realoca
3. Fundo do card fica vermelho quando over, amarelo quando under
4. Gráfico de fitness aparece e cresce ao longo das gerações

Parar dev.

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/views/MatchingView.tsx
git commit -m "feat(matching): trader status cards + fitness evolution chart"
```

---

## Task 13: Atualizar brainstorm doc + finalizar

**Files:**
- Modify: `docs/brainstorm-GA.md`

- [ ] **Step 1: Adicionar bloco no topo do brainstorm doc**

Inserir logo após o `# Brainstorm — ...` da linha 1, antes do `> Transferência...`:

```markdown
> **Status (atualizado 2026-05-06):** Etapa 1 em andamento na branch `feat/matching-marketplace`.
> Plano detalhado em [docs/superpowers/plans/2026-05-06-matching-marketplace-v1.md](superpowers/plans/2026-05-06-matching-marketplace-v1.md).
> Etapa 1 = single-objective + greedy baseline + cenários "Balanceado" e "Crise de Qualidade" (6×4).
> Próximas etapas (Hungarian, NSGA-II, modo 60×6, dados reais, IOSCO, modal, robustez) ficam para etapas posteriores.

```

- [ ] **Step 2: Smoke test final completo**

```bash
make build
cd web/server && go test ./matching/ -v
```

Expected: build passa + todos os testes Go passam.

```bash
make dev
```

End-to-end:
1. Abrir http://localhost:5173, ir em Matching · Soja
2. Carregar Balanceado → mapa renderiza
3. Treinar GA → ver animação completa, fitness subir, eventualmente convergir sem violações
4. Rodar baseline → comparar números
5. Trocar para Crise de Qualidade → recarregar cenário → treinar de novo, ver GA lutando com blends

- [ ] **Step 3: Commit**

```bash
git add docs/brainstorm-GA.md
git commit -m "docs(matching): mark etapa 1 in progress; link plan"
```

---

## Self-Review Notes

- **Não cobre:** estilos CSS dedicados — uso de inline styles (segue padrão de `TspView` que é grande mas usa muito inline também). Se virar bagunça, tarefa futura é extrair pra `matching.css`.
- **Não cobre:** validação de seed/inputs do usuário no UI — defaults fixos.
- **Não cobre:** persistência entre sessões — recarregar página perde o resultado (mesmo padrão do TSP).
- **Conhecido:** o backend serializa structs `FitnessBreakdown` e `TraderStats` sem json tags em alguns casos. Se aparecer minúsculas no JSON (`fitness` em vez de `Fitness`), adicionar tags Go e ajustar tipo TS. Confirmar no Step 7 da Task 7 ao olhar a saída do curl.
- **Conhecido:** distância haversine subestima rota real (estradas curvam). Etapa 2 reusa OSRM.
