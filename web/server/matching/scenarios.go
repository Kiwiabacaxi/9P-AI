package matching

import "math/rand"

const (
	ScenarioBalanceado        = "balanceado"
	ScenarioCriseQualidade    = "crise-qualidade"
	ScenarioCompradorDominante = "comprador-dominante"
	ScenarioPrecoAlto         = "preco-alto"
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
			ID: 3, Nome: "COFCO", Cor: "#a78bfa",
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

// buildCompradorDominante: Cargill com 60% da capacidade total, outros 3 pequenos.
// Mostra como o GA distribui sob pressão de monopolização.
func buildCompradorDominante(seed int64) Scenario {
	rng := rand.New(rand.NewSource(seed))
	prods := baseProducers()
	traders := baseTraders()
	// 60% Cargill, 40/3 ≈ 13.3% cada um dos outros
	traders[0].CapacidadeT = 18000
	traders[1].CapacidadeT = 4000
	traders[2].CapacidadeT = 4000
	traders[3].CapacidadeT = 4000

	lots := make([]Lot, 0, len(prods))
	for i, p := range prods {
		volume := 2500 + rng.Float64()*2500
		lots = append(lots, Lot{
			ID:           i,
			ProducerID:   p.ID,
			VolumeT:      volume,
			Proteina:     36.0 + rng.Float64()*3.0,
			Umidade:      12.0 + rng.Float64()*2.0,
			Impurezas:    0.5 + rng.Float64()*0.5,
			PrecoReserva: 130 + rng.Float64()*8,
			JanelaSemana: 1,
		})
	}
	return Scenario{
		ID:        ScenarioCompradorDominante,
		Nome:      "Comprador Dominante",
		Descricao: "Cargill detém 60% da capacidade — clássico do agro brasileiro",
		Producers: prods,
		Lots:      lots,
		Traders:   traders,
		PrecoBase: 138,
		PortLat:   PortSantosLat,
		PortLng:   PortSantosLng,
	}
}

// buildPrecoAlto: papel Santos disparou — traders pagam mais, mas produtores também
// querem mais (preço de reserva sobe), apertando margens.
func buildPrecoAlto(seed int64) Scenario {
	rng := rand.New(rand.NewSource(seed))
	prods := baseProducers()
	traders := baseTraders()
	// traders pagam mais
	for i := range traders {
		traders[i].PrecoMaximo += 15 // todos +R$15/saca
	}

	lots := make([]Lot, 0, len(prods))
	for i, p := range prods {
		volume := 2500 + rng.Float64()*2500
		lots = append(lots, Lot{
			ID:           i,
			ProducerID:   p.ID,
			VolumeT:      volume,
			Proteina:     36.0 + rng.Float64()*3.0,
			Umidade:      12.0 + rng.Float64()*2.0,
			Impurezas:    0.5 + rng.Float64()*0.5,
			PrecoReserva: 145 + rng.Float64()*8, // produtores querem 145..153 (vs 130..138 normal)
			JanelaSemana: 1,
		})
	}
	return Scenario{
		ID:        ScenarioPrecoAlto,
		Nome:      "Preço Alto",
		Descricao: "Mercado apertado — preços de reserva e de teto subiram, margens estreitas",
		Producers: prods,
		Lots:      lots,
		Traders:   traders,
		PrecoBase: 152, // base maior também
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
	case ScenarioCompradorDominante:
		return buildCompradorDominante(seed), true
	case ScenarioPrecoAlto:
		return buildPrecoAlto(seed), true
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
		{ID: ScenarioCompradorDominante, Nome: "Comprador Dominante", Descricao: "Cargill com 60% da capacidade total"},
		{ID: ScenarioPrecoAlto, Nome: "Preço Alto", Descricao: "Mercado apertado — preços e reservas elevados"},
	}
}
