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
	PrecoReserva  float64 `json:"precoReserva"`
	JanelaSemana  int     `json:"janelaSemana"`
}

// Trader é um comprador (Cargill, Bunge, etc.) com hub geográfico.
type Trader struct {
	ID             int     `json:"id"`
	Nome           string  `json:"nome"`
	Cor            string  `json:"cor"`
	HubMunicipio   string  `json:"hubMunicipio"`
	HubLat         float64 `json:"hubLat"`
	HubLng         float64 `json:"hubLng"`
	CapacidadeT    float64 `json:"capacidadeT"`
	ProteinaMin    float64 `json:"proteinaMin"`
	UmidadeMax     float64 `json:"umidadeMax"`
	ImpurezasMax   float64 `json:"impurezasMax"`
	PrecoMaximo    float64 `json:"precoMaximo"`
	JanelaSemana   int     `json:"janelaSemana"`
}

// Scenario agrega um setup completo do problema.
type Scenario struct {
	ID          string     `json:"id"`
	Nome        string     `json:"nome"`
	Descricao   string     `json:"descricao"`
	Producers   []Producer `json:"producers"`
	Lots        []Lot      `json:"lots"`
	Traders     []Trader   `json:"traders"`
	PrecoBase   float64    `json:"precoBase"`
	PortLat     float64    `json:"portLat"`
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
	LambdaLog       float64 `json:"lambdaLog"`
	LambdaQual      float64 `json:"lambdaQual"`
	MBig            float64 `json:"mBig"`
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
