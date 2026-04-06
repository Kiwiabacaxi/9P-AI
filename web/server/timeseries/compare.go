package timeseries

// =============================================================================
// Comparação de modelos — orquestração e tipos compartilhados
// =============================================================================

// CompareModelResult é o resultado de um modelo individual na comparação
type CompareModelResult struct {
	Modelo string           `json:"modelo"`
	Result TimeSeriesResult `json:"result"`
	Cor    string           `json:"cor"`
	Erro   string           `json:"erro,omitempty"` // erro se o modelo falhou
}

// ModelColors mapeia modelo → cor hex para o gráfico
var ModelColors = map[string]string{
	"SMA":          "#00ff00",
	"EMA":          "#77ff61",
	"ARIMA":        "#ffaa00",
	"RandomForest": "#ff8800",
	"XGBoost":      "#ff6600",
	"MLP":          "#00fbfb",
	"LSTM":         "#ff6ec7",
	"GRU":          "#aa66ff",
	"BiLSTM":       "#ff44aa",
	"Seq2Seq":      "#ff3333",
	"Prophet":      "#4488ff",
	"ProphetLike":  "#66bbff",
}

// AvailableModels lista todos os modelos disponíveis por categoria
var AvailableModels = []struct {
	Nome      string `json:"nome"`
	Categoria string `json:"categoria"`
	Cor       string `json:"cor"`
	NeedsPython bool `json:"needsPython"`
}{
	{"SMA", "Baselines", "#00ff00", false},
	{"EMA", "Baselines", "#77ff61", false},
	{"ARIMA", "Estatísticos", "#ffaa00", false},
	{"MLP", "Deep Learning", "#00fbfb", false},
	{"LSTM", "Deep Learning", "#ff6ec7", false},
	{"GRU", "Deep Learning", "#aa66ff", false},
	{"BiLSTM", "Deep Learning", "#ff44aa", false},
	{"Seq2Seq", "Deep Learning", "#ff3333", false},
	{"ProphetLike", "Prophet", "#66bbff", false},
	{"RandomForest", "Machine Learning", "#ff8800", true},
	{"XGBoost", "Machine Learning", "#ff6600", true},
	{"Prophet", "Prophet", "#4488ff", true},
}
