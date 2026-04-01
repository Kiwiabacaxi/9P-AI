package timeseries

// =============================================================================
// Dados de mercado — busca via Yahoo Finance (Python subprocess) + preparação
//
// Aula 08: "Usaremos uma série temporal do Yahoo Finance.
// No eixo X teremos o dia, no eixo Y o valor de fechamento.
// Use 6 meses de 2025 para treinar, 7 dias para validar."
//
// Janela deslizante (sliding window):
//   Input:  [preço(t-N), preço(t-N+1), ..., preço(t-1)]
//   Output: preço(t)
//
// Normalização Min-Max para [0,1] — essencial para MLP com tanh/sigmoid.
// =============================================================================

import (
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"runtime"
	"strings"
)

// StockData contém os dados brutos do Yahoo Finance
type StockData struct {
	Ticker  string    `json:"ticker"`
	Dates   []string  `json:"dates"`
	Close   []float64 `json:"close"`
	Open    []float64 `json:"open"`
	High    []float64 `json:"high"`
	Low     []float64 `json:"low"`
	Volume  []float64 `json:"volume"`
}

// NormalizedData contém os dados preparados para treinamento
type NormalizedData struct {
	TrainX   [][]float64 // [nTrain][windowSize] — inputs normalizados
	TrainY   []float64   // [nTrain] — targets normalizados
	ValidX   [][]float64 // [nValid][windowSize]
	ValidY   []float64   // [nValid]
	MinPrice float64     // para desnormalizar
	MaxPrice float64
	Dates    []string    // datas correspondentes aos targets (após window)
	AllClose []float64   // preços originais (para gráfico)
}

// TimeSeriesPoint representa um ponto no gráfico de previsão
type TimeSeriesPoint struct {
	Data    string  `json:"data"`
	Preco   float64 `json:"preco"`   // preço real
	Predito float64 `json:"predito"` // preço predito pela rede
}

// FetchStockData busca dados do Yahoo Finance usando Python/yfinance
func FetchStockData(ticker, period string) (*StockData, error) {
	// Sanitizar ticker para evitar injeção
	ticker = strings.ReplaceAll(ticker, `"`, "")
	ticker = strings.ReplaceAll(ticker, `'`, "")
	period = strings.ReplaceAll(period, `"`, "")

	script := fmt.Sprintf(`
import yfinance as yf, json, sys
t = yf.Ticker("%s")
h = t.history(period="%s")
if h.empty:
    print("{}", file=sys.stderr)
    sys.exit(1)
data = {"ticker":"%s","dates":[],"close":[],"open":[],"high":[],"low":[],"volume":[]}
for d, r in h.iterrows():
    data["dates"].append(d.strftime("%%Y-%%m-%%d"))
    data["close"].append(round(r["Close"], 4))
    data["open"].append(round(r["Open"], 4))
    data["high"].append(round(r["High"], 4))
    data["low"].append(round(r["Low"], 4))
    data["volume"].append(float(r["Volume"]))
json.dump(data, sys.stdout)
`, ticker, period, ticker)

	// Encontrar Python com yfinance
	pythonCmd := findPython()
	cmd := exec.Command(pythonCmd, "-c", script)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("erro ao buscar dados: %w (python=%s)", err, pythonCmd)
	}

	var data StockData
	if err := json.Unmarshal(output, &data); err != nil {
		return nil, fmt.Errorf("erro ao parsear dados: %w", err)
	}
	if len(data.Close) == 0 {
		return nil, fmt.Errorf("nenhum dado retornado para %s", ticker)
	}
	return &data, nil
}

// findPython encontra o executável Python com yfinance instalado
func findPython() string {
	// Tentar caminhos comuns
	candidates := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		// Caminho específico onde yfinance foi instalado
		candidates = append([]string{
			`C:\Users\carlo\AppData\Local\Programs\Python\Python313\python.exe`,
		}, candidates...)
	}
	for _, p := range candidates {
		cmd := exec.Command(p, "-c", "import yfinance; print('ok')")
		if err := cmd.Run(); err == nil {
			return p
		}
	}
	return "python" // fallback
}

// PrepareData normaliza e cria janelas deslizantes para treino/validação
func PrepareData(closes []float64, dates []string, windowSize, validDays int) NormalizedData {
	n := len(closes)

	// Encontrar min/max para normalização
	minP, maxP := closes[0], closes[0]
	for _, p := range closes {
		if p < minP {
			minP = p
		}
		if p > maxP {
			maxP = p
		}
	}
	rangeP := maxP - minP
	if rangeP < 0.0001 {
		rangeP = 1 // evitar divisão por zero
	}

	// Normalizar preços para [0, 1]
	norm := make([]float64, n)
	for i, p := range closes {
		norm[i] = (p - minP) / rangeP
	}

	// Criar pares (janela → próximo dia)
	// Total de pares: n - windowSize
	totalPairs := n - windowSize
	if totalPairs <= 0 {
		return NormalizedData{MinPrice: minP, MaxPrice: maxP}
	}

	allX := make([][]float64, totalPairs)
	allY := make([]float64, totalPairs)
	pairDates := make([]string, totalPairs)

	for i := range totalPairs {
		window := make([]float64, windowSize)
		copy(window, norm[i:i+windowSize])
		allX[i] = window
		allY[i] = norm[i+windowSize]
		pairDates[i] = dates[i+windowSize]
	}

	// Separar treino e validação
	trainEnd := totalPairs - validDays
	if trainEnd < 1 {
		trainEnd = totalPairs - 1
	}

	return NormalizedData{
		TrainX:   allX[:trainEnd],
		TrainY:   allY[:trainEnd],
		ValidX:   allX[trainEnd:],
		ValidY:   allY[trainEnd:],
		MinPrice: minP,
		MaxPrice: maxP,
		Dates:    pairDates,
		AllClose: closes,
	}
}

// Desnormalizar converte valor [0,1] de volta para preço real
func Desnormalizar(normalized, minP, maxP float64) float64 {
	return normalized*(maxP-minP) + minP
}

// CalcularMetricas calcula MSE, RMSE e MAE entre preços reais e preditos
func CalcularMetricas(reais, preditos []float64) (mse, rmse, mae float64) {
	n := len(reais)
	if n == 0 {
		return
	}
	for i := range n {
		d := reais[i] - preditos[i]
		mse += d * d
		mae += math.Abs(d)
	}
	mse /= float64(n)
	rmse = math.Sqrt(mse)
	mae /= float64(n)
	return
}
