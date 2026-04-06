package timeseries

// =============================================================================
// ML Python — Wrappers para modelos Python via subprocess
//
// Random Forest (sklearn), XGBoost, Prophet (Meta) — todos executados via
// Python subprocess. O Go serializa dados → Python treina → retorna JSON.
//
// Padrão: mesmo que usamos para yfinance em data.go.
// =============================================================================

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

// pythonTimeSeriesScript gera o script Python para um modelo específico
func pythonTimeSeriesScript(modelName string, closes []float64, dates []string, windowSize, forecastDays, validDays int) string {
	// Serializar dados como listas Python
	closesStr := fmt.Sprintf("%v", closes)
	closesStr = strings.ReplaceAll(closesStr, " ", ",")

	datesStr := "["
	for i, d := range dates {
		if i > 0 { datesStr += "," }
		datesStr += fmt.Sprintf(`"%s"`, d)
	}
	datesStr += "]"

	base := fmt.Sprintf(`
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np

closes = np.array(%s)
dates = %s
window = %d
forecast_days = %d
valid_days = %d
n = len(closes)

# Criar features sliding window
X, y, d_out = [], [], []
for i in range(window, n):
    X.append(closes[i-window:i])
    y.append(closes[i])
    d_out.append(dates[i] if i < len(dates) else "")
X = np.array(X)
y = np.array(y)

train_end = len(X) - valid_days
X_train, y_train = X[:train_end], y[:train_end]
X_valid, y_valid = X[train_end:], y[train_end:]

start = time.time()
`, closesStr, datesStr, windowSize, forecastDays, validDays)

	var modelCode string
	switch modelName {
	case "RandomForest":
		modelCode = `
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred_all = model.predict(X)
y_pred_valid = model.predict(X_valid)
`
	case "XGBoost":
		modelCode = `
try:
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    y_pred_all = model.predict(X)
    y_pred_valid = model.predict(X_valid)
except ImportError:
    print('{"error":"xgboost not installed. pip install xgboost"}', file=sys.stderr)
    sys.exit(1)
`
	case "Prophet":
		modelCode = `
try:
    from prophet import Prophet
    import pandas as pd
    # Prophet precisa de df com colunas 'ds' e 'y'
    df = pd.DataFrame({'ds': pd.to_datetime(dates[:n]), 'y': closes})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df.iloc[:n-valid_days])
    # Predições para todo o período
    future = model.make_future_dataframe(periods=valid_days+forecast_days, freq='B')
    forecast = model.predict(future)
    # Mapear predições para o formato esperado
    y_pred_all = forecast['yhat'].values[:n-window] if len(forecast) >= n-window else forecast['yhat'].values
    y_pred_valid = forecast['yhat'].values[n-valid_days-window:n-window] if len(forecast) >= n-window else []
    # Forecast futuro
    future_preds = forecast['yhat'].values[n-window:n-window+forecast_days] if len(forecast) >= n-window+forecast_days else forecast['yhat'].values[-forecast_days:]
    future_upper = forecast['yhat_upper'].values[n-window:n-window+forecast_days] if len(forecast) >= n-window+forecast_days else forecast['yhat_upper'].values[-forecast_days:]
    future_lower = forecast['yhat_lower'].values[n-window:n-window+forecast_days] if len(forecast) >= n-window+forecast_days else forecast['yhat_lower'].values[-forecast_days:]
    has_prophet_forecast = True
except ImportError:
    print('{"error":"prophet not installed. pip install prophet"}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error":str(e)}), file=sys.stderr)
    sys.exit(1)
`
	default:
		return ""
	}

	output := fmt.Sprintf(`
elapsed = (time.time() - start) * 1000

# Métricas na validação
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_valid, y_pred_valid[:len(y_valid)]) if len(y_pred_valid) > 0 else 0
rmse = mse ** 0.5
mae = mean_absolute_error(y_valid, y_pred_valid[:len(y_valid)]) if len(y_pred_valid) > 0 else 0

# Pontos (treino + validação)
pontos = []
for i in range(min(len(y), len(y_pred_all))):
    pontos.append({"data": d_out[i] if i < len(d_out) else "", "preco": float(y[i]), "predito": float(y_pred_all[i])})

pontos_valid = []
for i in range(len(y_valid)):
    idx = train_end + i
    pontos_valid.append({"data": d_out[idx] if idx < len(d_out) else "", "preco": float(y_valid[i]), "predito": float(y_pred_valid[i]) if i < len(y_pred_valid) else 0})

# Forecast
forecast_pts = []
if '%s' == 'Prophet' and 'has_prophet_forecast' in dir() and has_prophet_forecast:
    for d in range(min(forecast_days, len(future_preds))):
        forecast_pts.append({"dia": d+1, "predito": float(future_preds[d]), "upper": float(future_upper[d]), "lower": float(future_lower[d])})
else:
    # Forecast recursivo para RF/XGBoost
    last_window = closes[-window:].tolist()
    for d in range(forecast_days):
        pred = model.predict(np.array([last_window[-window:]]))[0]
        spread = rmse * ((d+1)**0.5)
        forecast_pts.append({"dia": d+1, "predito": float(pred), "upper": float(pred+spread), "lower": float(pred-spread)})
        last_window.append(pred)

pred_amanha = forecast_pts[0]["predito"] if forecast_pts else 0

result = {
    "ciclos": 0,
    "mseFinal": float(mse),
    "rmseFinal": float(rmse),
    "maeFinal": float(mae),
    "mseHistorico": [],
    "pontos": pontos,
    "pontosValid": pontos_valid,
    "predicaoAmanha": float(pred_amanha),
    "forecast": forecast_pts,
    "ticker": "",
    "tempoMs": int(elapsed)
}
json.dump(result, sys.stdout)
`, modelName)

	return base + modelCode + output
}

// TreinarPythonModel executa um modelo Python e retorna o resultado
func TreinarPythonModel(modelName string, cfg Config, stockData *StockData) (TimeSeriesResult, error) {
	script := pythonTimeSeriesScript(modelName, stockData.Close, stockData.Dates, cfg.WindowSize, cfg.ForecastDays, cfg.ValidDays)
	if script == "" {
		return TimeSeriesResult{}, fmt.Errorf("modelo desconhecido: %s", modelName)
	}

	pythonCmd := findPython()
	cmd := exec.Command(pythonCmd, "-c", script)
	output, err := cmd.Output()
	if err != nil {
		// Tentar ler stderr para mensagem de erro
		if exitErr, ok := err.(*exec.ExitError); ok {
			return TimeSeriesResult{}, fmt.Errorf("%s falhou: %s", modelName, string(exitErr.Stderr))
		}
		return TimeSeriesResult{}, fmt.Errorf("%s falhou: %w", modelName, err)
	}

	var result TimeSeriesResult
	if err := json.Unmarshal(output, &result); err != nil {
		return TimeSeriesResult{}, fmt.Errorf("erro ao parsear resultado de %s: %w", modelName, err)
	}
	result.Ticker = cfg.Ticker
	return result, nil
}
