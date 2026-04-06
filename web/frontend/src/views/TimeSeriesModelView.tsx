import { useState, useRef, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import LogChart from '../components/viz/LogChart';
import TimeSeriesChart from '../components/viz/TimeSeriesChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip } from 'recharts';
import type { TsStockData, TsResult } from '../api/types';

// ---------------------------------------------------------------------------
// Model metadata — adapts the generic view per model
// ---------------------------------------------------------------------------

interface ModelInfo {
  title: string;
  subtitle: string;
  description: string;
  color: string;
  hasTraining: boolean;  // SMA/EMA/ARIMA/ProphetLike are instant
  needsPython: boolean;  // RandomForest/XGBoost/Prophet need Python
  showHidden: boolean;   // show hidden neurons config
  showCiclos: boolean;   // show max cycles config
  showLR: boolean;       // show learning rate config
  category: string;
}

const MODEL_INFO: Record<string, ModelInfo> = {
  SMA: {
    title: 'SMA', subtitle: 'Simple Moving Average', category: 'Baseline',
    description: 'Predição = média aritmética dos últimos N dias. O baseline mais simples — se um modelo não bater o SMA, algo está errado. Sem treinamento, resultado instantâneo.',
    color: '#00ff00', hasTraining: false, needsPython: false,
    showHidden: false, showCiclos: false, showLR: false,
  },
  EMA: {
    title: 'EMA', subtitle: 'Exponential Moving Average', category: 'Baseline',
    description: 'Média ponderada que dá mais peso aos dados recentes. α = 2/(N+1). Melhor que SMA para capturar tendências, mas ainda é baseline.',
    color: '#77ff61', hasTraining: false, needsPython: false,
    showHidden: false, showCiclos: false, showLR: false,
  },
  ARIMA: {
    title: 'ARIMA', subtitle: 'AutoRegressive Integrated Moving Average', category: 'Estatístico',
    description: 'Modelo AR(p,1,0): diferenciação para tornar série estacionária + coeficientes autoregressivos estimados por Yule-Walker (gonum/mat). Clássico em econometria.',
    color: '#ffaa00', hasTraining: false, needsPython: false,
    showHidden: false, showCiclos: false, showLR: false,
  },
  MLP: {
    title: 'MLP', subtitle: 'Multi-Layer Perceptron', category: 'Deep Learning',
    description: 'Rede feedforward com sliding window + backpropagation. A mesma arquitetura da Aula 08 — windowSize entradas → hidden → 1 saída.',
    color: '#00fbfb', hasTraining: true, needsPython: false,
    showHidden: true, showCiclos: true, showLR: true,
  },
  LSTM: {
    title: 'LSTM', subtitle: 'Long Short-Term Memory', category: 'Deep Learning',
    description: '4 gates (forget, input, cell, output) + BPTT com gradient clipping. Captura dependências de longo prazo que o MLP não consegue. Hochreiter & Schmidhuber 1997.',
    color: '#ff6ec7', hasTraining: true, needsPython: false,
    showHidden: true, showCiclos: true, showLR: true,
  },
  GRU: {
    title: 'GRU', subtitle: 'Gated Recurrent Unit', category: 'Deep Learning',
    description: 'Versão simplificada do LSTM com 2 gates (reset + update). Menos parâmetros, treina mais rápido, performance comparável. Cho et al. 2014.',
    color: '#aa66ff', hasTraining: true, needsPython: false,
    showHidden: true, showCiclos: true, showLR: true,
  },
  BiLSTM: {
    title: 'BiLSTM', subtitle: 'Bidirectional LSTM', category: 'Deep Learning',
    description: 'Dois LSTMs processando a sequência em ambas as direções (forward + backward). Captura contexto passado E futuro, útil para classificação de sequências.',
    color: '#ff44aa', hasTraining: true, needsPython: false,
    showHidden: true, showCiclos: true, showLR: true,
  },
  Seq2Seq: {
    title: 'Seq2Seq', subtitle: 'Sequence-to-Sequence (Encoder-Decoder)', category: 'Deep Learning',
    description: 'LSTM encoder comprime o histórico num vetor de contexto, LSTM decoder gera previsão multi-step. Primeira arquitetura pensada para horizontes longos. Sutskever et al. 2014.',
    color: '#ff3333', hasTraining: true, needsPython: false,
    showHidden: true, showCiclos: true, showLR: true,
  },
  ProphetLike: {
    title: 'Prophet-like', subtitle: 'Decomposição Tendência + Sazonalidade (Go)', category: 'Prophet',
    description: 'Versão simplificada do Prophet em Go puro: tendência por regressão linear + sazonalidade semanal (5 dias úteis). Didático, não é o Prophet real.',
    color: '#66bbff', hasTraining: false, needsPython: false,
    showHidden: false, showCiclos: false, showLR: false,
  },
  RandomForest: {
    title: 'Random Forest', subtitle: 'Ensemble de Árvores de Decisão (sklearn)', category: 'Machine Learning',
    description: '100 árvores de decisão treinadas em subsets aleatórios dos dados. Robusto contra overfitting. Requer Python + scikit-learn.',
    color: '#ff8800', hasTraining: false, needsPython: true,
    showHidden: false, showCiclos: false, showLR: false,
  },
  XGBoost: {
    title: 'XGBoost', subtitle: 'Extreme Gradient Boosting', category: 'Machine Learning',
    description: 'Gradient boosting otimizado — treina árvores sequencialmente, cada uma corrigindo os erros da anterior. Estado da arte em competições. Requer Python + xgboost.',
    color: '#ff6600', hasTraining: false, needsPython: true,
    showHidden: false, showCiclos: false, showLR: false,
  },
  Prophet: {
    title: 'Prophet', subtitle: 'Meta Prophet (Python)', category: 'Prophet',
    description: 'Decomposição bayesiana de séries temporais da Meta. Detecta tendência + sazonalidade + feriados automaticamente. Requer Python + prophet.',
    color: '#4488ff', hasTraining: false, needsPython: true,
    showHidden: false, showCiclos: false, showLR: false,
  },
};

// ---------------------------------------------------------------------------
// Generic Model View
// ---------------------------------------------------------------------------

export default function TimeSeriesModelView({ modelo }: { modelo: string }) {
  const info = MODEL_INFO[modelo];
  const { show } = useToast();

  // Config
  const [ticker, setTicker] = useState('COGN3.SA');
  const [windowSize, setWindowSize] = useState(5);
  const [hiddenSize, setHiddenSize] = useState(16);
  const [alfa, setAlfa] = useState(0.005);
  const [maxCiclo, setMaxCiclo] = useState(2000);
  const [forecastDays, setForecastDays] = useState(7);
  const [validDays] = useState(7);

  // State
  const [stockData, setStockData] = useState<TsStockData | null>(null);
  const [fetching, setFetching] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<TsResult | null>(null);
  const [mseHist, setMseHist] = useState<number[]>([]);
  const sseCleanup = useRef<(() => void) | null>(null);

  // Log
  const [logLines, setLogLines] = useState<{ msg: string; cls: string }[]>([
    { msg: `// ${info?.title || modelo} — busque dados para começar`, cls: 'dim' },
  ]);
  const [progress, setProgress] = useState(0);

  const addLog = useCallback((msg: string, cls: string = '') => {
    setLogLines(prev => [{ msg, cls }, ...prev].slice(0, 80));
  }, []);

  if (!info) {
    return <div style={{ padding: 40, fontFamily: 'var(--font-mono)', color: 'var(--pink)' }}>Modelo "{modelo}" não encontrado</div>;
  }

  const validStart = result ? result.pontos.length - (result.pontosValid?.length || 0) : undefined;

  // Fetch data
  const handleFetch = async () => {
    setFetching(true);
    try {
      const data = await apiPost<TsStockData>('/timeseries/fetch-data', { ticker, period: '6mo' });
      setStockData(data);
      addLog(`✓ ${data.dates.length} dias carregados`, 'ok');
      show(`${ticker}: ${data.dates.length} dias`);
    } catch (e) {
      addLog('// erro: ' + (e instanceof Error ? e.message : String(e)), 'err');
    }
    setFetching(false);
  };

  // Train
  const handleTrain = async () => {
    if (training || !stockData) return;
    setTraining(true);
    setResult(null);
    setMseHist([]);
    setProgress(0);
    addLog(`// treinando ${info.title}...`, 'dim');

    // Send config
    await apiPost('/timeseries/config', {
      ticker, windowSize, hiddenSize, alfa, maxCiclo,
      ativacao: 'tanh', validDays, forecastDays, validPct: 0,
    }).catch(() => {});

    if (info.hasTraining) {
      // SSE-based training (DL models)
      const cleanup = apiSSE('/timeseries/train-model', {
        body: { modelo },
        onMessage(data) {
          const step = data as { ciclo: number; mseTreino: number; mseValid: number };
          setProgress(Math.min((step.ciclo / maxCiclo) * 100, 99));
          if (step.ciclo % 500 === 0) {
            addLog(`ciclo ${step.ciclo} · MSE ${step.mseTreino.toFixed(6)}`, 'ok');
          }
        },
        onDone(data) {
          const res = data as TsResult;
          setResult(res);
          setMseHist(res.mseHistorico || []);
          setProgress(100);
          setTraining(false);
          addLog(`✓ RMSE: R$${res.rmseFinal.toFixed(4)} · Pred: R$${res.predicaoAmanha.toFixed(2)}`, 'ok');
          show(`${info.title}: RMSE R$${res.rmseFinal.toFixed(4)}`);
        },
        onError() {
          addLog('// erro de conexão', 'err');
          setTraining(false);
        },
      });
      sseCleanup.current = cleanup;
    } else {
      // Instant models (SMA, EMA, ARIMA, ProphetLike, Python models)
      try {
        const res = await apiPost<TsResult>('/timeseries/train-model', { modelo });
        setResult(res);
        setMseHist(res.mseHistorico || []);
        setProgress(100);
        addLog(`✓ RMSE: R$${res.rmseFinal.toFixed(4)} · Pred: R$${res.predicaoAmanha.toFixed(2)}`, 'ok');
        show(`${info.title}: RMSE R$${res.rmseFinal.toFixed(4)}`);
      } catch (e) {
        addLog('// erro: ' + (e instanceof Error ? e.message : String(e)), 'err');
      }
      setTraining(false);
    }
  };

  // Reset
  const handleReset = () => {
    sseCleanup.current?.();
    setResult(null); setMseHist([]); setStockData(null); setTraining(false); setProgress(0);
    setLogLines([{ msg: `// ${info.title} — busque dados para começar`, cls: 'dim' }]);
  };

  // Price chart data
  const priceChartData = stockData ? stockData.dates.map((d, i) => ({
    data: d.slice(5), preco: stockData.close[i],
  })) : [];

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title" style={{ color: info.color }}>
            {info.title} <span style={{ color: 'var(--on-surface)' }}>{info.subtitle}</span>
          </div>
          <div className="page-sub">{info.category} &mdash; Séries Temporais</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={training}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training || !stockData}>
            {training && <span className="spin" />}
            {info.hasTraining ? 'TREINAR' : 'EXECUTAR'}
          </button>
        </div>
      </div>

      {/* Config */}
      <div className={info.showHidden ? 'grid-3' : 'grid-2'} style={{ marginBottom: 24 }}>
        <Card title="Ação">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input type="text" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
              style={{ background: 'var(--surface-low)', border: '1px solid var(--border)', color: 'var(--on-surface)',
                fontFamily: 'var(--font-mono)', fontSize: 12, padding: '6px 10px', flex: 1, fontWeight: 700 }} />
            <button className="btn btn-primary" style={{ fontSize: 10, padding: '6px 12px' }}
              onClick={handleFetch} disabled={fetching || training}>
              {fetching ? '...' : 'BUSCAR'}
            </button>
          </div>
          {stockData && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--primary-glow)', marginTop: 6 }}>
              {stockData.dates.length} dias · R${Math.min(...stockData.close).toFixed(2)}–R${Math.max(...stockData.close).toFixed(2)}
            </div>
          )}
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            {[3, 5, 10, 20].map(w => (
              <button key={w} className={`porta-chip${windowSize === w ? ' selected' : ''}`}
                onClick={() => setWindowSize(w)} disabled={training}>{w}d janela</button>
            ))}
          </div>
        </Card>

        {info.showHidden && (
          <Card title="Neurônios · Ciclos · LR">
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              {[8, 16, 32].map(h => (
                <button key={h} className={`porta-chip${hiddenSize === h ? ' selected' : ''}`}
                  onClick={() => setHiddenSize(h)} disabled={training}>{h}n</button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              {[1000, 2000, 5000].map(c => (
                <button key={c} className={`porta-chip${maxCiclo === c ? ' selected' : ''}`}
                  onClick={() => setMaxCiclo(c)} disabled={training}>{c >= 1000 ? `${c/1000}k` : c}</button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[0.001, 0.005, 0.01].map(a => (
                <button key={a} className={`porta-chip${alfa === a ? ' selected' : ''}`}
                  onClick={() => setAlfa(a)} disabled={training}>{a}</button>
              ))}
            </div>
          </Card>
        )}

        <Card title="Previsão Futura">
          <div style={{ display: 'flex', gap: 6 }}>
            {[3, 5, 7, 14].map(d => (
              <button key={d} className={`porta-chip${forecastDays === d ? ' selected' : ''}`}
                onClick={() => setForecastDays(d)} disabled={training}>{d} dias</button>
            ))}
          </div>
          {info.needsPython && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--pink)', marginTop: 8 }}>
              Requer Python + libs instaladas
            </div>
          )}
        </Card>
      </div>

      {/* Raw price chart (before training) */}
      {stockData && !result && (
        <Card title={`Histórico — ${ticker}`} style={{ marginBottom: 24 }}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={priceChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
              <XAxis dataKey="data" stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} interval={Math.max(0, Math.floor(priceChartData.length / 8) - 1)} />
              <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }} />
              <Line type="monotone" dataKey="preco" stroke={info.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <MetricCard title="RMSE" value={result ? `R$${result.rmseFinal.toFixed(4)}` : '--'}
          label="erro quadrático médio (raiz)" color="cyan" pulse={training} />
        <MetricCard title="MAE" value={result ? `R$${result.maeFinal.toFixed(4)}` : '--'}
          label="erro absoluto médio" color="green" />
        <MetricCard title="Predição Amanhã"
          value={result ? `R$${result.predicaoAmanha.toFixed(2)}` : '--'}
          label={result ? `${ticker} · próximo dia útil` : 'aguardando'}
          color={result ? 'green' : undefined}
          valueStyle={result ? { fontSize: 28, color: info.color } : undefined} />
      </div>

      {/* Log + MSE */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <Card title="Log">
          <div className="progress-wrap">
            <div className="progress-fill" style={{ width: `${progress}%`, background: info.color }} />
          </div>
          <div className="log-panel">
            {logLines.map((line, i) => (
              <div key={i} className={`log-line ${line.cls}`}>{line.msg}</div>
            ))}
          </div>
        </Card>
        <Card title="Curva de MSE">
          {mseHist.length > 0 ? <LogChart data={mseHist} color={info.color} /> : (
            <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555' }}>
              {info.hasTraining ? 'MSE aparece após treino' : 'Modelo sem treinamento iterativo'}
            </div>
          )}
        </Card>
      </div>

      {/* Prediction chart */}
      {result && (
        <Card title="Preço Real vs Predição" style={{ marginBottom: 24 }}>
          <TimeSeriesChart pontos={result.pontos} forecast={result.forecast} validStart={validStart} height={300}
            showConfidence={info.hasTraining} />
        </Card>
      )}

      {/* Validation table */}
      {result && result.pontosValid?.length > 0 && (
        <Card title="Validação" style={{ marginBottom: 24 }}>
          <table className="data-table" style={{ fontSize: 11 }}>
            <thead><tr><th>Data</th><th>Real</th><th>Predição</th><th>Erro</th><th>%</th></tr></thead>
            <tbody>
              {result.pontosValid.map((p, i) => {
                const erro = Math.abs(p.preco - p.predito);
                const pct = p.preco > 0 ? (erro / p.preco) * 100 : 0;
                return (
                  <tr key={i}>
                    <td className="td-cyan">{p.data}</td>
                    <td>R${p.preco.toFixed(2)}</td>
                    <td style={{ color: info.color }}>R${p.predito.toFixed(2)}</td>
                    <td>R${erro.toFixed(4)}</td>
                    <td style={{ color: pct > 3 ? 'var(--pink)' : 'var(--primary-glow)' }}>{pct.toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      {/* Forecast table */}
      {result && result.forecast && result.forecast.length > 0 && (
        <Card title={`Previsão Futura — ${result.forecast.length} dias`} style={{ marginBottom: 24 }}>
          <table className="data-table" style={{ fontSize: 11 }}>
            <thead><tr><th>Dia</th><th>Predição</th><th>Intervalo</th></tr></thead>
            <tbody>
              {result.forecast.map(f => (
                <tr key={f.dia}>
                  <td className="td-cyan">D+{f.dia}</td>
                  <td style={{ color: info.color, fontWeight: 700 }}>R${f.predito.toFixed(2)}</td>
                  <td>R${f.lower.toFixed(2)} — R${f.upper.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* Description */}
      <Card title="Sobre o Modelo">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div style={{ color: info.color, fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
            {info.title} — {info.subtitle}
          </div>
          <div>{info.description}</div>
          <div style={{ marginTop: 8, color: '#888' }}>
            Categoria: {info.category} · Cor: <span style={{ color: info.color }}>{info.color}</span>
            {info.needsPython && ' · Requer Python'}
            {info.hasTraining && ' · Treinamento iterativo com backpropagation'}
          </div>
        </div>
      </Card>
    </div>
  );
}
