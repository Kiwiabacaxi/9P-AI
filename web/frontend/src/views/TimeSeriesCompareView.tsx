import { useState, useRef, useCallback, useEffect } from 'react';
import Card from '../components/shared/Card';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip, Legend } from 'recharts';
import type { TsStockData, TsCompareModelResult, TsAvailableModel } from '../api/types';

const DEFAULT_MODELS = ['SMA', 'EMA', 'ARIMA', 'MLP', 'LSTM', 'ProphetLike'];

export default function TimeSeriesCompareView() {
  const { show } = useToast();

  // Config
  const [ticker, setTicker] = useState('COGN3.SA');
  const [windowSize, setWindowSize] = useState(5);
  const [hiddenSize, setHiddenSize] = useState(16);
  const [alfa] = useState(0.005);
  const [maxCiclo, setMaxCiclo] = useState(2000);
  const [forecastDays, setForecastDays] = useState(7);
  const [validDays] = useState(7);

  // Data
  const [stockData, setStockData] = useState<TsStockData | null>(null);
  const [fetching, setFetching] = useState(false);

  // Models
  const [availableModels, setAvailableModels] = useState<TsAvailableModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set(DEFAULT_MODELS));

  // Results
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<Map<string, TsCompareModelResult>>(new Map());
  const [activeModel, setActiveModel] = useState('');
  const sseCleanup = useRef<(() => void) | null>(null);

  // Log
  const [logLines, setLogLines] = useState<{ msg: string; cls: string }[]>([
    { msg: '// selecione modelos e clique COMPARAR', cls: 'dim' },
  ]);

  useEffect(() => {
    apiGet<TsAvailableModel[]>('/timeseries/available-models').then(setAvailableModels).catch(() => {});
  }, []);

  const addLog = useCallback((msg: string, cls: string = '') => {
    setLogLines(prev => [{ msg, cls }, ...prev].slice(0, 100));
  }, []);

  const toggleModel = (nome: string) => {
    setSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(nome)) next.delete(nome);
      else next.add(nome);
      return next;
    });
  };

  // Fetch data
  const handleFetch = useCallback(async () => {
    setFetching(true);
    try {
      const data = await apiPost<TsStockData>('/timeseries/fetch-data', { ticker, period: '6mo' });
      setStockData(data);
      addLog(`✓ ${data.dates.length} dias carregados de ${ticker}`, 'ok');
      show(`${ticker}: ${data.dates.length} dias`);
    } catch (e) {
      addLog('// erro: ' + (e instanceof Error ? e.message : String(e)), 'err');
    }
    setFetching(false);
  }, [ticker, addLog, show]);

  // Run comparison
  const handleCompare = useCallback(async () => {
    if (running || !stockData) return;
    setRunning(true);
    setResults(new Map());
    setLogLines([]);
    addLog(`// comparando ${selectedModels.size} modelos em ${ticker}...`, 'dim');

    // Send config first
    await apiPost('/timeseries/config', {
      ticker, windowSize, hiddenSize, alfa, maxCiclo,
      ativacao: 'tanh', validDays, forecastDays, validPct: 0,
    }).catch(() => {});

    const cleanup = apiSSE('/timeseries/compare', {
      body: { modelos: Array.from(selectedModels) },
      onMessage(data, event) {
        if (event === 'model-start') {
          const d = data as { modelo: string };
          setActiveModel(d.modelo);
          addLog(`▶ ${d.modelo} iniciando...`, 'dim');
        } else if (event === 'model-done') {
          const mr = data as TsCompareModelResult;
          if (mr.erro) {
            addLog(`✗ ${mr.modelo}: ${mr.erro}`, 'err');
          } else {
            addLog(`✓ ${mr.modelo}: RMSE R$${mr.result.rmseFinal.toFixed(4)} · ${mr.result.tempoMs}ms`, 'ok');
          }
          setResults(prev => new Map([...prev, [mr.modelo, mr]]));
          setActiveModel('');
        } else if (event === 'progress') {
          const p = data as { modelo: string; ciclo: number; mseTreino: number };
          if (p.ciclo % 500 === 0) {
            addLog(`  ${p.modelo} ciclo ${p.ciclo} · MSE ${p.mseTreino.toFixed(6)}`, '');
          }
        }
      },
      onDone() {
        setRunning(false);
        setActiveModel('');
        addLog('// comparação concluída!', 'ok');
        show('Comparação concluída');
      },
      onError() {
        setRunning(false);
        addLog('// erro de conexão', 'err');
      },
    });
    sseCleanup.current = cleanup;
  }, [running, stockData, selectedModels, ticker, windowSize, hiddenSize, alfa, maxCiclo, forecastDays, validDays, addLog, show]);

  // Sorted results by RMSE
  const sortedResults = Array.from(results.values())
    .filter(r => !r.erro && r.result.rmseFinal > 0)
    .sort((a, b) => a.result.rmseFinal - b.result.rmseFinal);

  // Build overlay chart data
  const overlayData = (() => {
    if (sortedResults.length === 0) return [];
    const base = sortedResults[0].result.pontos;
    if (!base) return [];
    return base.map((pt, i) => {
      const entry: Record<string, unknown> = { data: pt.data?.slice(5) || `${i}`, real: pt.preco };
      for (const mr of sortedResults) {
        if (mr.result.pontosValid && i >= base.length - (mr.result.pontosValid?.length || 0)) {
          const validIdx = i - (base.length - (mr.result.pontosValid?.length || 0));
          if (validIdx >= 0 && validIdx < mr.result.pontosValid.length) {
            entry[mr.modelo] = mr.result.pontosValid[validIdx].predito;
          }
        }
      }
      return entry;
    });
  })();

  // Group models by category
  const categories = new Map<string, TsAvailableModel[]>();
  for (const m of availableModels) {
    const list = categories.get(m.categoria) || [];
    list.push(m);
    categories.set(m.categoria, list);
  }

  const interval = Math.max(0, Math.floor(overlayData.length / 8) - 1);

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Comparar <span>Modelos</span></div>
          <div className="page-sub">Séries temporais — SMA, EMA, ARIMA, MLP, LSTM, GRU, BiLSTM, Seq2Seq, Prophet</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={handleCompare} disabled={running || !stockData || selectedModels.size === 0}>
            {running && <span className="spin" />}
            COMPARAR ({selectedModels.size})
          </button>
        </div>
      </div>

      {/* Config */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <Card title="Ação">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input type="text" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
              style={{ background: 'var(--surface-low)', border: '1px solid var(--border)', color: 'var(--on-surface)',
                fontFamily: 'var(--font-mono)', fontSize: 12, padding: '6px 10px', flex: 1, fontWeight: 700 }} />
            <button className="btn btn-primary" style={{ fontSize: 10, padding: '6px 12px' }}
              onClick={handleFetch} disabled={fetching || running}>
              {fetching ? '...' : 'BUSCAR'}
            </button>
          </div>
          {stockData && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--primary-glow)', marginTop: 6 }}>
              {stockData.dates.length} dias · R${Math.min(...stockData.close).toFixed(2)}–R${Math.max(...stockData.close).toFixed(2)}
            </div>
          )}
        </Card>
        <Card title="Janela · Ocultos (DL)">
          <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
            {[3, 5, 10, 20].map(w => (
              <button key={w} className={`porta-chip${windowSize === w ? ' selected' : ''}`}
                onClick={() => setWindowSize(w)} disabled={running}>{w}d</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[8, 16, 32].map(h => (
              <button key={h} className={`porta-chip${hiddenSize === h ? ' selected' : ''}`}
                onClick={() => setHiddenSize(h)} disabled={running}>{h}n</button>
            ))}
          </div>
        </Card>
        <Card title="Ciclos (DL) · Forecast">
          <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
            {[1000, 2000, 5000].map(c => (
              <button key={c} className={`porta-chip${maxCiclo === c ? ' selected' : ''}`}
                onClick={() => setMaxCiclo(c)} disabled={running}>{c >= 1000 ? `${c / 1000}k` : c}</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[3, 5, 7, 14].map(d => (
              <button key={d} className={`porta-chip${forecastDays === d ? ' selected' : ''}`}
                onClick={() => setForecastDays(d)} disabled={running}>{d}d</button>
            ))}
          </div>
        </Card>
      </div>

      {/* Model selector */}
      <Card title="Selecionar Modelos" style={{ marginBottom: 24 }}>
        {Array.from(categories.entries()).map(([cat, models]) => (
          <div key={cat} style={{ marginBottom: 12 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginBottom: 4 }}>{cat}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {models.map(m => (
                <button key={m.nome}
                  className={`porta-chip${selectedModels.has(m.nome) ? ' selected' : ''}`}
                  style={{
                    borderLeft: `3px solid ${m.cor}`,
                    opacity: m.needsPython ? 0.8 : 1,
                  }}
                  onClick={() => toggleModel(m.nome)} disabled={running}>
                  {m.nome} {m.needsPython ? '(Py)' : ''}
                </button>
              ))}
            </div>
          </div>
        ))}
      </Card>

      {/* Progress + Log */}
      {(running || results.size > 0) && (
        <div className="grid-2" style={{ marginBottom: 24 }}>
          <Card title="Log de Comparação">
            {activeModel && (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)', marginBottom: 8 }}>
                <span className="spin" style={{ display: 'inline-block', marginRight: 8 }} />
                Treinando: {activeModel}
              </div>
            )}
            <div className="log-panel">
              {logLines.map((line, i) => (
                <div key={i} className={`log-line ${line.cls}`}>{line.msg}</div>
              ))}
            </div>
          </Card>
          <Card title="Ranking por RMSE">
            {sortedResults.length > 0 ? (
              <table className="data-table" style={{ fontSize: 11 }}>
                <thead>
                  <tr><th>#</th><th>Modelo</th><th>RMSE</th><th>MAE</th><th>Pred. Amanhã</th><th>Tempo</th></tr>
                </thead>
                <tbody>
                  {sortedResults.map((mr, i) => (
                    <tr key={mr.modelo}>
                      <td style={{ color: i === 0 ? 'var(--primary-glow)' : 'var(--on-surface)', fontWeight: i === 0 ? 700 : 400 }}>
                        {i === 0 ? '🏆' : i + 1}
                      </td>
                      <td style={{ color: mr.cor, fontWeight: 700 }}>{mr.modelo}</td>
                      <td>R${mr.result.rmseFinal.toFixed(4)}</td>
                      <td>R${mr.result.maeFinal.toFixed(4)}</td>
                      <td className="td-green">R${mr.result.predicaoAmanha.toFixed(2)}</td>
                      <td>{mr.result.tempoMs < 1000 ? `${mr.result.tempoMs}ms` : `${(mr.result.tempoMs / 1000).toFixed(1)}s`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555' }}>aguardando resultados...</div>
            )}
          </Card>
        </div>
      )}

      {/* Overlay chart — all predictions on same chart */}
      {overlayData.length > 0 && (
        <Card title="Predições — Todos os Modelos" style={{ marginBottom: 24 }}>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
              <XAxis dataKey="data" stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} interval={interval} />
              <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }} />
              <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
              <Line type="monotone" dataKey="real" stroke="#00fbfb" strokeWidth={2} dot={false}
                name="preço real" isAnimationActive={false} />
              {sortedResults.map(mr => (
                <Line key={mr.modelo} type="monotone" dataKey={mr.modelo} stroke={mr.cor}
                  strokeWidth={1.5} dot={false} name={mr.modelo} isAnimationActive={false}
                  strokeDasharray="4 2" connectNulls={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Forecast comparison */}
      {sortedResults.length > 0 && sortedResults[0].result.forecast?.length > 0 && (
        <Card title="Previsão Futura — Comparação" style={{ marginBottom: 24 }}>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={Array.from({ length: sortedResults[0].result.forecast.length }, (_, d) => {
              const entry: Record<string, unknown> = { dia: `D+${d + 1}` };
              for (const mr of sortedResults) {
                if (mr.result.forecast?.[d]) {
                  entry[mr.modelo] = parseFloat(mr.result.forecast[d].predito.toFixed(2));
                }
              }
              return entry;
            })}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
              <XAxis dataKey="dia" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }} />
              <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
              {sortedResults.map(mr => (
                <Line key={mr.modelo} type="monotone" dataKey={mr.modelo} stroke={mr.cor}
                  strokeWidth={2} dot={{ r: 2, fill: mr.cor }} name={mr.modelo} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Details */}
      <Card title="Detalhes dos Modelos">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div><b style={{ color: '#00ff00' }}>SMA:</b> Média Móvel Simples — baseline, média dos últimos N dias</div>
          <div><b style={{ color: '#77ff61' }}>EMA:</b> Média Exponencial — pesa mais os dados recentes (α=2/(N+1))</div>
          <div><b style={{ color: '#ffaa00' }}>ARIMA:</b> AutoRegressive — coeficientes AR estimados por Yule-Walker + diferenciação</div>
          <div><b style={{ color: '#00fbfb' }}>MLP:</b> Multi-Layer Perceptron — sliding window + backpropagation</div>
          <div><b style={{ color: '#ff6ec7' }}>LSTM:</b> Long Short-Term Memory — 4 gates (forget, input, cell, output) + BPTT</div>
          <div><b style={{ color: '#aa66ff' }}>GRU:</b> Gated Recurrent Unit — 2 gates (reset, update), menos parâmetros que LSTM</div>
          <div><b style={{ color: '#ff44aa' }}>BiLSTM:</b> Bidirectional LSTM — processa sequência em ambas direções</div>
          <div><b style={{ color: '#ff3333' }}>Seq2Seq:</b> Encoder-Decoder LSTM — encoder comprime, decoder gera multi-step</div>
          <div><b style={{ color: '#66bbff' }}>ProphetLike:</b> Decomposição tendência + sazonalidade semanal (Go puro)</div>
          <div><b style={{ color: '#ff8800' }}>RandomForest:</b> Ensemble de árvores de decisão (sklearn, Python)</div>
          <div><b style={{ color: '#ff6600' }}>XGBoost:</b> Gradient Boosting otimizado (xgboost, Python)</div>
          <div><b style={{ color: '#4488ff' }}>Prophet:</b> Decomposição bayesiana da Meta (prophet, Python)</div>
        </div>
      </Card>
    </div>
  );
}
