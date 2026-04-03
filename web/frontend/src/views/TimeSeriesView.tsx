import { useState, useRef, useCallback, useEffect } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import LogChart from '../components/viz/LogChart';
import TimeSeriesChart from '../components/viz/TimeSeriesChart';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type { TsStockData, TsResult, TsStep, TsModelMeta } from '../api/types';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip } from 'recharts';

export default function TimeSeriesView() {
  const { show } = useToast();

  // Config
  const [ticker, setTicker] = useState('COGN3.SA');
  const [windowSize, setWindowSize] = useState(5);
  const [hiddenSize, setHiddenSize] = useState(32);
  const [alfa, setAlfa] = useState(0.01);
  const [maxCiclo, setMaxCiclo] = useState(5000);
  const [ativacao, setAtivacao] = useState('tanh');
  const [validPct, setValidPct] = useState(0);   // 0 = usar validDays fixo
  const [validDays, setValidDays] = useState(7);
  const [forecastDays, setForecastDays] = useState(7);

  // Data
  const [stockData, setStockData] = useState<TsStockData | null>(null);
  const [fetching, setFetching] = useState(false);

  // Training
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<TsResult | null>(null);
  const [mseHist, setMseHist] = useState<number[]>([]);
  const sseCleanup = useRef<(() => void) | null>(null);

  // Log
  const [logLines, setLogLines] = useState<{ msg: string; cls: string }[]>([
    { msg: '// busque dados de uma ação para começar...', cls: 'dim' },
  ]);
  const [progress, setProgress] = useState(0);

  // Models
  const [models, setModels] = useState<TsModelMeta[]>([]);
  const [saveName, setSaveName] = useState('');

  const trained = result !== null;

  useEffect(() => {
    apiGet<TsModelMeta[]>('/timeseries/models').then(setModels).catch(() => {});
  }, []);

  const refreshModels = useCallback(() => {
    apiGet<TsModelMeta[]>('/timeseries/models').then(setModels).catch(() => {});
  }, []);

  const addLog = useCallback((msg: string, cls: string = '') => {
    setLogLines(prev => {
      const next = [{ msg, cls }, ...prev];
      return next.length > 80 ? next.slice(0, 80) : next;
    });
  }, []);

  // ---------- Fetch Data ----------

  const handleFetch = useCallback(async () => {
    setFetching(true);
    addLog(`// buscando dados de ${ticker}...`, 'dim');
    try {
      const data = await apiPost<TsStockData>('/timeseries/fetch-data', { ticker, period: '6mo' });
      setStockData(data);
      addLog(`✓ ${data.dates.length} dias carregados (${data.dates[0]} a ${data.dates[data.dates.length - 1]})`, 'ok');
      show(`${ticker}: ${data.dates.length} dias carregados`);
    } catch (e) {
      addLog('// erro ao buscar dados: ' + (e instanceof Error ? e.message : String(e)), 'err');
      show('Erro ao buscar dados');
    }
    setFetching(false);
  }, [ticker, addLog, show]);

  // ---------- Train ----------

  const handleTrain = useCallback(async () => {
    if (training || !stockData) return;
    setTraining(true);
    setResult(null);
    setMseHist([]);
    setProgress(0);
    addLog(`// treinando MLP: window=${windowSize} hidden=${hiddenSize} lr=${alfa} ciclos=${maxCiclo}`, 'dim');

    try {
      await apiPost('/timeseries/config', { ticker, windowSize, hiddenSize, alfa, maxCiclo, ativacao, validDays, forecastDays, validPct });
    } catch {
      addLog('// erro ao configurar', 'err');
      setTraining(false);
      return;
    }

    const cleanup = apiSSE('/timeseries/train', {
      onMessage(data) {
        const step = data as TsStep;
        setProgress(Math.min((step.ciclo / maxCiclo) * 100, 99));
        if (step.ciclo % 500 === 0) {
          addLog(`ciclo ${step.ciclo} · MSE treino ${step.mseTreino.toFixed(6)} · MSE valid ${step.mseValid.toFixed(6)}`, 'ok');
        }
      },
      onDone(data) {
        const res = data as TsResult;
        setResult(res);
        setMseHist(res.mseHistorico || []);
        setProgress(100);
        setTraining(false);
        sseCleanup.current = null;
        addLog(`✓ RMSE: R$${res.rmseFinal.toFixed(4)} · MAE: R$${res.maeFinal.toFixed(4)} · Predição amanhã: R$${res.predicaoAmanha.toFixed(2)}`, 'ok');
        show(`Treinado — predição amanhã: R$${res.predicaoAmanha.toFixed(2)}`);
        refreshModels();
      },
      onError() {
        addLog('// erro de conexão', 'err');
        setTraining(false);
        sseCleanup.current = null;
      },
    });
    sseCleanup.current = cleanup;
  }, [training, stockData, ticker, windowSize, hiddenSize, alfa, maxCiclo, ativacao, validDays, addLog, show, refreshModels]);

  // ---------- Reset ----------

  const handleReset = useCallback(async () => {
    sseCleanup.current?.();
    try {
      await apiPost('/timeseries/reset');
      setResult(null); setMseHist([]); setStockData(null); setTraining(false);
      setLogLines([{ msg: '// busque dados de uma ação para começar...', cls: 'dim' }]);
      setProgress(0);
      show('Resetado');
    } catch { show('Erro ao resetar'); }
  }, [show]);

  // ---------- Save/Load ----------

  const handleSave = useCallback(async () => {
    try {
      await apiPost('/timeseries/save', { nome: saveName || ticker });
      setSaveName('');
      refreshModels();
      show('Modelo salvo');
    } catch { show('Erro ao salvar'); }
  }, [saveName, ticker, refreshModels, show]);

  const handleLoad = useCallback(async (id: string) => {
    try {
      const res = await apiPost<TsResult>('/timeseries/load', { id });
      setResult(res);
      setMseHist(res.mseHistorico || []);
      addLog(`✓ modelo carregado: ${res.ticker} · RMSE R$${res.rmseFinal.toFixed(4)}`, 'ok');
      show('Modelo carregado');
    } catch { show('Erro ao carregar'); }
  }, [addLog, show]);

  const handleDeleteModel = useCallback(async (id: string) => {
    try {
      await apiPost('/timeseries/delete-model', { id });
      refreshModels();
      show('Modelo deletado');
    } catch { show('Erro ao deletar'); }
  }, [refreshModels, show]);

  // Price chart data from stock data
  const priceChartData = stockData ? stockData.dates.map((d, i) => ({
    data: d.slice(5),
    preco: stockData.close[i],
  })) : [];

  const validStart = result ? result.pontos.length - (result.pontosValid?.length || 0) : undefined;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">MLP <span>Série Temporal</span></div>
          <div className="page-sub">Previsão de preço de ações com backpropagation &mdash; Aula 08</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={training}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training || !stockData}>
            {training && <span className="spin" />}
            TREINAR
          </button>
        </div>
      </div>

      {/* Config */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <Card title="Ação">
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input type="text" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
              style={{ background: 'var(--surface-low)', border: '1px solid var(--border)', color: 'var(--on-surface)',
                fontFamily: 'var(--font-mono)', fontSize: 12, padding: '6px 10px', flex: 1, fontWeight: 700 }}
              placeholder="COGN3.SA" />
            <button className="btn btn-primary" style={{ fontSize: 10, padding: '6px 12px', whiteSpace: 'nowrap' }}
              onClick={handleFetch} disabled={fetching || training}>
              {fetching ? '...' : 'BUSCAR'}
            </button>
          </div>
          {stockData && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--primary-glow)', marginTop: 8 }}>
              {stockData.dates.length} dias · R${Math.min(...stockData.close).toFixed(2)} — R${Math.max(...stockData.close).toFixed(2)}
            </div>
          )}
        </Card>
        <Card title="Janela · Ocultos">
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            {[1, 3, 5, 10, 20].map(w => (
              <button key={w} className={`porta-chip${windowSize === w ? ' selected' : ''}`}
                onClick={() => setWindowSize(w)} disabled={training}>{w}d</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[16, 32, 64, 128].map(h => (
              <button key={h} className={`porta-chip${hiddenSize === h ? ' selected' : ''}`}
                onClick={() => setHiddenSize(h)} disabled={training}>{h}</button>
            ))}
          </div>
        </Card>
        <Card title="LR · Ciclos · Ativação">
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            {[0.001, 0.005, 0.01, 0.02].map(a => (
              <button key={a} className={`porta-chip${alfa === a ? ' selected' : ''}`}
                onClick={() => setAlfa(a)} disabled={training}>{a}</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[2000, 5000, 10000].map(c => (
              <button key={c} className={`porta-chip${maxCiclo === c ? ' selected' : ''}`}
                onClick={() => setMaxCiclo(c)} disabled={training}>{c >= 1000 ? `${c/1000}k` : c}</button>
            ))}
            <button className={`porta-chip${ativacao === 'tanh' ? ' selected' : ''}`}
              onClick={() => setAtivacao('tanh')} disabled={training}>tanh</button>
            <button className={`porta-chip${ativacao === 'sigmoid' ? ' selected' : ''}`}
              onClick={() => setAtivacao('sigmoid')} disabled={training}>sigm</button>
          </div>
        </Card>
      </div>

      {/* Validation split + Forecast config */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <Card title="Split Treino/Validação">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginBottom: 8 }}>
            {validPct > 0 ? `${((1 - validPct) * 100).toFixed(0)}% treino / ${(validPct * 100).toFixed(0)}% validação`
              : `Treino: tudo exceto últimos ${validDays} dias`}
          </div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
            <button className={`porta-chip${validPct === 0 ? ' selected' : ''}`}
              onClick={() => setValidPct(0)} disabled={training}>dias fixos</button>
            <button className={`porta-chip${validPct === 0.1 ? ' selected' : ''}`}
              onClick={() => setValidPct(0.1)} disabled={training}>90/10%</button>
            <button className={`porta-chip${validPct === 0.2 ? ' selected' : ''}`}
              onClick={() => setValidPct(0.2)} disabled={training}>80/20%</button>
            <button className={`porta-chip${validPct === 0.3 ? ' selected' : ''}`}
              onClick={() => setValidPct(0.3)} disabled={training}>70/30%</button>
          </div>
          {validPct === 0 && (
            <div style={{ display: 'flex', gap: 6 }}>
              {[3, 5, 7, 10, 14].map(d => (
                <button key={d} className={`porta-chip${validDays === d ? ' selected' : ''}`}
                  onClick={() => setValidDays(d)} disabled={training}>{d}d</button>
              ))}
            </div>
          )}
        </Card>
        <Card title="Previsão Futura">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginBottom: 8 }}>
            Prever {forecastDays} dias à frente com intervalo de confiança
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {[3, 5, 7, 10, 14].map(d => (
              <button key={d} className={`porta-chip${forecastDays === d ? ' selected' : ''}`}
                onClick={() => setForecastDays(d)} disabled={training}>{d} dias</button>
            ))}
          </div>
        </Card>
      </div>

      {/* Price chart (raw data) */}
      {stockData && !result && (
        <Card title={`Histórico de Preços — ${stockData.ticker}`} style={{ marginBottom: 24 }}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={priceChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
              <XAxis dataKey="data" stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} interval={Math.max(0, Math.floor(priceChartData.length / 8) - 1)} />
              <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                tickLine={false} domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }} />
              <Line type="monotone" dataKey="preco" stroke="#00fbfb" strokeWidth={2} dot={false} isAnimationActive={false} />
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
          label={result ? `${result.ticker} · próximo dia útil` : 'aguardando'}
          color={result ? 'green' : undefined}
          valueStyle={result ? { fontSize: 28, color: 'var(--primary-glow)' } : undefined} />
      </div>

      {/* Log + MSE Chart */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <Card title="Log de Treinamento">
          <div className="progress-wrap">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="log-panel">
            {logLines.map((line, i) => (
              <div key={i} className={`log-line ${line.cls}`}>{line.msg}</div>
            ))}
          </div>
        </Card>
        <Card title="Curva de MSE — escala log">
          <LogChart data={mseHist} color="#00fbfb" />
        </Card>
      </div>

      {/* Prediction vs Real chart + Forecast */}
      {result && (
        <Card title="Preço Real vs Predição da Rede + Previsão Futura" style={{ marginBottom: 24 }}>
          <TimeSeriesChart pontos={result.pontos} forecast={result.forecast} validStart={validStart} height={320} />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginTop: 8 }}>
            <span style={{ color: '#00fbfb' }}>cyan</span> = preço real · <span style={{ color: '#00ff00' }}>verde tracejado</span> = predição treino · <span style={{ color: '#ff6ec7' }}>rosa</span> = validação · <span style={{ color: '#ffaa00' }}>laranja</span> = previsão futura (±intervalo de confiança)
          </div>
        </Card>
      )}

      {/* Validation detail */}
      {result && result.pontosValid && result.pontosValid.length > 0 && (
        <Card title="Detalhe da Validação — Últimos 7 dias" style={{ marginBottom: 24 }}>
          <table className="data-table" style={{ fontSize: 11 }}>
            <thead>
              <tr>
                <th>Data</th>
                <th>Preço Real</th>
                <th>Predição MLP</th>
                <th>Erro</th>
                <th>Erro %</th>
              </tr>
            </thead>
            <tbody>
              {result.pontosValid.map((p, i) => {
                const erro = Math.abs(p.preco - p.predito);
                const erroPct = p.preco > 0 ? (erro / p.preco) * 100 : 0;
                return (
                  <tr key={i}>
                    <td className="td-cyan">{p.data}</td>
                    <td>R${p.preco.toFixed(2)}</td>
                    <td className="td-green">R${p.predito.toFixed(2)}</td>
                    <td style={{ color: erro > 0.1 ? 'var(--pink)' : 'var(--on-surface)' }}>R${erro.toFixed(4)}</td>
                    <td style={{ color: erroPct > 3 ? 'var(--pink)' : 'var(--primary-glow)' }}>{erroPct.toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      {/* Forecast detail table */}
      {result && result.forecast && result.forecast.length > 0 && (
        <Card title={`Previsão Futura — Próximos ${result.forecast.length} dias`} style={{ marginBottom: 24 }}>
          <table className="data-table" style={{ fontSize: 11 }}>
            <thead>
              <tr><th>Dia</th><th>Predição</th><th>Intervalo de Confiança</th></tr>
            </thead>
            <tbody>
              {result.forecast.map(f => (
                <tr key={f.dia}>
                  <td className="td-cyan">D+{f.dia}</td>
                  <td className="td-green" style={{ fontWeight: 700 }}>R${f.predito.toFixed(2)}</td>
                  <td>R${f.lower.toFixed(2)} — R${f.upper.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* Models */}
      <Card title="Modelos Salvos" style={{ marginBottom: 24 }}>
        {trained && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center' }}>
            <input type="text" placeholder="Nome do modelo" value={saveName}
              onChange={e => setSaveName(e.target.value)}
              style={{ background: 'var(--surface-low)', border: '1px solid var(--border)', color: 'var(--on-surface)',
                fontFamily: 'var(--font-mono)', fontSize: 11, padding: '6px 10px', flex: 1 }} />
            <button className="btn btn-primary" style={{ fontSize: 11, padding: '6px 14px' }} onClick={handleSave}>SALVAR</button>
          </div>
        )}
        {models.length === 0 ? (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)' }}>Nenhum modelo salvo</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {models.map(m => (
              <div key={m.id} style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px',
                background: 'var(--surface-low)', fontFamily: 'var(--font-mono)', fontSize: 11,
              }}>
                <span style={{ color: 'var(--cyan)', fontWeight: 700, minWidth: 120 }}>{m.id}</span>
                <span style={{ color: 'var(--on-surface)', flex: 1 }}>{m.nome || '—'}</span>
                <span style={{ color: 'var(--pink)' }}>{m.ticker}</span>
                <span style={{ color: 'var(--primary-glow)' }}>RMSE R${m.rmseFinal.toFixed(4)}</span>
                <span style={{ color: 'var(--cyan)' }}>→ R${m.predicaoAmanha.toFixed(2)}</span>
                <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }}
                  onClick={() => handleLoad(m.id)}>CARREGAR</button>
                <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px', color: 'var(--pink)' }}
                  onClick={() => handleDeleteModel(m.id)}>×</button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Details */}
      <Card title="Detalhes">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div><b style={{ color: 'var(--cyan)' }}>Arquitetura:</b> {windowSize} entradas (janela) → {hiddenSize} ocultos ({ativacao}) → 1 saída (preço)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Dados:</b> Yahoo Finance — preço de fechamento diário, normalizado Min-Max [0,1]</div>
          <div><b style={{ color: 'var(--cyan)' }}>Janela deslizante:</b> {windowSize} dias anteriores → prever dia seguinte</div>
          <div><b style={{ color: 'var(--cyan)' }}>Validação:</b> {validPct > 0 ? `${(validPct * 100).toFixed(0)}% dos dados` : `últimos ${validDays} dias`} separados do treino</div>
          <div><b style={{ color: 'var(--cyan)' }}>Previsão futura:</b> {forecastDays} dias com intervalo de confiança (±RMSE × √dia)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Métricas:</b> MSE, RMSE (raiz), MAE (absoluto) — em R$</div>
          <div><b style={{ color: 'var(--cyan)' }}>Referência:</b> Aula 08 — Investimentos e MLP (ver slide para referências)</div>
        </div>
      </Card>
    </div>
  );
}
