import { useState, useRef, useEffect, useCallback } from 'react';
import Card from '../components/shared/Card';
import Select from '../components/shared/Select';
import BenchChart from '../components/viz/BenchChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';

const IMAGENS = [
  { value: 'coracao', label: 'Coracao' },
  { value: 'smiley', label: 'Smiley' },
  { value: 'radial', label: 'Radial' },
  { value: 'brasil', label: 'Brasil' },
];

const LAYERS = [
  { value: '2', label: '2 camadas' },
  { value: '3', label: '3 camadas' },
  { value: '4', label: '4 camadas' },
  { value: '5', label: '5 camadas' },
];

const NEURONS = [
  { value: '16', label: '16 neuronios' },
  { value: '32', label: '32 neuronios' },
  { value: '64', label: '64 neuronios' },
  { value: '128', label: '128 neuronios' },
];

const EPOCAS_OPTIONS = [
  { value: '500', label: '500' },
  { value: '1000', label: '1k' },
  { value: '2000', label: '2k' },
  { value: '5000', label: '5k' },
];

interface BackendResult {
  backend: string;
  elapsedMs: number;
  loss: number;
  epocas: number;
  convergiu: boolean;
}

const BACKENDS = ['standard', 'goroutines', 'matrix', 'minibatch'] as const;

const BACKEND_LABELS: Record<string, string> = {
  standard: 'Standard',
  goroutines: 'Goroutines',
  matrix: 'Matrix',
  minibatch: 'Minibatch',
};

const BACKEND_COLORS: Record<string, string> = {
  standard: '#00ff00',
  goroutines: '#00fbfb',
  matrix: '#ff6ec7',
  minibatch: '#77ff61',
};

const BACKEND_PREFIXES: Record<string, string> = {
  standard: '/imgreg',
  goroutines: '/imgreg-goroutines',
  matrix: '/imgreg-matrix',
  minibatch: '/imgreg-minibatch',
};

function rgbToCSS(rgb: [number, number, number]): string {
  return `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`;
}

function MiniGrid({ pixels }: { pixels: [number, number, number][] }) {
  if (!pixels || pixels.length !== 256) return <div className="bench-mini-grid" style={{ width: 112, height: 112, background: 'var(--surface-low)' }} />;
  return (
    <div className="bench-mini-grid" style={{ width: 112 }}>
      {pixels.map((rgb, i) => (
        <div key={i} className="bench-mini-pixel" style={{ background: rgbToCSS(rgb) }} />
      ))}
    </div>
  );
}

export default function ImgregBenchView() {
  const toast = useToast();

  const [imagem, setImagem] = useState('coracao');
  const [hiddenLayers, setHiddenLayers] = useState('3');
  const [neuronsPerLayer, setNeuronsPerLayer] = useState('32');
  const [maxEpocas, setMaxEpocas] = useState('1000');
  const [mode, setMode] = useState<'sequencial' | 'paralelo'>('sequencial');

  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<BackendResult[]>([]);
  const [grids, setGrids] = useState<Record<string, [number, number, number][]>>({});
  const [progressMap, setProgressMap] = useState<Record<string, { epoca: number; loss: number }>>({});
  const [logLines, setLogLines] = useState<string[]>([]);

  const cleanups = useRef<(() => void)[]>([]);

  useEffect(() => {
    return () => { cleanups.current.forEach(c => c()); };
  }, []);

  const addLog = useCallback((msg: string) => {
    setLogLines(prev => [...prev.slice(-100), msg]);
  }, []);

  const buildConfig = () => ({
    hiddenLayers: parseInt(hiddenLayers),
    neuronsPerLayer: parseInt(neuronsPerLayer),
    learningRate: 0.01,
    imagem,
    maxEpocas: parseInt(maxEpocas),
  });

  // --- Sequential mode: use bench endpoint ---
  const runSequential = useCallback(async () => {
    const cfg = buildConfig();
    await apiPost('/imgreg-bench/config', cfg);

    addLog(`[sequencial] iniciando benchmark...`);

    const cleanup = apiSSE('/imgreg-bench/train', {
      onMessage(data: unknown) {
        const msg = data as { backend: string; step: { epoca: number; loss: number; done?: boolean; convergiu?: boolean; elapsedMs?: number; outputPixels?: [number, number, number][] } };
        const { backend, step } = msg;

        if (step.outputPixels) {
          setGrids(prev => ({ ...prev, [backend]: step.outputPixels! }));
        }

        if (step.done) {
          const r: BackendResult = { backend, elapsedMs: step.elapsedMs ?? 0, loss: step.loss, epocas: step.epoca, convergiu: step.convergiu ?? false };
          setResults(prev => [...prev.filter(x => x.backend !== backend), r]);
          addLog(`[${backend}] DONE ${step.elapsedMs}ms loss=${step.loss.toFixed(6)}`);
        } else {
          setProgressMap(prev => ({ ...prev, [backend]: { epoca: step.epoca, loss: step.loss } }));
        }
      },
      onError() {
        // Stream ended — might be normal completion
      },
    });
    cleanups.current.push(cleanup);
  }, [hiddenLayers, neuronsPerLayer, maxEpocas, imagem, addLog]);

  // --- Parallel mode: start 4 SSE streams simultaneously ---
  const runParallel = useCallback(async () => {
    const cfg = buildConfig();
    addLog(`[paralelo] iniciando 4 streams simultaneos...`);

    for (const backend of BACKENDS) {
      const prefix = BACKEND_PREFIXES[backend];
      const cfgForBackend = backend === 'minibatch'
        ? { ...cfg, batchSize: 32, numWorkers: 4 }
        : cfg;

      await apiPost(`${prefix}/config`, cfgForBackend);

      const cleanup = apiSSE(`${prefix}/train`, {
        onMessage(data: unknown) {
          const step = data as { epoca: number; loss: number; done?: boolean; convergiu?: boolean; elapsedMs?: number; outputPixels?: [number, number, number][] };

          if (step.outputPixels) {
            setGrids(prev => ({ ...prev, [backend]: step.outputPixels! }));
          }

          if (step.done) {
            const r: BackendResult = { backend, elapsedMs: step.elapsedMs ?? 0, loss: step.loss, epocas: step.epoca, convergiu: step.convergiu ?? false };
            setResults(prev => [...prev.filter(x => x.backend !== backend), r]);
            addLog(`[${backend}] DONE ${step.elapsedMs}ms loss=${step.loss.toFixed(6)}`);
          } else {
            setProgressMap(prev => ({ ...prev, [backend]: { epoca: step.epoca, loss: step.loss } }));
          }
        },
        onError() {},
      });
      cleanups.current.push(cleanup);
    }
  }, [hiddenLayers, neuronsPerLayer, maxEpocas, imagem, addLog]);

  const handleRun = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setResults([]);
    setGrids({});
    setProgressMap({});
    setLogLines([]);

    try {
      if (mode === 'paralelo') {
        await runParallel();
      } else {
        await runSequential();
      }
    } catch (e) {
      toast.show('Erro: ' + (e instanceof Error ? e.message : String(e)));
      setRunning(false);
    }
  }, [running, mode, runSequential, runParallel, toast]);

  // Detect completion
  useEffect(() => {
    if (running && results.length === 4) {
      setRunning(false);
      cleanups.current.forEach(c => c());
      cleanups.current = [];
      toast.show('Benchmark concluido!');
    }
  }, [results, running, toast]);

  const handleReset = useCallback(async () => {
    cleanups.current.forEach(c => c());
    cleanups.current = [];
    try { await apiPost('/imgreg-bench/reset'); } catch { /* ignore */ }
    for (const prefix of Object.values(BACKEND_PREFIXES)) {
      try { await apiPost(`${prefix}/reset`); } catch { /* ignore */ }
    }
    setRunning(false);
    setResults([]);
    setGrids({});
    setProgressMap({});
    setLogLines([]);
    toast.show('Benchmark resetado');
  }, [toast]);

  const chartData = results.map(r => ({
    metodo: BACKEND_LABELS[r.backend] || r.backend,
    tempoMs: r.elapsedMs,
    loss: r.loss,
  }));

  const sortedResults = [...results].sort((a, b) => a.elapsedMs - b.elapsedMs);
  const fastest = sortedResults.length > 0 ? sortedResults[0].elapsedMs : 0;

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Benchmark <span>Comparativo</span></div>
          <div className="page-sub">standard vs goroutines vs matrix vs minibatch</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={running}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleRun} disabled={running}>
            {running && <span className="spin" />}
            RODAR BENCHMARK
          </button>
        </div>
      </div>

      {/* Mode toggle + Config */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Modo</div>
          <div className="bench-mode-toggle">
            <button className={`bench-mode-btn${mode === 'sequencial' ? ' active' : ''}`} onClick={() => setMode('sequencial')}>SEQUENCIAL</button>
            <button className={`bench-mode-btn${mode === 'paralelo' ? ' active' : ''}`} onClick={() => setMode('paralelo')}>PARALELO</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <Select label="Imagem" options={IMAGENS} value={imagem} onChange={setImagem} style={{ width: '100%' }} />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Camadas &middot; Neuronios</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={LAYERS} value={hiddenLayers} onChange={setHiddenLayers} style={{ flex: 1 }} />
            <Select options={NEURONS} value={neuronsPerLayer} onChange={setNeuronsPerLayer} style={{ flex: 1 }} />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select label="Epocas" options={EPOCAS_OPTIONS} value={maxEpocas} onChange={setMaxEpocas} style={{ width: '100%' }} />
        </Card>
      </div>

      {/* Live image grids */}
      {(running || Object.keys(grids).length > 0) && (
        <Card title="Imagens em Tempo Real" pulse={running} style={{ marginBottom: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            {BACKENDS.map(backend => {
              const done = results.find(r => r.backend === backend);
              const prog = progressMap[backend];
              return (
                <div key={backend} style={{ textAlign: 'center' }}>
                  <div style={{
                    fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
                    color: BACKEND_COLORS[backend], letterSpacing: '0.1em',
                    textTransform: 'uppercase', marginBottom: 6,
                  }}>
                    {BACKEND_LABELS[backend]}
                  </div>
                  <MiniGrid pixels={grids[backend]} />
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginTop: 4 }}>
                    {done
                      ? <span style={{ color: 'var(--primary-glow)' }}>{done.elapsedMs}ms</span>
                      : prog
                        ? `ep ${prog.epoca} · ${prog.loss.toFixed(4)}`
                        : 'aguardando...'}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Result Cards */}
      {sortedResults.length > 0 && (
        <div className="bench-result-cards" style={{ marginBottom: 16 }}>
          {sortedResults.map((r, i) => {
            const speedup = fastest > 0 && r.elapsedMs > 0 ? (r.elapsedMs / fastest).toFixed(2) + 'x' : '';
            return (
              <div key={r.backend} className={`bench-result-card${i === 0 ? ' fastest' : ''}`}>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
                  color: BACKEND_COLORS[r.backend], letterSpacing: '0.1em',
                  textTransform: 'uppercase', marginBottom: 12,
                }}>
                  {i === 0 ? '\u2605 ' : ''}{BACKEND_LABELS[r.backend]}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <div>
                    <div className="metric-label">Tempo</div>
                    <div className="metric-val" style={{
                      color: i === 0 ? 'var(--primary-glow)' : 'var(--secondary)',
                      textShadow: i === 0 ? '0 0 12px rgba(0,255,0,0.3)' : 'none',
                    }}>{r.elapsedMs}ms</div>
                  </div>
                  <div>
                    <div className="metric-label">Loss</div>
                    <div className="metric-val" style={{ color: 'var(--cyan)' }}>{r.loss.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="metric-label">Epocas</div>
                    <div className="metric-val">{r.epocas}</div>
                  </div>
                  <div>
                    <div className="metric-label">Relativo</div>
                    <div className="metric-val" style={{ color: i === 0 ? 'var(--primary-glow)' : 'var(--on-surface)' }}>{speedup}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="grid-2" style={{ marginBottom: 16 }}>
          <Card title="Tempo (ms)">
            <BenchChart data={chartData} dataKey="tempoMs" height={250} />
          </Card>
          <Card title="Loss Final">
            <BenchChart data={chartData} dataKey="loss" height={250} />
          </Card>
        </div>
      )}

      {/* Log */}
      <Card title="Log" style={{ marginBottom: 16 }}>
        <div className="log-panel">
          {logLines.map((line, i) => (
            <div key={i} className={`log-line${i < logLines.length - 3 ? ' dim' : ''}`}>{line}</div>
          ))}
          {logLines.length === 0 && <div className="log-line dim">aguardando inicio do benchmark...</div>}
        </div>
      </Card>

      <Card title="Detalhes">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          Compara 4 implementacoes da mesma rede MLP para regressao de imagem:
          <br /><br />
          <b style={{ color: '#00ff00' }}>Standard</b> — implementacao sequencial basica
          <br />
          <b style={{ color: '#00fbfb' }}>Goroutines</b> — paralelismo por camada
          <br />
          <b style={{ color: '#ff6ec7' }}>Matrix</b> — operacoes matriciais em batch
          <br />
          <b style={{ color: '#77ff61' }}>Minibatch</b> — mini-batch SGD com workers
          <br /><br />
          <b>Modo {mode}:</b> {mode === 'sequencial'
            ? 'executa cada metodo um apos o outro (benchmark justo)'
            : 'executa todos os 4 metodos simultaneamente (divertido mas nao justo)'}
        </div>
      </Card>
    </div>
  );
}
