import { useState, useEffect, useRef, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import PixelGrid from '../components/shared/PixelGrid';
import LogChart from '../components/viz/LogChart';
import NetworkViz from '../components/viz/NetworkViz';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';

// ---------------------------------------------------------------------------
// Types matching the Go structs (letras package)
// ---------------------------------------------------------------------------

interface LtrStep {
  ciclo: number;
  letraIdx: number;
  letra: string;
  erroTotal: number;
}

interface LtrResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  acertos: number;
  total: number;
  acuracia: number;
}

interface LtrCandidate {
  letra: string;
  score: number;
  idx: number;
}

interface LtrClassifyResp {
  letraIdx: number;
  letra: string;
  scores: number[];
  top5: LtrCandidate[];
}

interface LtrDatasetEntry {
  letra: string;
  idx: number;
  grade: number[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MlpLetrasView() {
  const { show } = useToast();

  // Training state
  const [training, setTraining] = useState(false);
  const [ciclos, setCiclos] = useState('\u2014');
  const [acuracia, setAcuracia] = useState('\u2014');
  const [erro, setErro] = useState('\u2014');
  const sseCleanup = useRef<(() => void) | null>(null);

  // Error chart
  const [erroHistorico, setErroHistorico] = useState<number[]>([]);

  // Log panel
  const [logLines, setLogLines] = useState<{ msg: string; cls: string }[]>([
    { msg: '// aguardando inicio do treinamento...', cls: 'dim' },
  ]);
  const [progress, setProgress] = useState(0);

  // Training result (needed to know if trained)
  const [trained, setTrained] = useState(false);

  // Dataset + letter browser predictions
  const [dataset, setDataset] = useState<LtrDatasetEntry[] | null>(null);
  const [browserPreds, setBrowserPreds] = useState<Map<number, LtrClassifyResp>>(new Map());
  const [activeLetter, setActiveLetter] = useState<number | null>(null);

  // Interactive test grid
  const [testPixels, setTestPixels] = useState<number[]>(new Array(35).fill(-1));
  const [classifyResult, setClassifyResult] = useState<LtrClassifyResp | null>(null);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => { sseCleanup.current?.(); };
  }, []);

  // ---------- Log helper ----------

  const addLog = useCallback((msg: string, cls: string = '') => {
    setLogLines(prev => {
      const next = [{ msg, cls }, ...prev];
      return next.length > 60 ? next.slice(0, 60) : next;
    });
  }, []);

  // ---------- Train ----------

  const handleTrain = useCallback(() => {
    if (training) return;
    setTraining(true);
    setLogLines([]);
    addLog('// iniciando treinamento MLP A-Z...', 'dim');
    setProgress(0);

    const cleanup = apiSSE('/letras/train', {
      onMessage(data) {
        const step = data as LtrStep;
        setProgress(Math.min((step.ciclo / 50000) * 100, 99));
        if (step.ciclo % 100 === 1 || step.letraIdx === 0) {
          addLog(`ciclo ${String(step.ciclo).padStart(5)} \u00B7 erro ${step.erroTotal.toFixed(4)}`, 'ok');
        }
      },
      onDone(data) {
        const result = data as LtrResult;
        setProgress(100);
        setAcuracia(result.acuracia.toFixed(0) + '%');
        setCiclos(result.ciclos.toLocaleString());
        setErro(result.erroFinal.toFixed(4));
        setErroHistorico(result.erroHistorico);
        addLog(`\u2713 concluido: ${result.acertos}/${result.total} (${result.acuracia.toFixed(0)}%)`, 'ok');
        setTraining(false);
        setTrained(true);
        sseCleanup.current = null;
        show('MLP Letras treinado \u2014 ' + result.acuracia.toFixed(0) + '%');
        loadDatasetAndBrowser();
      },
      onError() {
        addLog('// erro de conexao', 'err');
        setTraining(false);
        sseCleanup.current = null;
      },
    });
    sseCleanup.current = cleanup;
  }, [training, addLog, show]);

  // ---------- Load dataset + classify all letters for the browser ----------

  const loadDatasetAndBrowser = useCallback(async () => {
    try {
      const entries = await apiGet<LtrDatasetEntry[]>('/letras/dataset');
      setDataset(entries);

      const predMap = new Map<number, LtrClassifyResp>();
      const results = await Promise.all(
        entries.map(entry =>
          apiPost<LtrClassifyResp>('/letras/classify', { grade: entry.grade })
        )
      );
      results.forEach((resp, i) => {
        predMap.set(i, resp);
      });
      setBrowserPreds(predMap);
    } catch {
      // ignore
    }
  }, []);

  // ---------- Classify interactive grid ----------

  const handleClassify = useCallback(async () => {
    try {
      const resp = await apiPost<LtrClassifyResp>('/letras/classify', { grade: testPixels });
      setClassifyResult(resp);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }, [testPixels, show]);

  const handleClear = useCallback(() => {
    setTestPixels(new Array(35).fill(-1));
    setClassifyResult(null);
  }, []);

  // ---------- Letter browser click ----------

  const handleLetterClick = useCallback((idx: number) => {
    setActiveLetter(prev => prev === idx ? null : idx);
  }, []);

  // Active letter preview data
  const activeEntry = activeLetter !== null && dataset ? dataset[activeLetter] : null;
  const activePred = activeLetter !== null ? browserPreds.get(activeLetter) : null;

  // ---------- Render ----------

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">MLP <span>Letras</span></div>
          <div className="page-sub">35 entradas &middot; 15 ocultos &middot; 26 saidas &middot; reconhecimento A&ndash;Z</div>
        </div>
        <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
          {training && <span className="spin" />}
          TREINAR REDE
        </button>
      </div>

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Acuracia"
          value={acuracia}
          label="nos dados de treino"
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Ciclos"
          value={ciclos}
          label="ate convergencia"
          color="cyan"
        />
        <MetricCard
          title="Erro Final"
          value={erro}
          label="quadratico total"
          valueStyle={{ fontSize: 24, color: 'var(--on-surface)' }}
        />
      </div>

      {/* Training Log */}
      <Card title="Log de Treinamento" style={{ marginBottom: 16 }}>
        <div className="progress-wrap">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <div className="log-panel">
          {logLines.map((line, i) => (
            <div key={i} className={`log-line ${line.cls}`}>{line.msg}</div>
          ))}
        </div>
      </Card>

      {/* Error Chart */}
      <Card title="Curva de Erro \u2014 escala log" style={{ marginBottom: 16 }}>
        <LogChart data={erroHistorico} color="#00fbfb" />
      </Card>

      {/* Letter browser + Interactive test */}
      <div className="grid-2">
        {/* Results by Letter */}
        <Card title="Resultados por Letra">
          {!trained || !dataset ? (
            <div className="letter-browser">
              <div className="empty" style={{ padding: '12px 0', fontSize: 10 }}>
                Treine a rede para ver resultados
              </div>
            </div>
          ) : (
            <>
              <div className="letter-browser">
                {dataset.map((entry, i) => {
                  const pred = browserPreds.get(i);
                  const correct = pred ? pred.letraIdx === entry.idx : false;
                  const isActive = activeLetter === i;
                  return (
                    <div
                      key={i}
                      className={`letter-chip${isActive ? ' active' : ''}${pred ? (correct ? ' correct' : ' wrong') : ''}`}
                      onClick={() => handleLetterClick(i)}
                    >
                      {entry.letra}
                    </div>
                  );
                })}
              </div>

              {activeEntry && activePred && (
                <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                  {/* Mini preview grid */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(5, 8px)',
                      gridTemplateRows: 'repeat(7, 8px)',
                      gap: 1,
                    }}
                  >
                    {activeEntry.grade.slice(0, 35).map((v, pi) => (
                      <div
                        key={pi}
                        className={`preview-pixel${v > 0 ? ' on' : ''}`}
                        style={{ width: 8, height: 8 }}
                      />
                    ))}
                  </div>
                  <div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', marginBottom: 6 }}>
                      Letra: <span style={{ color: 'var(--secondary)', fontWeight: 700 }}>{activeEntry.letra}</span>
                      &nbsp;&rarr;&nbsp;
                      Pred: <span style={{ color: activePred.letraIdx === activeEntry.idx ? 'var(--primary-glow)' : 'var(--pink)' }}>
                        {activePred.letra}
                      </span>
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)' }}>
                      Conf: <span style={{ color: 'var(--cyan)' }}>{activePred.top5[0].score.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        {/* Interactive Test */}
        <Card title="Teste Manual \u2014 Grade 5x7">
          <div className="letter-section">
            <div>
              <PixelGrid
                rows={7}
                cols={5}
                cellSize={28}
                values={testPixels}
                onChange={setTestPixels}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button
                  className="btn btn-ghost"
                  style={{ padding: '6px 12px', fontSize: 10 }}
                  onClick={handleClear}
                >LIMPAR</button>
                <button
                  className="btn btn-primary"
                  style={{ padding: '6px 12px', fontSize: 10 }}
                  onClick={handleClassify}
                >CLASSIFICAR</button>
              </div>
            </div>

            <div className="letter-controls">
              {classifyResult ? (
                <>
                  <div>
                    <div className="result-big">{classifyResult.letra}</div>
                    <div className="result-label">
                      confianca: {classifyResult.top5[0].score.toFixed(4)}
                    </div>
                  </div>
                  <div className="conf-list">
                    {classifyResult.top5.map((c, i) => {
                      const norm = Math.max(0, Math.min(1, (c.score + 1) / 2));
                      return (
                        <div className="conf-row" key={i}>
                          <span className="conf-letter">{c.letra}</span>
                          <div className="conf-bar-wrap">
                            <div className="conf-bar-fill" style={{ width: `${(norm * 100).toFixed(1)}%` }} />
                          </div>
                          <span className="conf-score">{c.score.toFixed(4)}</span>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div>
                  <div className="result-big">?</div>
                  <div className="result-label" />
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>

      {/* Network Visualization */}
      <Card title="Arquitetura da Rede" style={{ marginTop: 16 }}>
        <NetworkViz layerSizes={[35, 15, 26]} hudText="tanh" />
      </Card>
    </div>
  );
}
