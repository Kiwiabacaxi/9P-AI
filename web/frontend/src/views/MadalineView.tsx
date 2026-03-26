import { useState, useEffect, useRef, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import PixelGrid from '../components/shared/PixelGrid';
import LogChart from '../components/viz/LogChart';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const LETRAS = ['A','B','C','D','E','F','G','H','I','J','K','L','M'] as const;

interface MadStep {
  ciclo: number;
  letraIdx: number;
  letra: string;
  yIn: number[];
  y: number[];
  target: number[];
  erros: boolean[];
}

interface MadResult {
  convergiu: boolean;
  ciclos: number;
  steps: MadStep[];
  acertos: number;
  total: number;
  acuracia: number;
}

interface MadCandidate {
  letra: string;
  idx: number;
  score: number;
}

interface MadClassifyResp {
  letraIdx: number;
  letra: string;
  scores: number[];
  top5: MadCandidate[];
}

interface DatasetEntry {
  letra: string;
  idx: number;
  grade: number[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MadalineView() {
  const { show } = useToast();

  // Training state
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<MadResult | null>(null);
  const [ciclos, setCiclos] = useState<string>('\u2014');
  const [acuracia, setAcuracia] = useState<string>('\u2014');
  const [status, setStatus] = useState('aguardando');
  const [statusColor, setStatusColor] = useState('var(--on-surface)');

  // Error history for chart
  const [erroHist, setErroHist] = useState<number[]>([]);

  // Dataset
  const [dataset, setDataset] = useState<DatasetEntry[]>([]);

  // Letter browser
  const [activeLetter, setActiveLetter] = useState<number | null>(null);

  // Classification
  const [testPixels, setTestPixels] = useState<number[]>(new Array(35).fill(-1));
  const [classifyResp, setClassifyResp] = useState<MadClassifyResp | null>(null);

  // SSE ref
  const sseCleanup = useRef<(() => void) | null>(null);

  // Track which letters are correctly classified after training
  const [letterCorrect, setLetterCorrect] = useState<Record<number, boolean>>({});

  // Fetch dataset on mount
  useEffect(() => {
    apiGet<DatasetEntry[]>('/madaline/dataset').then(setDataset).catch(() => {});
  }, []);

  // Try to restore a previous result
  useEffect(() => {
    apiGet<MadResult>('/madaline/result').then(res => {
      setResult(res);
      setCiclos(res.ciclos.toLocaleString());
      setAcuracia(res.acuracia.toFixed(1) + '%');
      setStatus(res.convergiu ? 'convergiu' : 'limite atingido');
      setStatusColor(res.convergiu ? 'var(--primary-glow)' : 'var(--pink)');
      // Build letter correctness map
      checkLetterCorrectness();
    }).catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => { sseCleanup.current?.(); };
  }, []);

  // ---------- Check letter correctness ----------

  const checkLetterCorrectness = useCallback(async () => {
    if (dataset.length === 0) return;
    const map: Record<number, boolean> = {};
    for (let i = 0; i < dataset.length; i++) {
      try {
        const resp = await apiPost<MadClassifyResp>('/madaline/classify', { grade: dataset[i].grade });
        map[i] = resp.letraIdx === i;
      } catch {
        // ignore
      }
    }
    setLetterCorrect(map);
  }, [dataset]);

  // ---------- Train ----------

  const handleTrain = useCallback(async () => {
    if (training) return;
    setTraining(true);
    setStatus('treinando...');
    setStatusColor('var(--cyan)');
    setResult(null);
    setErroHist([]);
    setLetterCorrect({});
    setClassifyResp(null);

    // Track the last known ciclo to build error history
    let lastCiclo = 0;
    const errorsPerCiclo: number[] = [];

    const cleanup = apiSSE('/madaline/train', {
      onMessage(data) {
        const step = data as MadStep;
        setCiclos(step.ciclo.toLocaleString());

        // Count errors in this step for a rough per-cycle error metric
        const nErrors = step.erros.filter(Boolean).length;
        if (step.ciclo !== lastCiclo) {
          if (lastCiclo > 0 && errorsPerCiclo.length > 0) {
            setErroHist(prev => [...prev, errorsPerCiclo[errorsPerCiclo.length - 1]]);
          }
          errorsPerCiclo.length = 0;
          lastCiclo = step.ciclo;
        }
        errorsPerCiclo.push(nErrors);
      },
      onDone(data) {
        const res = data as MadResult;
        setResult(res);
        setCiclos(res.ciclos.toLocaleString());
        setAcuracia(res.acuracia.toFixed(1) + '%');
        setStatus(res.convergiu ? 'convergiu' : 'limite atingido');
        setStatusColor(res.convergiu ? 'var(--primary-glow)' : 'var(--pink)');
        setTraining(false);
        sseCleanup.current = null;
        show(res.convergiu ? 'MADALINE convergiu!' : 'Limite de ciclos atingido');
        // Check letter correctness after training
        checkLetterCorrectness();
      },
      onError() {
        setTraining(false);
        setStatus('erro de conexao');
        setStatusColor('var(--pink)');
        sseCleanup.current = null;
        show('Erro no treinamento');
      },
    });
    sseCleanup.current = cleanup;
  }, [training, show, checkLetterCorrectness]);

  // ---------- Classify ----------

  const handleClassify = useCallback(async () => {
    try {
      const resp = await apiPost<MadClassifyResp>('/madaline/classify', { grade: testPixels });
      setClassifyResp(resp);
    } catch {
      show('Erro ao classificar (rede nao treinada?)');
    }
  }, [testPixels, show]);

  // Load a letter from dataset into the test grid
  const handleLoadLetter = useCallback((entry: DatasetEntry) => {
    setActiveLetter(entry.idx);
    setTestPixels(entry.grade.map(v => (v >= 0 ? 1 : -1)));
    setClassifyResp(null);
  }, []);

  const trained = result !== null;

  return (
    <div>
      {/* ===== Page Header ===== */}
      <div className="page-header">
        <div>
          <div className="page-title">MADALINE <span>MRII &mdash; Classificacao A-M</span></div>
          <div className="page-sub">minimo impacto &middot; 13 ADALINE + OR &middot; Trab 04</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            TREINAR
          </button>
        </div>
      </div>

      {/* ===== Metrics ===== */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Ciclos"
          value={ciclos}
          label={result ? (result.convergiu ? 'convergiu' : 'limite atingido') : 'aguardando'}
          color="cyan"
          pulse={training}
        />
        <MetricCard
          title="Acuracia"
          value={acuracia}
          label={result ? `${result.acertos}/${result.total} letras corretas` : 'aguardando'}
          color="green"
        />
        <MetricCard
          title="Status"
          value={status}
          label="estado da rede"
          valueStyle={{ fontSize: 18, color: statusColor }}
        />
      </div>

      {/* ===== Error Chart ===== */}
      {erroHist.length > 0 && (
        <Card title="Erros por Ciclo" style={{ marginBottom: 16 }}>
          <LogChart data={erroHist} color="#ff007f" />
        </Card>
      )}

      {/* ===== Letter Browser ===== */}
      <Card title="Letras do Dataset" style={{ marginBottom: 16 }}>
        <div className="letter-browser">
          {LETRAS.map((nome, idx) => {
            const isActive = activeLetter === idx;
            const correctness = letterCorrect[idx];
            const cls = [
              'letter-chip',
              isActive ? 'active' : '',
              trained && correctness === true ? 'correct' : '',
              trained && correctness === false ? 'wrong' : '',
            ].filter(Boolean).join(' ');
            return (
              <div
                key={nome}
                className={cls}
                onClick={() => {
                  const entry = dataset.find(d => d.idx === idx);
                  if (entry) handleLoadLetter(entry);
                }}
              >
                {nome}
              </div>
            );
          })}
        </div>

        {/* Mini preview grids */}
        {dataset.length > 0 && (
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {dataset.map(entry => (
              <div
                key={entry.idx}
                style={{
                  textAlign: 'center',
                  cursor: 'pointer',
                  opacity: activeLetter === entry.idx ? 1 : 0.6,
                  transition: 'opacity 100ms',
                }}
                onClick={() => handleLoadLetter(entry)}
              >
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(5, 6px)',
                  gap: 1,
                  marginBottom: 4,
                }}>
                  {entry.grade.slice(0, 35).map((v, i) => (
                    <div
                      key={i}
                      style={{
                        width: 6,
                        height: 6,
                        background: v >= 1 ? 'var(--pink)' : 'var(--surface-low)',
                      }}
                    />
                  ))}
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 9,
                  fontWeight: 700,
                  color: activeLetter === entry.idx ? 'var(--primary-glow)' : 'var(--on-surface)',
                }}>{entry.letra}</div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* ===== Interactive Classification ===== */}
      <Card title="Classificacao Interativa" style={{ marginBottom: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'start' }}>
          <div>
            <PixelGrid
              rows={7}
              cols={5}
              cellSize={28}
              values={testPixels}
              onChange={(vals) => { setTestPixels(vals); setActiveLetter(null); setClassifyResp(null); }}
            />
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button
                className="btn btn-ghost"
                style={{ fontSize: 10, padding: '6px 12px' }}
                onClick={() => { setTestPixels(new Array(35).fill(-1)); setClassifyResp(null); setActiveLetter(null); }}
              >LIMPAR</button>
              <button
                className="btn btn-primary"
                style={{ fontSize: 10, padding: '6px 12px' }}
                onClick={handleClassify}
                disabled={!trained}
              >CLASSIFICAR</button>
            </div>
          </div>

          <div>
            {classifyResp ? (
              <>
                <div className="result-big">{classifyResp.letra}</div>
                <div className="result-label" style={{ marginBottom: 16 }}>
                  score: {classifyResp.top5[0]?.score.toFixed(4)}
                </div>
                <div className="conf-list">
                  {classifyResp.top5.map((c, i) => {
                    // Normalize bars: top score is 100%
                    const maxScore = classifyResp.top5[0].score;
                    const minScore = classifyResp.top5[classifyResp.top5.length - 1].score;
                    const range = maxScore - minScore || 1;
                    const pct = Math.max(5, ((c.score - minScore) / range) * 100);
                    return (
                      <div className="conf-row" key={i}>
                        <span className="conf-letter">{c.letra}</span>
                        <div className="conf-bar-wrap">
                          <div className="conf-bar-fill" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="conf-score">{c.score.toFixed(4)}</span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="empty">
                <div className="empty-icon">?</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)' }}>
                  {trained
                    ? 'Desenhe uma letra e clique CLASSIFICAR'
                    : 'Treine a rede primeiro'}
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* ===== Details ===== */}
      <Card title="Detalhes">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div><b style={{ color: 'var(--cyan)' }}>Modelo:</b> MADALINE com 13 unidades ADALINE + camada OR</div>
          <div><b style={{ color: 'var(--cyan)' }}>Entrada:</b> grade 5x7 bipolar (-1/+1) = 35 entradas</div>
          <div><b style={{ color: 'var(--cyan)' }}>Treinamento:</b> MRII (Minimo Impacto) &mdash; ajusta pesos apenas no neuronio de menor |y_in|</div>
          <div><b style={{ color: 'var(--cyan)' }}>Ativacao:</b> degrau bipolar</div>
          <div><b style={{ color: 'var(--cyan)' }}>Dataset:</b> 13 letras (A-M) em grid 5x7</div>
          <div><b style={{ color: 'var(--cyan)' }}>Classificacao:</b> maior y_in entre as 13 unidades</div>
        </div>
      </Card>
    </div>
  );
}
