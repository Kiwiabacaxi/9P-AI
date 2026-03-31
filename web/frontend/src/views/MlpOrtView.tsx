import { useState, useEffect, useRef, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import NetworkViz from '../components/viz/NetworkViz';
import LogChart from '../components/viz/LogChart';
import PixelGrid from '../components/shared/PixelGrid';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type { OrtResult, OrtClassifyResp, OrtDatasetInfo } from '../api/types';

// ---------------------------------------------------------------------------
// Orthogonal vector generation (client-side, mirrors Go's GerarVetoresOrtogonais)
// ---------------------------------------------------------------------------

interface OrtStep {
  vetores: number[][];
  dims: number;
}

function gerarVetoresOrtogonais(): OrtStep[] {
  let vetores: number[][] = [[1, 1], [1, -1]];
  const steps: OrtStep[] = [{ vetores: vetores.map(v => [...v]), dims: 2 }];
  for (let p = 0; p < 4; p++) {
    const novos: number[][] = [];
    for (const v of vetores) {
      novos.push([...v, ...v]);
      novos.push([...v, ...v.map(x => -x)]);
    }
    vetores = novos;
    steps.push({ vetores: vetores.map(v => [...v]), dims: vetores[0].length });
  }
  return steps;
}

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MlpOrtView() {
  const toast = useToast();

  // Config
  const [nHid, setNHid] = useState(15);
  const [alfa, setAlfa] = useState(0.01);
  const [maxCiclo, setMaxCiclo] = useState(50000);

  // Training state
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<OrtResult | null>(null);
  const [erroHist, setErroHist] = useState<number[]>([]);
  const sseCleanup = useRef<(() => void) | null>(null);
  const [activeLayer, setActiveLayer] = useState(-1);

  // Dataset from server
  const [dataset, setDataset] = useState<OrtDatasetInfo | null>(null);

  // Orthogonal vector steps (client-side)
  const [ortSteps] = useState<OrtStep[]>(() => gerarVetoresOrtogonais());
  const [activeStep, setActiveStep] = useState(0);

  // Demo classification (dropdown)
  const [demoLetter, setDemoLetter] = useState('');
  const [demoResp, setDemoResp] = useState<OrtClassifyResp | null>(null);

  // Interactive test
  const [testPixels, setTestPixels] = useState<number[]>(new Array(35).fill(-1));
  const [testResp, setTestResp] = useState<OrtClassifyResp | null>(null);

  // Fetch dataset on mount
  useEffect(() => {
    apiGet<OrtDatasetInfo>('/mlport/dataset').then(setDataset).catch(() => {});
  }, []);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => { sseCleanup.current?.(); };
  }, []);

  // ---------- Actions ----------

  const handleReset = useCallback(async () => {
    sseCleanup.current?.();
    try {
      await apiPost('/mlport/reset');
      setResult(null);
      setErroHist([]);
      setTraining(false);
      setActiveLayer(-1);
      setDemoLetter('');
      setDemoResp(null);
      setTestResp(null);
      setTestPixels(new Array(35).fill(-1));
      toast.show('Rede resetada');
    } catch {
      toast.show('Erro ao resetar');
    }
  }, [toast]);

  const handleTrain = useCallback(async () => {
    if (training) return;
    try {
      await apiPost('/mlport/config', { nHid, alfa, maxCiclo });
    } catch {
      toast.show('Erro ao salvar config');
      return;
    }

    setTraining(true);
    setResult(null);
    setErroHist([]);
    setDemoResp(null);
    setTestResp(null);

    const cleanup = apiSSE('/mlport/train', {
      onMessage: (data) => {
        const step = data as { erroTotal: number; activeLayer: number };
        setActiveLayer(step.activeLayer);
        setErroHist(prev => {
          if (prev.length === 0 || step.erroTotal !== prev[prev.length - 1]) {
            return [...prev, step.erroTotal];
          }
          return prev;
        });
      },
      onDone: (data) => {
        const res = data as OrtResult;
        setResult(res);
        setErroHist(res.erroHistorico || []);
        setTraining(false);
        setActiveLayer(-1);
        toast.show(res.convergiu ? 'Convergiu!' : 'Limite de ciclos atingido');
      },
      onError: () => {
        setTraining(false);
        setActiveLayer(-1);
        toast.show('Erro no treinamento');
      },
    });
    sseCleanup.current = cleanup;
  }, [training, nHid, alfa, maxCiclo, toast]);

  const handleDemoSelect = useCallback(async (letter: string) => {
    setDemoLetter(letter);
    if (!letter || !dataset) { setDemoResp(null); return; }
    const idx = LETTERS.indexOf(letter);
    if (idx < 0) return;
    const grade = dataset.letras[idx].grade;
    try {
      const resp = await apiPost<OrtClassifyResp>('/mlport/classify', { grade });
      setDemoResp(resp);
    } catch {
      toast.show('Erro ao classificar');
    }
  }, [dataset, toast]);

  const classifyTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const autoClassify = useCallback((pixels: number[]) => {
    setTestPixels(pixels);
    if (classifyTimeout.current) clearTimeout(classifyTimeout.current);
    if (result && pixels.some(v => v === 1)) {
      classifyTimeout.current = setTimeout(async () => {
        try {
          const resp = await apiPost<OrtClassifyResp>('/mlport/classify', { grade: pixels });
          setTestResp(resp);
        } catch { /* ignore */ }
      }, 150);
    } else {
      setTestResp(null);
    }
  }, [result]);

  // ---------- Render helpers ----------

  const trained = result !== null;

  return (
    <div>
      {/* ===== Page Header ===== */}
      <div className="page-header">
        <div>
          <div className="page-title">MLP <span>Ortogonal</span></div>
          <div className="page-sub">
            Classificacao A-Z com vetores bipolares ortogonais e distancia euclidiana &mdash; Aula 06
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={training}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            TREINAR REDE
          </button>
        </div>
      </div>

      {/* ===== Config Panel ===== */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <Card title="Neuronios Ocultos">
          <div className="porta-chips">
            {[10, 15, 20, 30].map(n => (
              <button
                key={n}
                className={`porta-chip${nHid === n ? ' selected' : ''}`}
                onClick={() => setNHid(n)}
                disabled={training}
              >{n}</button>
            ))}
          </div>
        </Card>
        <Card title="Learning Rate">
          <div className="porta-chips">
            {[0.005, 0.01, 0.02, 0.05].map(a => (
              <button
                key={a}
                className={`porta-chip${alfa === a ? ' selected' : ''}`}
                onClick={() => setAlfa(a)}
                disabled={training}
              >{a}</button>
            ))}
          </div>
        </Card>
        <Card title="Max Epocas">
          <div className="porta-chips">
            {[10000, 50000, 100000].map(m => (
              <button
                key={m}
                className={`porta-chip${maxCiclo === m ? ' selected' : ''}`}
                onClick={() => setMaxCiclo(m)}
                disabled={training}
              >{m >= 1000 ? `${m / 1000}k` : m}</button>
            ))}
          </div>
        </Card>
      </div>

      {/* ===== Orthogonal Vector Construction ===== */}
      <Card title="Construcao dos Vetores Bipolares Ortogonais" style={{ marginBottom: 24 }}>
        <div className="porta-chips" style={{ marginBottom: 16 }}>
          {ortSteps.map((s, i) => (
            <button
              key={i}
              className={`porta-chip${activeStep === i ? ' selected' : ''}`}
              onClick={() => setActiveStep(i)}
            >
              Passo {i} ({s.dims}x{s.dims})
            </button>
          ))}
        </div>

        <div style={{ overflowX: 'auto', marginBottom: 12 }}>
          <OrtVectorTable step={ortSteps[activeStep]} stepIdx={activeStep} />
        </div>

        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', lineHeight: 1.6 }}>
          Construcao recursiva: comecar com v0=[1,1] e v1=[1,-1].
          A cada passo, cada vetor <b>v</b> gera dois novos: <b>(v,v)</b> e <b>(v,-v)</b>.{' '}
          Apos 4 expansoes: 32 vetores ortogonais de 32 dimensoes.
          Para 26 letras, usamos os primeiros 26.
        </div>
      </Card>

      {/* ===== Letter Map ===== */}
      {dataset && (
        <Card title="Mapa Letra &rarr; Vetor Ortogonal" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {dataset.letras.map((l, i) => (
              <LetterCard key={i} nome={l.nome} grade={l.grade} vetor={l.vetor} />
            ))}
          </div>
        </Card>
      )}

      {/* ===== Network Viz ===== */}
      <Card title="Arquitetura" style={{ marginBottom: 24 }}>
        <NetworkViz
          layerSizes={[35, nHid, 32]}
          activeLayer={activeLayer}
          hudText={`35in \u00b7 tanh \u00b7 32out`}
          animate={!training}
        />
      </Card>

      {/* ===== Metrics ===== */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <MetricCard
          title="Ciclos"
          value={result ? result.ciclos.toLocaleString() : '--'}
          label={result ? (result.convergiu ? 'convergiu' : 'limite atingido') : 'aguardando'}
          color="cyan"
          pulse={training}
        />
        <MetricCard
          title="Acuracia"
          value={result ? `${result.acuracia.toFixed(1)}%` : '--'}
          label={result ? `${result.acertos}/${result.total} corretas` : 'aguardando'}
          color="green"
        />
        <MetricCard
          title="Status"
          value={training ? 'TREINANDO' : (trained ? 'PRONTO' : 'IDLE')}
          label={result ? `erro final: ${result.erroFinal.toFixed(4)}` : 'nenhum resultado'}
          color={training ? 'pink' : (trained ? 'green' : undefined)}
          pulse={training}
        />
      </div>

      {/* ===== Error Chart ===== */}
      {erroHist.length > 0 && (
        <Card title="Curva de Erro (log)" style={{ marginBottom: 24 }}>
          <LogChart data={erroHist} />
        </Card>
      )}

      {/* ===== Euclidean Distance Demo ===== */}
      {trained && dataset && (
        <Card title="Classificacao por Distancia Euclidiana" style={{ marginBottom: 24 }}>
          <div style={{ marginBottom: 16 }}>
            <Select
              label="Selecionar letra do dataset"
              options={[
                { value: '', label: '-- selecione --' },
                ...LETTERS.map(l => ({ value: l, label: l })),
              ]}
              value={demoLetter}
              onChange={handleDemoSelect}
            />
          </div>

          {demoResp && demoLetter && (
            <DistanceTable
              resp={demoResp}
              vetores={dataset.vetores}
              selectedLetter={demoLetter}
            />
          )}

          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)',
            marginTop: 12, lineHeight: 1.6,
          }}>
            D = sqrt( sum_k (t_k - y_k)^2 ) &mdash; a letra com <b>menor</b> distancia euclidiana eh a classificacao.
          </div>
        </Card>
      )}

      {/* ===== Interactive Test ===== */}
      {trained && (
        <Card title="Teste Interativo" style={{ marginBottom: 24 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'start' }}>
            <div>
              <PixelGrid
                rows={7}
                cols={5}
                cellSize={28}
                values={testPixels}
                onChange={autoClassify}
                showClear={false}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 10, padding: '6px 12px' }}
                  onClick={() => { setTestPixels(new Array(35).fill(-1)); setTestResp(null); }}
                >LIMPAR</button>
              </div>
            </div>

            <div>
              {testResp ? (
                <>
                  <div className="result-big">{testResp.letra}</div>
                  <div className="result-label" style={{ marginBottom: 16 }}>
                    distancia: {testResp.distancias[testResp.letraIdx].toFixed(4)}
                  </div>
                  <div className="conf-list">
                    {testResp.top5.map((c, i) => {
                      const maxDist = testResp.top5[testResp.top5.length - 1].distancia || 1;
                      const pct = Math.max(0, 100 - (c.distancia / maxDist) * 100);
                      return (
                        <div className="conf-row" key={i}>
                          <span className="conf-letter">{c.letra}</span>
                          <div className="conf-bar-wrap">
                            <div className="conf-bar-fill" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="conf-score">{c.distancia.toFixed(4)}</span>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                <div className="empty">
                  <div className="empty-icon">?</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)' }}>
                    Desenhe uma letra e clique CLASSIFICAR
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* ===== Details Card ===== */}
      <Card title="Detalhes">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div><b style={{ color: 'var(--cyan)' }}>Modelo:</b> MLP 35-{nHid}-32 com ativacao tanh</div>
          <div><b style={{ color: 'var(--cyan)' }}>Targets:</b> 32 vetores bipolares ortogonais (Fausett 1994)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Classificacao:</b> distancia euclidiana entre saida da rede e cada vetor-alvo</div>
          <div><b style={{ color: 'var(--cyan)' }}>Entrada:</b> grade 5x7 bipolar (-1/+1) = 35 entradas</div>
          <div><b style={{ color: 'var(--cyan)' }}>Saida:</b> 32 neuronios (tanh), sem limiar &mdash; vetores continuos</div>
          <div><b style={{ color: 'var(--cyan)' }}>Referencia:</b> Manzan 2016, Aula 06</div>
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Orthogonal Vector Table sub-component
// ---------------------------------------------------------------------------

function OrtVectorTable({ step, stepIdx }: { step: OrtStep; stepIdx: number }) {
  const { vetores, dims } = step;
  const halfDims = dims / 2;

  return (
    <table className="data-table" style={{ fontSize: 10 }}>
      <thead>
        <tr>
          <th style={{ width: 40 }}>#</th>
          {Array.from({ length: dims }, (_, i) => (
            <th key={i} style={{ textAlign: 'center', padding: '4px 2px', minWidth: 20 }}>
              {i}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {vetores.map((v, rowIdx) => (
          <tr key={rowIdx}>
            <td style={{ color: 'var(--cyan)', fontWeight: 700 }}>v{rowIdx}</td>
            {v.map((val, colIdx) => {
              // Color coding: first half = (v,v) part = yellow bg; second half = (v,-v) new part = green bg
              // Only color the background if we're past step 0
              let bg = 'transparent';
              if (stepIdx > 0) {
                bg = colIdx < halfDims
                  ? 'rgba(255, 200, 0, 0.08)'
                  : 'rgba(0, 255, 0, 0.08)';
              }
              return (
                <td
                  key={colIdx}
                  style={{
                    textAlign: 'center',
                    padding: '4px 2px',
                    color: val === 1 ? 'var(--secondary)' : 'var(--pink)',
                    fontWeight: 700,
                    background: bg,
                  }}
                >
                  {val === 1 ? '+1' : '-1'}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// Letter Card (mini grid + ortogonal vector)
// ---------------------------------------------------------------------------

function LetterCard({ nome, grade, vetor }: { nome: string; grade: number[]; vetor: number[] }) {
  return (
    <div style={{
      background: 'var(--surface-low)',
      padding: 8,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 4,
      minWidth: 52,
    }}>
      {/* Mini 5x7 pixel grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 6px)',
        gap: 1,
      }}>
        {grade.slice(0, 35).map((v, i) => (
          <div
            key={i}
            style={{
              width: 6,
              height: 6,
              background: v === 1 ? 'var(--pink)' : 'var(--surface)',
            }}
          />
        ))}
      </div>

      {/* Letter name */}
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        fontWeight: 700,
        color: 'var(--secondary)',
      }}>{nome}</div>

      {/* Compact orthogonal vector */}
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 7,
        lineHeight: 1.2,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        maxWidth: 48,
      }}>
        {vetor.map((v, i) => (
          <span key={i} style={{ color: v === 1 ? 'var(--on-surface)' : 'var(--pink)' }}>
            {v === 1 ? '+' : '-'}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Euclidean Distance Table (professor's slide style)
// ---------------------------------------------------------------------------

function DistanceTable({
  resp,
  vetores,
  selectedLetter,
}: {
  resp: OrtClassifyResp;
  vetores: number[][];
  selectedLetter: string;
}) {
  // Show first 8 letters (A-H) and first 8 of 32 dims for readability
  const showLetters = 8;
  const showDims = 8;

  const minDist = Math.min(...resp.distancias.slice(0, showLetters));

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table" style={{ fontSize: 10 }}>
        <thead>
          <tr>
            <th style={{ minWidth: 90 }}>dim</th>
            <th style={{ color: 'var(--pink)' }}>saida rede</th>
            {LETTERS.slice(0, showLetters).map(l => (
              <th key={l} style={{
                textAlign: 'center',
                color: l === selectedLetter ? 'var(--primary-glow)' : undefined,
              }}>
                t({l})
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: showDims }, (_, d) => (
            <tr key={d}>
              <td style={{ color: 'var(--cyan)' }}>k={d}</td>
              <td style={{ color: 'var(--pink)', fontVariantNumeric: 'tabular-nums' }}>
                {formatNum(resp.saidaRede[d])}
              </td>
              {Array.from({ length: showLetters }, (_, li) => (
                <td
                  key={li}
                  style={{
                    textAlign: 'center',
                    color: vetores[li][d] === 1 ? 'var(--secondary)' : 'var(--pink)',
                    fontWeight: 700,
                  }}
                >
                  {vetores[li][d] === 1 ? '+1' : '-1'}
                </td>
              ))}
            </tr>
          ))}

          {/* Ellipsis row */}
          <tr>
            <td colSpan={2 + showLetters} style={{ textAlign: 'center', color: 'var(--on-surface)', opacity: 0.4 }}>
              ... (mostrando 8 de 32 dimensoes) ...
            </td>
          </tr>

          {/* Distance row */}
          <tr style={{ borderTop: '2px solid var(--border)' }}>
            <td colSpan={2} style={{ fontWeight: 700, color: 'var(--cyan)' }}>
              Dist. Eucl.
            </td>
            {Array.from({ length: showLetters }, (_, li) => {
              const dist = resp.distancias[li];
              const isMin = Math.abs(dist - minDist) < 1e-9;
              return (
                <td
                  key={li}
                  style={{
                    textAlign: 'center',
                    fontWeight: 700,
                    fontVariantNumeric: 'tabular-nums',
                    color: isMin ? '#013a00' : 'var(--on-surface)',
                    background: isMin ? 'rgba(255, 200, 0, 0.6)' : 'transparent',
                  }}
                >
                  {dist.toFixed(2)}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function formatNum(v: number): string {
  if (v === undefined || v === null) return '--';
  return v.toFixed(4);
}
