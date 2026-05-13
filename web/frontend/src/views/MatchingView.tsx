import { useEffect, useState } from 'react';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  MatchingScenario, MatchingScenarioMeta,
  MatchingStep, MatchingResult, MatchingTraderStats,
  MatchingBaselineResp,
  MatchingStepNSGA, MatchingResultNSGA, MatchingFrontPoint,
} from '../api/types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer,
} from 'recharts';
import MatchingMap from '../components/viz/MatchingMap';

type Mode = 'single' | 'nsga2';

function FitnessChart({ history }: { history: MatchingStep[] }) {
  if (history.length === 0) return null;
  const data = history.map(s => ({
    gen: s.geracao,
    melhor: s.melhorFitness,
    media: s.mediaFitness,
  }));
  return (
    <div style={{ height: 180, marginTop: 8 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="gen" tick={{ fontSize: 10, fill: 'var(--muted)' }} stroke="var(--border)" />
          <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} stroke="var(--border)" />
          <RTooltip contentStyle={{ background: 'var(--surface-high)', border: '1px solid var(--border)', color: 'var(--primary)', fontSize: 11 }} />
          <Line type="monotone" dataKey="melhor" stroke="var(--primary-glow)" dot={false} />
          <Line type="monotone" dataKey="media" stroke="var(--cyan)" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ParetoScatter({ points, selected, onPick }: {
  points: MatchingFrontPoint[];
  selected: MatchingFrontPoint | null;
  onPick: (p: MatchingFrontPoint) => void;
}) {
  if (points.length === 0) return null;
  // dedupe por (sup, inc, div) pra não plotar dezenas de pontos sobrepostos
  const seen = new Set<string>();
  const unique = points.filter(p => {
    const key = `${p.superavit.toFixed(0)}|${p.inclusao.toFixed(3)}|${p.diversidade.toFixed(3)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  // viewBox em px; vamos mapear superavit→x, diversidade→y manualmente
  const W = 280, H = 180;
  const padL = 32, padR = 8, padT = 8, padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xs = unique.map(p => p.superavit);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xSpan = xMax - xMin || 1;
  const yMin = 0, yMax = 1;

  const xPx = (sup: number) => padL + ((sup - xMin) / xSpan) * plotW;
  const yPx = (div: number) => padT + (1 - (div - yMin) / (yMax - yMin)) * plotH;

  const isSelected = (p: MatchingFrontPoint) =>
    selected != null
    && Math.abs(p.superavit - selected.superavit) < 0.5
    && Math.abs(p.diversidade - selected.diversidade) < 1e-4
    && Math.abs(p.inclusao - selected.inclusao) < 1e-4;

  return (
    <div style={{ marginTop: 8 }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 200, display: 'block' }}>
        {/* grid */}
        {[0.25, 0.5, 0.75, 1].map(y => (
          <line key={`gy-${y}`} x1={padL} y1={yPx(y)} x2={W - padR} y2={yPx(y)} stroke="var(--border)" strokeDasharray="3 3" />
        ))}
        {[0, 0.5, 1].map(t => {
          const x = padL + t * plotW;
          const v = xMin + t * xSpan;
          return (
            <g key={`gx-${t}`}>
              <line x1={x} y1={padT} x2={x} y2={H - padB} stroke="var(--border)" strokeDasharray="3 3" />
              <text x={x} y={H - padB + 12} fontSize="9" fill="var(--muted)" textAnchor="middle">{`${(v / 1000).toFixed(0)}k`}</text>
            </g>
          );
        })}
        {[0, 0.5, 1].map(y => (
          <text key={`yl-${y}`} x={padL - 4} y={yPx(y) + 3} fontSize="9" fill="var(--muted)" textAnchor="end">{y.toFixed(1)}</text>
        ))}
        {/* axis labels */}
        <text x={padL + plotW / 2} y={H - 4} fontSize="9" fill="var(--muted)" textAnchor="middle">superávit R$</text>
        <text x={4} y={padT + plotH / 2} fontSize="9" fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 4 ${padT + plotH / 2})`}>1 − HHI</text>
        {/* points */}
        {unique.map((p, i) => {
          const sel = isSelected(p);
          const r = 3 + p.inclusao * 5; // 3..8 px conforme inclusão
          return (
            <circle
              key={i}
              cx={xPx(p.superavit)}
              cy={yPx(p.diversidade)}
              r={sel ? r + 3 : r}
              fill={sel ? 'var(--pink)' : 'var(--primary-glow)'}
              fillOpacity={sel ? 1 : 0.7}
              stroke={sel ? 'var(--primary)' : 'transparent'}
              strokeWidth={sel ? 1.5 : 0}
              style={{ cursor: 'pointer' }}
              onClick={() => onPick(p)}
            >
              <title>{`sup=R$${p.superavit.toFixed(0)} · inc=${(p.inclusao * 100).toFixed(0)}% · div=${p.diversidade.toFixed(2)}`}</title>
            </circle>
          );
        })}
      </svg>
      <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4, textAlign: 'center' }}>
        clique num ponto · tamanho = inclusão · {unique.length} pontos únicos
      </div>
    </div>
  );
}

function TraderCards({ scenario, traderStats }: {
  scenario: MatchingScenario;
  traderStats: MatchingTraderStats[] | null;
}) {
  if (!traderStats) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ fontSize: 13, marginBottom: 6 }}>Traders</h3>
      {scenario.traders.map(t => {
        const st = traderStats.find(s => s.traderId === t.id);
        if (!st) return null;
        const pct = (st.volumeAlocadoT / t.capacidadeT) * 100;
        const status = st.overCapacity ? 'over' : st.underSpec ? 'under' : st.numLotes > 0 ? 'ok' : '—';
        const bg = st.overCapacity ? 'rgba(230, 57, 70, 0.18)' : st.underSpec ? 'rgba(244, 162, 97, 0.18)' : 'var(--surface)';
        return (
          <div key={t.id} style={{
            padding: 8, marginBottom: 6,
            border: `2px solid ${t.cor}`, borderRadius: 4, fontSize: 12,
            background: bg,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong style={{ color: t.cor }}>{t.nome}</strong>
              <span style={{ color: 'var(--muted)' }}>{status}</span>
            </div>
            <div style={{ marginTop: 4, height: 6, background: 'var(--surface-high)', borderRadius: 3 }}>
              <div style={{
                width: `${Math.min(pct, 100)}%`, height: '100%',
                background: t.cor, borderRadius: 3,
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, color: 'var(--muted)' }}>
              <span>{st.volumeAlocadoT.toFixed(0)}/{t.capacidadeT.toFixed(0)} t</span>
              <span>{st.numLotes} lotes</span>
            </div>
            <div style={{ color: 'var(--muted)' }}>blend prot: {st.blendProteina.toFixed(2)} (≥ {t.proteinaMin})</div>
          </div>
        );
      })}
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  marginTop: 16,
  fontSize: 12,
  padding: 10,
  background: 'var(--surface)',
  border: '1px solid var(--border)',
  borderRadius: 4,
};

const mutedStyle: React.CSSProperties = {
  fontSize: 13,
  color: 'var(--muted)',
};

export default function MatchingView() {
  const [scenarios, setScenarios] = useState<MatchingScenarioMeta[]>([]);
  const [scenarioId, setScenarioId] = useState<string>('');
  const [scale, setScale] = useState<'small' | 'large'>('small');
  const [scenario, setScenario] = useState<MatchingScenario | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>('single');

  const [training, setTraining] = useState(false);
  const [step, setStep] = useState<MatchingStep | null>(null);
  const [result, setResult] = useState<MatchingResult | null>(null);
  const [baselineGreedy, setBaselineGreedy] = useState<MatchingBaselineResp | null>(null);
  const [baselineHungarian, setBaselineHungarian] = useState<MatchingBaselineResp | null>(null);
  const [history, setHistory] = useState<MatchingStep[]>([]);

  const [nsgaStep, setNsgaStep] = useState<MatchingStepNSGA | null>(null);
  const [nsgaResult, setNsgaResult] = useState<MatchingResultNSGA | null>(null);
  const [selectedFront, setSelectedFront] = useState<MatchingFrontPoint | null>(null);

  useEffect(() => {
    apiGet<MatchingScenarioMeta[]>('/matching/scenarios')
      .then(s => {
        setScenarios(s);
        if (s.length > 0) setScenarioId(s[0].id);
      })
      .catch(e => setErr(String(e)));
  }, []);

  async function loadScenario() {
    if (!scenarioId) return;
    setLoading(true); setErr(null); setStep(null); setResult(null);
    setBaselineGreedy(null); setBaselineHungarian(null);
    setNsgaStep(null); setNsgaResult(null); setSelectedFront(null);
    try {
      const s = await apiPost<MatchingScenario>('/matching/scenario', { id: scenarioId, seed: 42, scale });
      setScenario(s);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  function startTrainNSGA() {
    if (!scenario) return;
    setTraining(true); setErr(null);
    setNsgaStep(null); setNsgaResult(null); setSelectedFront(null);
    setBaselineGreedy(null); setBaselineHungarian(null);
    const stop = apiSSE('/matching/train-nsga2', {
      onMessage: (data: unknown) => setNsgaStep(data as MatchingStepNSGA),
      onDone: (data: unknown) => {
        const res = data as MatchingResultNSGA;
        setNsgaResult(res);
        if (res.front.length > 0) setSelectedFront(res.front[0]);
        setTraining(false);
      },
      onError: () => {
        setErr('erro no streaming');
        setTraining(false);
      },
    });
    void stop;
  }

  function startTrain() {
    if (!scenario) return;
    setTraining(true); setStep(null); setResult(null); setErr(null); setHistory([]);
    setBaselineGreedy(null); setBaselineHungarian(null);
    setNsgaStep(null); setNsgaResult(null); setSelectedFront(null);
    const stop = apiSSE('/matching/train', {
      onMessage: (data: unknown) => {
        const s = data as MatchingStep;
        setStep(s);
        setHistory(h => [...h, s]);
      },
      onDone: (data: unknown) => {
        setResult(data as MatchingResult);
        setTraining(false);
      },
      onError: () => {
        setErr('erro no streaming');
        setTraining(false);
      },
    });
    void stop;
  }

  async function runBaseline(algoritmo: 'greedy' | 'hungarian') {
    if (!scenario) return;
    setErr(null);
    try {
      const r = await apiPost<MatchingBaselineResp>('/matching/baseline', { algoritmo });
      if (algoritmo === 'greedy') setBaselineGreedy(r);
      else setBaselineHungarian(r);
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div className="view matching-view" style={{
      display: 'grid',
      gridTemplateColumns: '320px 1fr',
      gap: 16,
      height: 'calc(100vh - 140px)',
      minHeight: 600,
    }}>
      <aside className="matching-sidebar" style={{ overflowY: 'auto', paddingRight: 4 }}>
        <h2 style={{ marginBottom: 4 }}>Matching · Soja</h2>
        <p style={mutedStyle}>
          Matching marketplace single-objective ({scale === 'large' ? '60 produtores × 6 traders' : '6 produtores × 4 traders'} → Santos).
        </p>

        <div style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontSize: 12, marginBottom: 4, color: 'var(--muted)' }}>Cenário</label>
          <select
            value={scenarioId}
            onChange={e => setScenarioId(e.target.value)}
            style={{ width: '100%', padding: 6, background: 'var(--surface)', color: 'var(--primary)', border: '1px solid var(--border)', borderRadius: 3 }}
          >
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>{s.nome}</option>
            ))}
          </select>
          <label style={{ display: 'block', fontSize: 12, marginTop: 8, marginBottom: 4, color: 'var(--muted)' }}>Escala</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            <button
              className={scale === 'small' ? 'btn btn-primary' : 'btn btn-ghost'}
              onClick={() => setScale('small')}
              disabled={training}
              style={{ justifyContent: 'center', padding: '6px 8px' }}
            >
              6×4
            </button>
            <button
              className={scale === 'large' ? 'btn btn-primary' : 'btn btn-ghost'}
              onClick={() => setScale('large')}
              disabled={training}
              style={{ justifyContent: 'center', padding: '6px 8px' }}
            >
              60×6
            </button>
          </div>
          <button
            className="btn btn-primary"
            onClick={loadScenario}
            disabled={loading || !scenarioId || training}
            style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
          >
            {loading ? 'Carregando…' : 'Carregar cenário'}
          </button>
        </div>

        {err && <div className="error" style={{ marginTop: 8, color: '#e63946' }}>{err}</div>}

        {scenario && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ fontSize: 14, marginBottom: 4 }}>{scenario.nome}</h3>
            <p style={{ fontSize: 12, color: 'var(--muted)' }}>{scenario.descricao}</p>
            <ul style={{ fontSize: 12, paddingLeft: 16, color: 'var(--muted)' }}>
              <li>{scenario.producers.length} produtores</li>
              <li>{scenario.lots.length} lotes</li>
              <li>{scenario.traders.length} traders</li>
              <li>preço base: R${scenario.precoBase.toFixed(2)}/saca</li>
            </ul>
          </div>
        )}

        {scenario && (
          <div style={{ marginTop: 16 }}>
            <label style={{ display: 'block', fontSize: 12, marginBottom: 4, color: 'var(--muted)' }}>Modo</label>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 8 }}>
              <button
                className={mode === 'single' ? 'btn btn-primary' : 'btn btn-ghost'}
                onClick={() => setMode('single')}
                disabled={training}
                style={{ justifyContent: 'center', padding: '6px 8px' }}
              >
                Single-Obj
              </button>
              <button
                className={mode === 'nsga2' ? 'btn btn-primary' : 'btn btn-ghost'}
                onClick={() => setMode('nsga2')}
                disabled={training}
                style={{ justifyContent: 'center', padding: '6px 8px' }}
              >
                NSGA-II
              </button>
            </div>

            {mode === 'single' && (
              <button
                className="btn btn-primary"
                onClick={startTrain}
                disabled={training}
                style={{ width: '100%', marginBottom: 6, justifyContent: 'center' }}
              >
                {training ? `Treinando… (gen ${step?.geracao ?? 0})` : 'Treinar GA'}
              </button>
            )}
            {mode === 'nsga2' && (
              <button
                className="btn btn-primary"
                onClick={startTrainNSGA}
                disabled={training}
                style={{ width: '100%', marginBottom: 6, justifyContent: 'center' }}
              >
                {training ? `NSGA-II… (gen ${nsgaStep?.geracao ?? 0})` : 'Treinar NSGA-II'}
              </button>
            )}
            <button
              className="btn btn-ghost"
              onClick={() => runBaseline('greedy')}
              disabled={training}
              style={{ width: '100%', justifyContent: 'center', marginBottom: 4 }}
            >
              Baseline · Greedy
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => runBaseline('hungarian')}
              disabled={training}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              Baseline · Hungarian (sem cap)
            </button>
          </div>
        )}

        {mode === 'single' && step && (
          <div style={cardStyle}>
            <h3 style={{ fontSize: 13, marginBottom: 4 }}>Geração {step.geracao}</h3>
            <div>fitness: {step.melhorFitness.toFixed(0)}</div>
            <div>superávit: R${step.melhorSuperavit.toFixed(0)}</div>
            <div>matched: {step.numMatched}/{scenario?.lots.length ?? 0}</div>
            <div>violações: {step.melhorViolacoes}</div>
            <FitnessChart history={history} />
          </div>
        )}

        {mode === 'nsga2' && (nsgaStep || nsgaResult) && (
          <div style={{ ...cardStyle, borderColor: 'var(--primary-glow)' }}>
            <h3 style={{ fontSize: 13, marginBottom: 4, color: 'var(--primary-glow)' }}>
              NSGA-II {nsgaResult ? '· final' : `· gen ${nsgaStep?.geracao ?? 0}`}
            </h3>
            <div>fronteira: {(nsgaResult ?? nsgaStep)?.front.length ?? 0} pontos</div>
            {nsgaStep && !nsgaResult && (
              <>
                <div>melhor superávit: R${nsgaStep.bestSuperavit.toFixed(0)}</div>
                <div>melhor inclusão: {(nsgaStep.bestInclusao * 100).toFixed(0)}%</div>
                <div>melhor diversidade: {nsgaStep.bestDiversidade.toFixed(2)}</div>
                <div>feasible na pop: {nsgaStep.numFeasible}</div>
              </>
            )}
            <ParetoScatter
              points={(nsgaResult ?? nsgaStep)?.front ?? []}
              selected={selectedFront}
              onPick={setSelectedFront}
            />
          </div>
        )}

        {mode === 'nsga2' && selectedFront && (
          <div style={{ ...cardStyle, borderColor: 'var(--pink)' }}>
            <h3 style={{ fontSize: 13, marginBottom: 4, color: 'var(--pink)' }}>Ponto selecionado</h3>
            <div>superávit: R${selectedFront.superavit.toFixed(0)}</div>
            <div>inclusão: {(selectedFront.inclusao * 100).toFixed(0)}% ({selectedFront.numMatched}/{scenario?.lots.length ?? 0})</div>
            <div>diversidade: {selectedFront.diversidade.toFixed(2)}</div>
            <div>violações: {selectedFront.violacoes}</div>
          </div>
        )}

        {baselineGreedy && (
          <div style={{ ...cardStyle, borderColor: 'var(--cyan)' }}>
            <h3 style={{ fontSize: 13, marginBottom: 4, color: 'var(--cyan)' }}>Baseline · Greedy</h3>
            <div>fitness: {baselineGreedy.breakdown.Fitness.toFixed(0)}</div>
            <div>superávit: R${baselineGreedy.breakdown.SuperavitTotal.toFixed(0)}</div>
            <div>matched: {baselineGreedy.breakdown.NumMatched}</div>
            <div>violações: {baselineGreedy.breakdown.Violacoes}</div>
          </div>
        )}

        {baselineHungarian && (
          <div style={{ ...cardStyle, borderColor: 'var(--pink)' }}>
            <h3 style={{ fontSize: 13, marginBottom: 4, color: 'var(--pink)' }}>Baseline · Hungarian (sem cap)</h3>
            <div>fitness: {baselineHungarian.breakdown.Fitness.toFixed(0)}</div>
            <div>superávit: R${baselineHungarian.breakdown.SuperavitTotal.toFixed(0)}</div>
            <div>matched: {baselineHungarian.breakdown.NumMatched}</div>
            <div>violações: {baselineHungarian.breakdown.Violacoes}</div>
            {baselineHungarian.breakdown.Violacoes > 0 && (
              <div style={{ marginTop: 4, color: 'var(--pink)', fontSize: 11 }}>
                ↑ ignora capacidade — viola limites
              </div>
            )}
          </div>
        )}

        {result && !training && (
          <div style={{ ...cardStyle, borderColor: 'var(--primary-glow)' }}>
            <h3 style={{ fontSize: 13, marginBottom: 4, color: 'var(--primary-glow)' }}>Resultado final</h3>
            <div>gerações: {result.geracoes}</div>
            <div>fitness: {result.melhorFitness.toFixed(0)}</div>
          </div>
        )}

        {scenario && mode === 'single' && (step || result) && (
          <TraderCards
            scenario={scenario}
            traderStats={step?.traderStats ?? result?.traderStats ?? null}
          />
        )}
        {scenario && mode === 'nsga2' && selectedFront && (
          <TraderCards
            scenario={scenario}
            traderStats={selectedFront.traderStats}
          />
        )}
      </aside>

      <div className="matching-map-area" style={{ height: '100%', minHeight: 500, border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}>
        <MatchingMap
          scenario={scenario}
          chromosome={
            mode === 'nsga2'
              ? (selectedFront?.chrom ?? null)
              : (step?.melhorCrom ?? result?.melhorCrom ?? null)
          }
          traderStats={
            mode === 'nsga2'
              ? (selectedFront?.traderStats ?? null)
              : (step?.traderStats ?? result?.traderStats ?? null)
          }
        />
      </div>
    </div>
  );
}
