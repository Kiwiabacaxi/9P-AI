import { useEffect, useState } from 'react';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  MatchingScenario, MatchingScenarioMeta,
  MatchingStep, MatchingResult, MatchingTraderStats,
  MatchingBaselineResp,
} from '../api/types';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer,
} from 'recharts';
import MatchingMap from '../components/viz/MatchingMap';

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
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="gen" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <RTooltip />
          <Line type="monotone" dataKey="melhor" stroke="#2a9d8f" dot={false} />
          <Line type="monotone" dataKey="media" stroke="#e9c46a" dot={false} />
        </LineChart>
      </ResponsiveContainer>
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
      <h3 style={{ fontSize: 13 }}>Traders</h3>
      {scenario.traders.map(t => {
        const st = traderStats.find(s => s.traderId === t.id);
        if (!st) return null;
        const pct = (st.volumeAlocadoT / t.capacidadeT) * 100;
        const status = st.overCapacity ? 'over' : st.underSpec ? 'under' : st.numLotes > 0 ? 'ok' : '—';
        return (
          <div key={t.id} style={{
            padding: 8, marginBottom: 6,
            border: `2px solid ${t.cor}`, borderRadius: 4, fontSize: 12,
            background: st.overCapacity ? '#ffe6e6' : st.underSpec ? '#fff5e6' : 'white',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong style={{ color: t.cor }}>{t.nome}</strong>
              <span>{status}</span>
            </div>
            <div style={{ marginTop: 4, height: 6, background: '#eee', borderRadius: 3 }}>
              <div style={{
                width: `${Math.min(pct, 100)}%`, height: '100%',
                background: t.cor, borderRadius: 3,
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, color: '#666' }}>
              <span>{st.volumeAlocadoT.toFixed(0)}/{t.capacidadeT.toFixed(0)} t</span>
              <span>{st.numLotes} lotes</span>
            </div>
            <div style={{ color: '#666' }}>blend prot: {st.blendProteina.toFixed(2)} (≥ {t.proteinaMin})</div>
          </div>
        );
      })}
    </div>
  );
}

export default function MatchingView() {
  const [scenarios, setScenarios] = useState<MatchingScenarioMeta[]>([]);
  const [scenarioId, setScenarioId] = useState<string>('');
  const [scenario, setScenario] = useState<MatchingScenario | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [training, setTraining] = useState(false);
  const [step, setStep] = useState<MatchingStep | null>(null);
  const [result, setResult] = useState<MatchingResult | null>(null);
  const [baseline, setBaseline] = useState<MatchingBaselineResp | null>(null);
  const [history, setHistory] = useState<MatchingStep[]>([]);

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
    setLoading(true); setErr(null); setStep(null); setResult(null); setBaseline(null);
    try {
      const s = await apiPost<MatchingScenario>('/matching/scenario', { id: scenarioId, seed: 42 });
      setScenario(s);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  function startTrain() {
    if (!scenario) return;
    setTraining(true); setStep(null); setResult(null); setErr(null); setHistory([]);
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

  async function runBaseline() {
    if (!scenario) return;
    setErr(null);
    try {
      const r = await apiPost<MatchingBaselineResp>('/matching/baseline', { algoritmo: 'greedy' });
      setBaseline(r);
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div className="view matching-view" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 16, height: '100%' }}>
      <aside className="matching-sidebar" style={{ padding: 12, overflowY: 'auto' }}>
        <h2>Matching · Soja</h2>
        <p style={{ fontSize: 13, color: '#666' }}>
          Etapa 1: matching marketplace single-objective (6 produtores × 4 traders → Santos).
        </p>

        <div style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontSize: 12, marginBottom: 4 }}>Cenário</label>
          <select
            value={scenarioId}
            onChange={e => setScenarioId(e.target.value)}
            style={{ width: '100%', padding: 6 }}
          >
            {scenarios.map(s => (
              <option key={s.id} value={s.id}>{s.nome}</option>
            ))}
          </select>
          <button
            onClick={loadScenario}
            disabled={loading || !scenarioId || training}
            style={{ marginTop: 8, width: '100%' }}
          >
            {loading ? 'Carregando…' : 'Carregar cenário'}
          </button>
        </div>

        {err && <div className="error" style={{ marginTop: 8, color: '#e63946' }}>{err}</div>}

        {scenario && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ fontSize: 14 }}>{scenario.nome}</h3>
            <p style={{ fontSize: 12, color: '#666' }}>{scenario.descricao}</p>
            <ul style={{ fontSize: 12, paddingLeft: 16 }}>
              <li>{scenario.producers.length} produtores</li>
              <li>{scenario.lots.length} lotes</li>
              <li>{scenario.traders.length} traders</li>
              <li>preço base: R${scenario.precoBase.toFixed(2)}/saca</li>
            </ul>
          </div>
        )}

        {scenario && (
          <div style={{ marginTop: 16 }}>
            <button
              onClick={startTrain}
              disabled={training}
              style={{ width: '100%', marginBottom: 4 }}
            >
              {training ? `Treinando… (gen ${step?.geracao ?? 0})` : 'Treinar GA'}
            </button>
            <button onClick={runBaseline} disabled={training} style={{ width: '100%' }}>
              Rodar baseline (greedy)
            </button>
          </div>
        )}

        {step && (
          <div style={{ marginTop: 16, fontSize: 12 }}>
            <h3 style={{ fontSize: 13 }}>Geração {step.geracao}</h3>
            <div>fitness: {step.melhorFitness.toFixed(0)}</div>
            <div>superávit: R${step.melhorSuperavit.toFixed(0)}</div>
            <div>matched: {step.numMatched}/{scenario?.lots.length ?? 0}</div>
            <div>violações: {step.melhorViolacoes}</div>
            <FitnessChart history={history} />
          </div>
        )}

        {baseline && (
          <div style={{ marginTop: 16, fontSize: 12, padding: 8, background: '#f6f6f6' }}>
            <h3 style={{ fontSize: 13 }}>Baseline (greedy)</h3>
            <div>fitness: {baseline.breakdown.Fitness.toFixed(0)}</div>
            <div>superávit: R${baseline.breakdown.SuperavitTotal.toFixed(0)}</div>
            <div>matched: {baseline.breakdown.NumMatched}</div>
            <div>violações: {baseline.breakdown.Violacoes}</div>
          </div>
        )}

        {result && !training && (
          <div style={{ marginTop: 16, fontSize: 12, padding: 8, background: '#eef9f4' }}>
            <h3 style={{ fontSize: 13 }}>Resultado final</h3>
            <div>gerações: {result.geracoes}</div>
            <div>fitness: {result.melhorFitness.toFixed(0)}</div>
          </div>
        )}

        {scenario && (step || result) && (
          <TraderCards
            scenario={scenario}
            traderStats={step?.traderStats ?? result?.traderStats ?? null}
          />
        )}
      </aside>

      <div className="matching-map-area" style={{ height: '100%', minHeight: 500 }}>
        <MatchingMap
          scenario={scenario}
          chromosome={step?.melhorCrom ?? result?.melhorCrom ?? null}
          traderStats={step?.traderStats ?? result?.traderStats ?? null}
        />
      </div>
    </div>
  );
}
