import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import type { MatchingScenario, MatchingScenarioMeta } from '../api/types';
import MatchingMap from '../components/viz/MatchingMap';

export default function MatchingView() {
  const [scenarios, setScenarios] = useState<MatchingScenarioMeta[]>([]);
  const [scenarioId, setScenarioId] = useState<string>('');
  const [scenario, setScenario] = useState<MatchingScenario | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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
    setLoading(true); setErr(null);
    try {
      const s = await apiPost<MatchingScenario>('/matching/scenario', { id: scenarioId, seed: 42 });
      setScenario(s);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
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
            disabled={loading || !scenarioId}
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
      </aside>

      <div className="matching-map-area" style={{ height: '100%', minHeight: 500 }}>
        <MatchingMap scenario={scenario} chromosome={null} traderStats={null} />
      </div>
    </div>
  );
}
