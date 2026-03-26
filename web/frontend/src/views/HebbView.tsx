import { useState } from 'react';
import { apiPost } from '../api/client';
import { useToast } from '../components/shared/Toast';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';

const PORTAS = ['AND', 'OR', 'NAND', 'NOR', 'XOR'] as const;

interface HebbStep {
  amostra: number;
  x1: number;
  x2: number;
  target: number;
  w1: number;
  w2: number;
  bias: number;
}

interface HebbTest {
  x1: number;
  x2: number;
  target: number;
  yIn: number;
  predicao: number;
  acertou: boolean;
}

interface HebbResult {
  porta: string;
  w1: number;
  w2: number;
  bias: number;
  acertos: number;
  acuracia: number;
  steps: HebbStep[];
  testes: HebbTest[];
}

export default function HebbView() {
  const { show } = useToast();
  const [results, setResults] = useState<Record<string, HebbResult>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const current = selected ? results[selected] : null;

  async function trainGate(porta: string) {
    try {
      const res = await apiPost<HebbResult>(`/hebb/train?porta=${porta}`);
      setResults(prev => ({ ...prev, [porta]: res }));
      setSelected(porta);
      show(`Hebb ${porta} treinado`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  async function trainAll() {
    setLoading(true);
    for (const p of PORTAS) {
      await trainGate(p);
    }
    setLoading(false);
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">
            Regra de <span>Hebb</span>
          </div>
          <div className="page-sub">
            2 entradas + bias &middot; degrau bipolar &middot; atualiza sempre &middot; Trab 01
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={trainAll} disabled={loading}>
            TREINAR TODAS
          </button>
        </div>
      </div>

      {/* Gate chips */}
      <div className="porta-chips">
        {PORTAS.map(nome => {
          const trained = !!results[nome];
          const isSelected = nome === selected;
          const isXor = nome === 'XOR';
          const cls = [
            'porta-chip',
            isXor ? 'xor-chip' : '',
            trained ? 'trained' : '',
            isSelected ? 'selected' : '',
          ]
            .filter(Boolean)
            .join(' ');
          return (
            <button key={nome} className={cls} onClick={() => trainGate(nome)}>
              {nome}
            </button>
          );
        })}
      </div>

      {/* Empty state */}
      {!current && (
        <div className="empty">
          <div className="empty-icon">{'\u25C8'}</div>
          <div>Selecione uma porta para treinar</div>
        </div>
      )}

      {/* Results */}
      {current && (
        <>
          <div className="grid-3" style={{ marginBottom: 16 }}>
            <Card title="Pesos Finais">
              <div className="weights-row">
                <div className="weight-box">
                  <div className="weight-val">{current.w1.toFixed(1)}</div>
                  <div className="weight-lbl">W1</div>
                </div>
                <div className="weight-box">
                  <div className="weight-val">{current.w2.toFixed(1)}</div>
                  <div className="weight-lbl">W2</div>
                </div>
                <div className="weight-box">
                  <div className="weight-val">{current.bias.toFixed(1)}</div>
                  <div className="weight-lbl">BIAS</div>
                </div>
              </div>
            </Card>

            <MetricCard
              title="Acuracia"
              value={current.acuracia.toFixed(0) + '%'}
              label="nos dados de treino"
              color={current.acuracia === 100 ? 'green' : 'pink'}
            />

            <MetricCard
              title="Porta"
              value={current.porta}
              label=""
              valueStyle={{ color: 'var(--on-surface)', fontSize: 28 }}
            />
          </div>

          <div className="grid-2">
            {/* Training steps table */}
            <Card title="Passos de Treinamento">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Amostra</th>
                    <th>X1</th>
                    <th>X2</th>
                    <th>Target</th>
                    <th>W1</th>
                    <th>W2</th>
                    <th>Bias</th>
                  </tr>
                </thead>
                <tbody>
                  {current.steps.map(s => (
                    <tr key={s.amostra}>
                      <td className="td-cyan">A{s.amostra}</td>
                      <td>{s.x1}</td>
                      <td>{s.x2}</td>
                      <td className="td-white">{s.target}</td>
                      <td className="td-cyan">{s.w1.toFixed(1)}</td>
                      <td className="td-cyan">{s.w2.toFixed(1)}</td>
                      <td className="td-cyan">{s.bias.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Truth table results */}
            <Card title="Tabela Verdade — Resultado">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>X1</th>
                    <th>X2</th>
                    <th>Target</th>
                    <th>y_in</th>
                    <th>Pred</th>
                    <th>OK</th>
                  </tr>
                </thead>
                <tbody>
                  {current.testes.map((t, i) => (
                    <tr key={i}>
                      <td>{t.x1}</td>
                      <td>{t.x2}</td>
                      <td className="td-white">{t.target}</td>
                      <td className="td-cyan">{t.yIn.toFixed(2)}</td>
                      <td className={t.predicao >= 0 ? 'td-green' : 'td-pink'}>
                        {t.predicao}
                      </td>
                      <td className={t.acertou ? 'td-green' : 'td-pink'}>
                        {t.acertou ? '\u2713' : '\u2717'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
