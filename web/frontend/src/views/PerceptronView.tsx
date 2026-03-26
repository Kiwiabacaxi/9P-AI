import { useState, useEffect, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import PixelGrid from '../components/shared/PixelGrid';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost } from '../api/client';

// ---------------------------------------------------------------------------
// Types — Portas Logicas
// ---------------------------------------------------------------------------

const PORTAS = ['AND', 'OR', 'NAND', 'NOR', 'XOR'] as const;

interface PercPortasStep {
  ciclo: number;
  amostra: number;
  x1: number;
  x2: number;
  target: number;
  yLiq: number;
  y: number;
  teveErro: boolean;
  w1: number;
  w2: number;
  bias: number;
}

interface PercPortasTest {
  x1: number;
  x2: number;
  target: number;
  predicao: number;
  yLiq: number;
  acertou: boolean;
}

interface PercPortasResult {
  porta: string;
  convergiu: boolean;
  ciclos: number;
  w1: number;
  w2: number;
  bias: number;
  acertos: number;
  acuracia: number;
  steps: PercPortasStep[];
  testes: PercPortasTest[];
}

// ---------------------------------------------------------------------------
// Types — Letras
// ---------------------------------------------------------------------------

interface PercLetrasStep {
  ciclo: number;
  amostra: string;
  target: number;
  yLiq: number;
  y: number;
  delta: number;
  novoBias: number;
  teveErro: boolean;
}

interface PercLetrasTest {
  letra: string;
  target: number;
  predicao: number;
  yLiq: number;
  acertou: boolean;
}

interface PercLetrasResult {
  convergiu: boolean;
  ciclos: number;
  bias: number;
  acertos: number;
  acuracia: number;
  steps: PercLetrasStep[];
  testes: PercLetrasTest[];
}

interface DatasetEntry {
  letra: string;
  grade: number[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

type Tab = 'portas' | 'letras';

export default function PerceptronView() {
  const [tab, setTab] = useState<Tab>('portas');

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Perceptron <span>Portas + Letras</span></div>
          <div className="page-sub">atualiza so no erro &middot; convergencia garantida (sep. linear) &middot; Trab 02</div>
        </div>
      </div>

      <div className="tabs">
        <div
          className={`tab${tab === 'portas' ? ' active' : ''}`}
          onClick={() => setTab('portas')}
        >
          Portas Logicas
        </div>
        <div
          className={`tab${tab === 'letras' ? ' active' : ''}`}
          onClick={() => setTab('letras')}
        >
          Letras
        </div>
      </div>

      {tab === 'portas' ? <PortasTab /> : <LetrasTab />}
    </div>
  );
}

// ===========================================================================
// Portas Logicas Tab
// ===========================================================================

function PortasTab() {
  const { show } = useToast();
  const [results, setResults] = useState<Record<string, PercPortasResult>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const current = selected ? results[selected] : null;

  async function trainGate(porta: string) {
    try {
      const res = await apiPost<PercPortasResult>(`/perceptron-portas/train?porta=${porta}`);
      setResults(prev => ({ ...prev, [porta]: res }));
      setSelected(porta);
      show(`Perceptron ${porta} treinado`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  async function trainAll() {
    setLoading(true);
    try {
      const res = await apiPost<Record<string, PercPortasResult>>('/perceptron-portas/train?porta=ALL');
      setResults(prev => ({ ...prev, ...res }));
      const first = Object.keys(res)[0];
      if (first && !selected) setSelected(first);
      show('Todas as portas treinadas');
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
    setLoading(false);
  }

  return (
    <>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button className="btn btn-ghost" onClick={trainAll} disabled={loading}>
          {loading && <span className="spin" />}
          TREINAR TODAS
        </button>
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
          ].filter(Boolean).join(' ');
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
                  <div className="weight-val">{current.w1.toFixed(4)}</div>
                  <div className="weight-lbl">W1</div>
                </div>
                <div className="weight-box">
                  <div className="weight-val">{current.w2.toFixed(4)}</div>
                  <div className="weight-lbl">W2</div>
                </div>
                <div className="weight-box">
                  <div className="weight-val">{current.bias.toFixed(4)}</div>
                  <div className="weight-lbl">BIAS</div>
                </div>
              </div>
            </Card>

            <MetricCard
              title="Acuracia"
              value={current.acuracia.toFixed(0) + '%'}
              label={`${current.acertos}/4 acertos`}
              color={current.acuracia === 100 ? 'green' : 'pink'}
            />

            <MetricCard
              title="Convergencia"
              value={current.convergiu ? 'SIM' : 'NAO'}
              label={`${current.ciclos} ciclo${current.ciclos !== 1 ? 's' : ''}`}
              color={current.convergiu ? 'green' : 'pink'}
              valueStyle={{ fontSize: 24 }}
            />
          </div>

          <div className="grid-2">
            {/* Training steps table */}
            <Card title="Passos de Treinamento">
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Ciclo</th>
                      <th>Am</th>
                      <th>X1</th>
                      <th>X2</th>
                      <th>Target</th>
                      <th>y_liq</th>
                      <th>y</th>
                      <th>Erro</th>
                      <th>W1</th>
                      <th>W2</th>
                      <th>Bias</th>
                    </tr>
                  </thead>
                  <tbody>
                    {current.steps.map((s, i) => (
                      <tr key={i}>
                        <td className="td-cyan">{s.ciclo}</td>
                        <td>A{s.amostra}</td>
                        <td>{s.x1}</td>
                        <td>{s.x2}</td>
                        <td className="td-white">{s.target}</td>
                        <td className="td-cyan">{s.yLiq.toFixed(4)}</td>
                        <td className={s.y === s.target ? 'td-green' : 'td-pink'}>{s.y}</td>
                        <td className={s.teveErro ? 'td-pink' : 'td-green'}>
                          {s.teveErro ? '\u2717' : '\u2713'}
                        </td>
                        <td className="td-cyan">{s.w1.toFixed(4)}</td>
                        <td className="td-cyan">{s.w2.toFixed(4)}</td>
                        <td className="td-cyan">{s.bias.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Truth table */}
            <Card title="Tabela Verdade — Resultado">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>X1</th>
                    <th>X2</th>
                    <th>Target</th>
                    <th>y_liq</th>
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
                      <td className="td-cyan">{t.yLiq.toFixed(4)}</td>
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
    </>
  );
}

// ===========================================================================
// Letras Tab
// ===========================================================================

function LetrasTab() {
  const { show } = useToast();
  const [result, setResult] = useState<PercLetrasResult | null>(null);
  const [dataset, setDataset] = useState<DatasetEntry[]>([]);
  const [loading, setLoading] = useState(false);

  // Interactive testing
  const [testPixels, setTestPixels] = useState<number[]>(new Array(49).fill(-1));
  const [previewLetter, setPreviewLetter] = useState<string | null>(null);

  // Fetch dataset on mount
  useEffect(() => {
    apiGet<DatasetEntry[]>('/perceptron-letras/dataset').then(setDataset).catch(() => {});
  }, []);

  const handleTrain = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiPost<PercLetrasResult>('/perceptron-letras/train');
      setResult(res);
      show(res.convergiu ? 'Convergiu!' : 'Limite de ciclos atingido');
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
    setLoading(false);
  }, [show]);

  const handleLoadLetter = useCallback((entry: DatasetEntry) => {
    setPreviewLetter(entry.letra);
    setTestPixels(entry.grade.map(v => (v >= 0 ? 1 : -1)));
  }, []);

  return (
    <>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button className="btn btn-primary" onClick={handleTrain} disabled={loading}>
          {loading && <span className="spin" />}
          TREINAR
        </button>
      </div>

      {/* Metrics */}
      {result && (
        <div className="grid-3" style={{ marginBottom: 16 }}>
          <MetricCard
            title="Acuracia"
            value={result.acuracia.toFixed(0) + '%'}
            label={`${result.acertos}/2 letras`}
            color={result.acuracia === 100 ? 'green' : 'pink'}
          />
          <MetricCard
            title="Ciclos"
            value={result.ciclos.toLocaleString()}
            label={result.convergiu ? 'convergiu' : 'limite atingido'}
            color="cyan"
          />
          <MetricCard
            title="Bias Final"
            value={result.bias.toFixed(4)}
            label="valor do bias apos treinamento"
            valueStyle={{ fontSize: 22 }}
          />
        </div>
      )}

      <div className="grid-2">
        {/* Training steps */}
        {result && (
          <Card title="Passos de Treinamento">
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ciclo</th>
                    <th>Amostra</th>
                    <th>Target</th>
                    <th>y_liq</th>
                    <th>y</th>
                    <th>Delta</th>
                    <th>Bias</th>
                    <th>Erro</th>
                  </tr>
                </thead>
                <tbody>
                  {result.steps.map((s, i) => (
                    <tr key={i}>
                      <td className="td-cyan">{s.ciclo}</td>
                      <td className="td-white">{s.amostra}</td>
                      <td>{s.target}</td>
                      <td className="td-cyan">{s.yLiq.toFixed(4)}</td>
                      <td className={s.y === s.target ? 'td-green' : 'td-pink'}>{s.y}</td>
                      <td className={s.delta !== 0 ? 'td-pink' : 'td-green'}>{s.delta.toFixed(4)}</td>
                      <td className="td-cyan">{s.novoBias.toFixed(4)}</td>
                      <td className={s.teveErro ? 'td-pink' : 'td-green'}>
                        {s.teveErro ? '\u2717' : '\u2713'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* Test results */}
        {result && (
          <Card title="Resultado por Letra">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Letra</th>
                  <th>Target</th>
                  <th>y_liq</th>
                  <th>Pred</th>
                  <th>OK</th>
                </tr>
              </thead>
              <tbody>
                {result.testes.map((t, i) => (
                  <tr key={i}>
                    <td className="td-white">{t.letra}</td>
                    <td>{t.target}</td>
                    <td className="td-cyan">{t.yLiq.toFixed(4)}</td>
                    <td className={t.acertou ? 'td-green' : 'td-pink'}>{t.predicao}</td>
                    <td className={t.acertou ? 'td-green' : 'td-pink'}>
                      {t.acertou ? '\u2713' : '\u2717'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>

      {/* Dataset preview and interactive grid */}
      <Card title="Dataset — Grades 7x7" style={{ marginTop: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'start' }}>
          <div>
            {/* Letter buttons to load into grid */}
            <div style={{ marginBottom: 12 }}>
              <div className="porta-chips">
                {dataset.map(entry => (
                  <button
                    key={entry.letra}
                    className={`porta-chip${previewLetter === entry.letra ? ' selected' : ''}`}
                    onClick={() => handleLoadLetter(entry)}
                  >
                    {entry.letra}
                  </button>
                ))}
              </div>
            </div>
            <PixelGrid
              rows={7}
              cols={7}
              cellSize={28}
              values={testPixels}
              onChange={(vals) => { setTestPixels(vals); setPreviewLetter(null); }}
            />
          </div>

          <div>
            {/* Mini preview grids for each letter in dataset */}
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {dataset.map(entry => (
                <div key={entry.letra} style={{ textAlign: 'center' }}>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(7, 8px)',
                    gap: 1,
                    marginBottom: 4,
                  }}>
                    {entry.grade.slice(0, 49).map((v, i) => (
                      <div
                        key={i}
                        style={{
                          width: 8,
                          height: 8,
                          background: v >= 1 ? 'var(--pink)' : 'var(--surface-low)',
                        }}
                      />
                    ))}
                  </div>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    fontWeight: 700,
                    color: 'var(--secondary)',
                  }}>{entry.letra}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </>
  );
}
