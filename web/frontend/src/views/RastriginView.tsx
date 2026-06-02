import { useState, useRef, useMemo, useEffect, type ComponentType, type CSSProperties } from 'react';
import * as factoryNS from 'react-plotly.js/factory';
// @ts-expect-error — plotly.js-dist-min não tem types separados; o factory aceita.
import * as PlotlyNS from 'plotly.js-dist-min';

// Desembrulha CJS↔ESM caçando algo callable (a função factory).
function unwrapFactory(mod: unknown): (p: unknown) => ComponentType<{
  data: unknown; layout: unknown; config: unknown; style?: CSSProperties; useResizeHandler?: boolean;
}> {
  type Maybe = { default?: unknown; [k: string]: unknown };
  const m = mod as Maybe;
  if (typeof mod === 'function') return mod as never;
  if (typeof m.default === 'function') return m.default as never;
  if (m.default && typeof (m.default as Maybe).default === 'function') return (m.default as Maybe).default as never;
  // último recurso: log + erro claro
  // eslint-disable-next-line no-console
  console.error('react-plotly.js/factory shape inesperada:', mod);
  throw new Error('react-plotly.js factory não encontrado');
}
function unwrapPlotly(mod: unknown): unknown {
  type Maybe = { default?: unknown; newPlot?: unknown };
  const m = mod as Maybe;
  // o objeto Plotly tem newPlot. Procure o nível certo.
  if (m && typeof m.newPlot === 'function') return mod;
  if (m && m.default && typeof (m.default as Maybe).newPlot === 'function') return m.default;
  // eslint-disable-next-line no-console
  console.error('plotly.js-dist-min shape inesperada:', mod);
  return mod;
}

const createPlotlyComponent = unwrapFactory(factoryNS);
const Plotly = unwrapPlotly(PlotlyNS);
const Plot = createPlotlyComponent(Plotly);
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip,
} from 'recharts';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type {
  RastConfig, RastStep, RastResult,
  RastSelecao, RastCruzamento,
} from '../api/types';

const POP_OPTIONS = [
  { value: '20', label: '20' },
  { value: '40', label: '40' },
  { value: '60', label: '60' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
];
const GERACOES_OPTIONS = [
  { value: '50',  label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '500', label: '500' },
];
const PC_OPTIONS = [
  { value: '0.6',  label: 'Pc 0.60' },
  { value: '0.75', label: 'Pc 0.75' },
  { value: '0.85', label: 'Pc 0.85' },
  { value: '0.95', label: 'Pc 0.95' },
];
const PM_OPTIONS = [
  { value: '0.02', label: 'Pm 0.02' },
  { value: '0.05', label: 'Pm 0.05' },
  { value: '0.1',  label: 'Pm 0.10' },
  { value: '0.2',  label: 'Pm 0.20' },
];
const SELECAO_OPTIONS = [
  { value: 'torneio', label: 'Torneio' },
  { value: 'roleta',  label: 'Roleta' },
];
const TORNEIO_OPTIONS = [
  { value: '2', label: 'k = 2' },
  { value: '4', label: 'k = 4' },
  { value: '6', label: 'k = 6' },
];
const CRUZAMENTO_OPTIONS = [
  { value: 'radcliff', label: 'RADCLIFF (2 filhos, convexo)' },
  { value: 'wright',   label: 'WRIGHT (3 filhos, pega 2 melhores)' },
];
const ELITE_OPTIONS = [
  { value: '0', label: 'sem elite' },
  { value: '1', label: 'p = 1' },
  { value: '2', label: 'p = 2' },
  { value: '4', label: 'p = 4' },
];

export default function RastriginView() {
  const { show } = useToast();

  // Config
  const [popSize, setPopSize] = useState('60');
  const [maxGeracoes, setMaxGeracoes] = useState('200');
  const [probCruz, setProbCruz] = useState('0.85');
  const [probMut, setProbMut] = useState('0.05');
  const [selecao, setSelecao] = useState<RastSelecao>('torneio');
  const [tamTorneio, setTamTorneio] = useState('4');
  const [cruzamento, setCruzamento] = useState<RastCruzamento>('radcliff');
  const [elitismo, setElitismo] = useState('2');

  // Estado de treino
  const [training, setTraining] = useState(false);
  const [step, setStep] = useState<RastStep | null>(null);
  const [result, setResult] = useState<RastResult | null>(null);
  const [chartData, setChartData] = useState<{ gen: number; melhor: number; melhorAcum: number; media: number }[]>([]);

  const closeSSE = useRef<(() => void) | null>(null);

  useEffect(() => () => {
    if (closeSSE.current) closeSSE.current();
  }, []);

  const DOM_MIN = -5.12, DOM_MAX = 5.12;

  function limparEstado() {
    setStep(null);
    setResult(null);
    setChartData([]);
  }

  async function handleTrain() {
    setTraining(true);
    limparEstado();
    const cfg: RastConfig = {
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGeracoes),
      probCruzamento: parseFloat(probCruz),
      probMutacao: parseFloat(probMut),
      selecao,
      tamanhoTorneio: parseInt(tamTorneio),
      cruzamento,
      elitismo: parseInt(elitismo),
      dominioMin: DOM_MIN,
      dominioMax: DOM_MAX,
    };
    try {
      await apiPost('/agrastrigin/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      return;
    }
    let bestSoFar = Infinity;
    closeSSE.current = apiSSE('/agrastrigin/train', {
      onMessage(data) {
        const s = data as RastStep;
        setStep(s);
        if (s.melhorGlobalFx < bestSoFar) bestSoFar = s.melhorGlobalFx;
        setChartData(prev => [...prev, {
          gen: s.geracao,
          melhor: s.melhorFx,
          melhorAcum: s.melhorGlobalFx,
          media: s.mediaFx,
        }]);
      },
      onDone(data) {
        const r = data as RastResult;
        setResult(r);
        setTraining(false);
        closeSSE.current = null;
        const [x, y, z] = r.melhorX;
        show(`Melhor: f(${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}) = ${r.melhorFx.toFixed(4)}`);
      },
      onError() {
        setTraining(false);
        closeSSE.current = null;
        show('Erro no treino (stream)');
      },
    });
  }

  async function handleReset() {
    if (closeSSE.current) { closeSSE.current(); closeSSE.current = null; }
    try { await apiPost('/agrastrigin/reset'); } catch { /* ignore */ }
    limparEstado();
    setTraining(false);
    show('Rastrigin resetado');
  }

  // Dados pro Plotly 3D scatter
  const plotData = useMemo(() => {
    const pop = step?.populacao ?? [];
    const xs = pop.map(p => p.x[0]);
    const ys = pop.map(p => p.x[1]);
    const zs = pop.map(p => p.x[2]);
    const fxs = pop.map(p => p.fx);
    const traces: unknown[] = [];

    // 1) Pontos de referência: o ÓTIMO global em (0,0,0)
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      name: 'ótimo f(0,0,0)=0',
      x: [0], y: [0], z: [0],
      marker: { size: 11, color: '#ffff00', symbol: 'diamond',
        line: { color: '#fff', width: 2 } },
      hovertemplate: 'ÓTIMO (0,0,0)<br>f=0<extra></extra>',
    });

    // 2) População atual (cor = fitness)
    if (pop.length > 0) {
      traces.push({
        type: 'scatter3d',
        mode: 'markers',
        name: 'população',
        x: xs, y: ys, z: zs,
        marker: {
          size: 4,
          color: fxs,
          colorscale: 'Viridis',
          reversescale: true,
          cmin: 0,
          cmax: 80,
          colorbar: {
            title: 'f(x,y,z)',
            titleside: 'right',
            len: 0.7,
            x: 1.02,
            tickfont: { color: '#aaa', size: 9 },
            titlefont: { color: '#aaa', size: 10 },
          },
          opacity: 0.85,
        },
        text: pop.map((_, i) => `ind ${i + 1}`),
        hovertemplate: '%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>f=%{marker.color:.3f}<extra></extra>',
      });

      // 3) Melhor global em destaque (esfera grande rosa)
      const mg = step?.melhorGlobalX;
      if (mg) {
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          name: 'melhor global',
          x: [mg[0]], y: [mg[1]], z: [mg[2]],
          marker: { size: 9, color: '#ff00aa', symbol: 'circle',
            line: { color: '#fff', width: 1 } },
          hovertemplate: 'melhor global<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
        });
      }
    }
    return traces;
  }, [step]);

  const plotLayout = useMemo(() => ({
    autosize: true,
    height: 480,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: '#0a0a0a',
    plot_bgcolor: '#0a0a0a',
    scene: {
      xaxis: {
        title: { text: 'x', font: { color: '#aaa' } },
        range: [DOM_MIN, DOM_MAX],
        gridcolor: '#222',
        zerolinecolor: '#444',
        tickfont: { color: '#888', size: 10 },
      },
      yaxis: {
        title: { text: 'y', font: { color: '#aaa' } },
        range: [DOM_MIN, DOM_MAX],
        gridcolor: '#222',
        zerolinecolor: '#444',
        tickfont: { color: '#888', size: 10 },
      },
      zaxis: {
        title: { text: 'z', font: { color: '#aaa' } },
        range: [DOM_MIN, DOM_MAX],
        gridcolor: '#222',
        zerolinecolor: '#444',
        tickfont: { color: '#888', size: 10 },
      },
      bgcolor: '#0a0a0a',
      camera: { eye: { x: 1.6, y: 1.6, z: 1.2 } },
    },
    legend: { font: { color: '#aaa', size: 11 }, x: 0, y: 1 },
    showlegend: true,
  }), []);

  const isTorneio = selecao === 'torneio';
  const melhorAtual = step?.melhorGlobalFx;
  const distOtimo = step?.melhorGlobalX
    ? Math.sqrt(step.melhorGlobalX.reduce((a, b) => a + b * b, 0))
    : null;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">AG <span>Rastrigin 3D</span></div>
          <div className="page-sub">
            AG com cromossomos REAIS — Trabalho 13 · Aula 15 · minimiza f(x,y,z) = Σ(xᵢ² − 10·cos(2π·xᵢ)) + 30
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" onClick={handleReset} style={{ fontSize: 11, padding: '6px 12px' }}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            EVOLUIR
          </button>
        </div>
      </div>

      {/* Config linha 1 */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="População" options={POP_OPTIONS} value={popSize} onChange={setPopSize} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Gerações" options={GERACOES_OPTIONS} value={maxGeracoes} onChange={setMaxGeracoes} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">
            Pc · Pm <span style={{ color: 'var(--muted)', fontWeight: 400 }}>(cruzamento · mutação)</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={PC_OPTIONS} value={probCruz} onChange={setProbCruz} style={{ flex: 1 }} />
            <Select options={PM_OPTIONS} value={probMut} onChange={setProbMut} style={{ flex: 1 }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Seleção" options={SELECAO_OPTIONS} value={selecao} onChange={(v) => setSelecao(v as RastSelecao)} style={{ width: '100%' }} />
          <div style={{ marginTop: 10, opacity: isTorneio ? 1 : 0.4, pointerEvents: isTorneio ? 'auto' : 'none' }}>
            <Select label="Tamanho do torneio" options={TORNEIO_OPTIONS} value={tamTorneio} onChange={setTamTorneio} style={{ width: '100%' }} />
          </div>
        </Card>
      </div>

      {/* Config linha 2 */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Cruzamento (real)" options={CRUZAMENTO_OPTIONS} value={cruzamento} onChange={(v) => setCruzamento(v as RastCruzamento)} style={{ width: '100%' }} />
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Elitismo" options={ELITE_OPTIONS} value={elitismo} onChange={setElitismo} style={{ width: '100%' }} />
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Domínio</div>
          <div style={{ fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)', lineHeight: 1.7 }}>
            x, y, z ∈ <span style={{ color: 'var(--cyan)' }}>[{DOM_MIN}, {DOM_MAX}]</span>
            <br />
            <span style={{ color: '#888' }}>(domínio padrão do Rastrigin)</span>
          </div>
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Geração"
          value={step ? step.geracao.toLocaleString() : '—'}
          label={`de ${parseInt(maxGeracoes).toLocaleString()}`}
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Melhor f(x,y,z)"
          value={melhorAtual !== undefined ? melhorAtual.toFixed(4) : '—'}
          label={melhorAtual !== undefined && melhorAtual < 0.01 ? '≈ ótimo global (0)' : 'menor = melhor (alvo: 0)'}
          color="cyan"
        />
        <MetricCard
          title="Distância ao ótimo"
          value={distOtimo !== null ? `‖x‖ = ${distOtimo.toFixed(3)}` : '—'}
          label="‖(x,y,z) − (0,0,0)‖ (≤ √3·5.12 ≈ 8.87)"
        />
      </div>

      {/* Scatter 3D — coração do trabalho */}
      <Card title="Espaço de busca 3D — população convergindo pro mínimo global" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6,
            fontFamily: 'JetBrains Mono',
          }}>
            Cada ponto é um <b>indivíduo</b> (cromossomo real x,y,z). Cor = fitness (azul/escuro = melhor).{' '}
            <b style={{ color: '#ffff00' }}>★ amarelo</b> = ótimo global f(0,0,0)=0.{' '}
            <b style={{ color: '#ff00aa' }}>● rosa</b> = melhor global encontrado. Arraste pra girar, scroll pra zoom.
          </div>
          <Plot
            data={plotData as unknown as Plotly.Data[]}
            layout={plotLayout as unknown as Partial<Plotly.Layout>}
            config={{ displaylogo: false, responsive: true, displayModeBar: false } as unknown as Partial<Plotly.Config>}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      </Card>

      {/* Convergência */}
      {chartData.length > 0 && (
        <Card title="Convergência — f(x,y,z) por geração (escala log; alvo = 0)" style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="gen" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <YAxis
                  stroke="#555"
                  tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  tickLine={false}
                  scale="log"
                  domain={[0.001, 'auto']}
                  allowDataOverflow
                />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  labelFormatter={(v) => `geração ${v}`}
                  formatter={(v) => Number(v).toFixed(4)}
                />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Line name="média da pop" type="monotone" dataKey="media" stroke="#00ccff" strokeWidth={1} strokeOpacity={0.6} strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                <Line name="melhor da geração" type="monotone" dataKey="melhor" stroke="#ff00aa" strokeWidth={1} strokeOpacity={0.5} dot={false} isAnimationActive={false} />
                <Line name="melhor acumulado" type="monotone" dataKey="melhorAcum" stroke="#ffff00" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Posição do melhor */}
      {result && (
        <Card title="Melhor cromossomo encontrado" style={{ marginBottom: 16 }}>
          <div style={{ padding: 16, fontFamily: 'JetBrains Mono', fontSize: 14, color: 'var(--muted)', lineHeight: 1.9 }}>
            <div>x = <span style={{ color: 'var(--cyan)' }}>{result.melhorX[0].toFixed(6)}</span></div>
            <div>y = <span style={{ color: 'var(--cyan)' }}>{result.melhorX[1].toFixed(6)}</span></div>
            <div>z = <span style={{ color: 'var(--cyan)' }}>{result.melhorX[2].toFixed(6)}</span></div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid #222' }}>
              f(x, y, z) = <b style={{ color: '#ffff00' }}>{result.melhorFx.toFixed(8)}</b>
              <span style={{ color: '#888', marginLeft: 8 }}>
                (ótimo teórico: <b style={{ color: '#ffff00' }}>0</b>)
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Educacional */}
      <Card title="Como funciona o AG com cromossomos reais">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Cromossomo real (Aula 15):</b> em vez de cadeia de bits, cada gene é um <b>número real</b> diretamente
          — aqui um vetor [x, y, z]. Inicialização: <code>x = a + c·(b − a)</code> com c ∈ [0,1].
          <br /><br />
          <b>Cruzamento RADCLIFF</b> (combinação convexa, 2 filhos):
          <ul style={{ marginLeft: 18 }}>
            <li><code>xa(novo) = β·xa + (1 − β)·xb</code></li>
            <li><code>xb(novo) = (1 − β)·xa + β·xb</code> &nbsp; com β ∈ (0, 1) aleatório</li>
          </ul>
          Filhos sempre <b>dentro</b> do segmento entre os pais — nunca extrapola, então sempre válidos.
          <br /><br />
          <b>Cruzamento WRIGHT</b> (gera 3, fica com os 2 melhores):
          <ul style={{ marginLeft: 18 }}>
            <li><code>xa(novo) = 0,5·xa + 0,5·xb</code></li>
            <li><code>xb(novo) = 1,5·xa − 0,5·xb</code></li>
            <li><code>xc(novo) = −0,5·xa + 1,5·xb</code></li>
          </ul>
          As fórmulas 2 e 3 <b>extrapolam</b> — podem cair fora do domínio. Selecionamos
          os 2 válidos com menor f; se &lt; 2 forem válidos, completamos com clampados.
          <br /><br />
          <b>Mutação:</b> escolhe um gene aleatório e substitui por <code>a + c·(b − a)</code> — mesmo
          esquema da inicialização.
          <br /><br />
          <b>Função Rastrigin 3D:</b> f(x,y,z) = x² + y² + z² − 10·cos(2πx) − 10·cos(2πy) − 10·cos(2πz) + 30.
          Tem um único <b>mínimo global em (0, 0, 0)</b> com f = 0, mas <b>centenas de mínimos locais</b>
          em pontos como (1, 0, 0), (1, 1, 0), (2, 1, −1)… — exemplo clássico de paisagem multimodal
          que separa AG bom de AG mediano.
        </div>
      </Card>
    </div>
  );
}
