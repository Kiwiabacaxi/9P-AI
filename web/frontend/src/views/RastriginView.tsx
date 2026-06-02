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

const NX = 100; // resolução da malha da superfície (eixos x e y)
const CMAX_F = 50; // teto do colorscale (aptidão): perto disso já é "ruim"

// Termo 1D do Rastrigin: t(v) = v² − 10·cos(2π·v).
function termR(v: number): number {
  return v * v - 10 * Math.cos(2 * Math.PI * v);
}
// Rastrigin 3D completa.
function rastrigin3(x: number, y: number, z: number): number {
  return 30 + termR(x) + termR(y) + termR(z);
}
// Mínimo local 1D real perto do inteiro n — os mínimos do Rastrigin NÃO caem
// exatamente nos inteiros: o envelope v² puxa cada um levemente rumo à origem.
// Newton em t'(v) = 2v + 20π·sin(2πv).
function refineMin1D(n: number): number {
  let v = n;
  for (let it = 0; it < 8; it++) {
    const g = 2 * v + 20 * Math.PI * Math.sin(2 * Math.PI * v);
    const gp = 2 + 40 * Math.PI * Math.PI * Math.cos(2 * Math.PI * v);
    if (gp === 0) break;
    v -= g / gp;
  }
  return v;
}

// Camadas (só no modo Superfície): cada valor de z vira uma fatia z = c.
const SLICE_VALUES = [-2, -1, 0, 1, 2];

type Modo = 'superficie' | 'espaco';

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
  // Domínio em estado-string (permite editar/limpar sem virar NaN no controle).
  const [domMinStr, setDomMinStr] = useState('-5.12');
  const [domMaxStr, setDomMaxStr] = useState('5.12');
  const domMin = parseFloat(domMinStr);
  const domMax = parseFloat(domMaxStr);

  // Estado de treino
  const [training, setTraining] = useState(false);
  const [step, setStep] = useState<RastStep | null>(null);
  const [result, setResult] = useState<RastResult | null>(null);
  const [chartData, setChartData] = useState<{ gen: number; melhor: number; melhorAcum: number; media: number }[]>([]);

  // Controles do 3D
  const [modo, setModo] = useState<Modo>('superficie');
  const [slices, setSlices] = useState<Record<number, boolean>>({ [-2]: false, [-1]: false, 0: true, 1: false, 2: false });
  const [opacidade, setOpacidade] = useState(0.85);
  const [mostrarPop, setMostrarPop] = useState(true);
  const [mostrarMinimos, setMostrarMinimos] = useState(true);
  const [mostrarOtimo, setMostrarOtimo] = useState(true);

  const closeSSE = useRef<(() => void) | null>(null);

  useEffect(() => () => {
    if (closeSSE.current) closeSSE.current();
  }, []);

  function limparEstado() {
    setStep(null);
    setResult(null);
    setChartData([]);
  }

  function toggleSlice(c: number) {
    setSlices(prev => ({ ...prev, [c]: !prev[c] }));
  }

  async function handleTrain() {
    if (!(Number.isFinite(domMin) && Number.isFinite(domMax) && domMin < domMax)) {
      show('Domínio inválido: preencha min < max');
      return;
    }
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
      dominioMin: domMin,
      dominioMax: domMax,
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

  // Domínio efetivo da viz = o domínio do treino corrente (durante) ou o do
  // resultado (depois). Fallback pro padrão se o campo estiver vazio/inválido.
  const vizMin = result ? result.cfg.dominioMin : (Number.isFinite(domMin) ? domMin : -5.12);
  const vizMax = result ? result.cfg.dominioMax : (Number.isFinite(domMax) ? domMax : 5.12);

  // Malha 2D base da superfície: base[j][i] = t(x_i)+t(y_j). Memo em [viz].
  const surf = useMemo(() => {
    const axis: number[] = [];
    for (let i = 0; i < NX; i++) axis.push(vizMin + (i * (vizMax - vizMin)) / (NX - 1));
    const t = axis.map(termR);
    const base: number[][] = [];
    let baseMax = 0;
    for (let j = 0; j < NX; j++) {
      const row: number[] = [];
      for (let i = 0; i < NX; i++) {
        const v = t[j] + t[i];
        row.push(v);
        if (v > baseMax) baseMax = v;
      }
      base.push(row);
    }
    const maxStep = Math.max(...SLICE_VALUES.map(termR));
    const zMax = Math.ceil((baseMax + 30 + maxStep) / 10) * 10;
    return { axis, base, zMax };
  }, [vizMin, vizMax]);

  // Z de cada fatia (z=c) pré-computado — independe de opacidade/população, então
  // animar ao vivo ou arrastar o slider NÃO reconstrói a malha 100×100.
  const sliceGrids = useMemo(() => {
    const m: Record<number, number[][]> = {};
    for (const c of SLICE_VALUES) {
      const stepC = 30 + termR(c);
      m[c] = surf.base.map(row => row.map(v => v + stepC));
    }
    return m;
  }, [surf]);

  // Rede de mínimos locais (modo cubo): mínimos REAIS perto de cada inteiro do
  // domínio (limitado a |coord| ≤ 6), cor = f. São as "armadilhas" do AG.
  const lattice = useMemo(() => {
    const lo = Math.ceil(Math.max(vizMin, -6));
    const hi = Math.floor(Math.min(vizMax, 6));
    const refined: Record<number, number> = {};
    for (let n = lo; n <= hi; n++) refined[n] = refineMin1D(n);
    const xs: number[] = [], ys: number[] = [], zs: number[] = [], fs: number[] = [];
    for (let i = lo; i <= hi; i++)
      for (let j = lo; j <= hi; j++)
        for (let k = lo; k <= hi; k++) {
          const rx = refined[i], ry = refined[j], rz = refined[k];
          xs.push(rx); ys.push(ry); zs.push(rz); fs.push(rastrigin3(rx, ry, rz));
        }
    return { xs, ys, zs, fs };
  }, [vizMin, vizMax]);

  // Traces do Plotly 3D — montados a partir de `step` (ao vivo durante o treino).
  const plotData = useMemo(() => {
    if (!step) return [];
    const traces: unknown[] = [];
    const pop = step.populacao ?? [];
    const best = step.melhorGlobalX;
    const bestFx = step.melhorGlobalFx;
    const colorbar = {
      title: { text: 'f(x,y,z)', side: 'right', font: { color: '#aaa', size: 10 } },
      len: 0.7, x: 1.02, tickfont: { color: '#aaa', size: 9 },
    };
    const otimoVisivel = mostrarOtimo && vizMin <= 0 && vizMax >= 0;

    const traceMelhor = (z: number) => ({
      type: 'scatter3d', mode: 'markers', name: 'melhor (até agora)',
      x: best ? [best[0]] : [], y: best ? [best[1]] : [], z: best ? [z] : [],
      marker: { size: 12, color: '#ff2d9b', symbol: 'circle-open', line: { color: '#ff2d9b', width: 3 } },
      hovertemplate: `melhor<br>f=${(bestFx ?? 0).toFixed(4)}<br>x=%{x:.3f}  y=%{y:.3f}<extra></extra>`,
    });
    const traceOtimo = {
      type: 'scatter3d', mode: 'markers', name: 'mín. teórico (0,0,0)',
      x: [0], y: [0], z: [0],
      marker: { size: 11, color: '#00e5ff', symbol: 'diamond-open', line: { color: '#00e5ff', width: 3 } },
      hovertemplate: 'mínimo teórico<br>f(0,0,0)=0<extra></extra>',
    };

    if (modo === 'espaco') {
      // === CUBO DE BUSCA (x, y, z) — os filhos na posição real ===
      // Ordem: referência embaixo → marcadores vazados → FILHOS por cima.
      if (mostrarMinimos && lattice.xs.length > 0) {
        traces.push({
          type: 'scatter3d', mode: 'markers', name: 'mínimos locais',
          x: lattice.xs, y: lattice.ys, z: lattice.zs,
          marker: {
            size: 3, color: lattice.fs, colorscale: 'Jet', cmin: 0, cmax: CMAX_F,
            opacity: opacidade, showscale: true, colorbar, symbol: 'circle',
          },
          hovertemplate: 'mínimo local<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<br>f=%{marker.color:.3f}<extra></extra>',
        });
      }
      if (best) traces.push(traceMelhor(best[2]));
      if (otimoVisivel) traces.push(traceOtimo);
      if (mostrarPop && pop.length > 0) {
        traces.push({
          type: 'scatter3d', mode: 'markers', name: 'filhos (população)',
          x: pop.map(p => p.x[0]), y: pop.map(p => p.x[1]), z: pop.map(p => p.x[2]),
          marker: {
            size: 5, color: pop.map(p => p.fx), colorscale: 'Jet', cmin: 0, cmax: CMAX_F,
            opacity: 0.98, showscale: !mostrarMinimos, colorbar,
            line: { color: '#fff', width: 0.5 },
          },
          text: pop.map((_, i) => `filho ${i + 1}`),
          hovertemplate: '%{text}<br>x=%{x:.3f}  y=%{y:.3f}  z=%{z:.3f}<br>f=%{marker.color:.4f}<extra></extra>',
        });
      }
    } else {
      // === SUPERFÍCIE f(x, y) — a "caixa de ovos" por fatias em z ===
      const ativos = SLICE_VALUES.filter(c => slices[c]);
      ativos.forEach((c, idx) => {
        traces.push({
          type: 'surface', x: surf.axis, y: surf.axis, z: sliceGrids[c],
          colorscale: 'Jet', cmin: 0, cmax: surf.zMax, opacity: opacidade,
          showscale: idx === 0, colorbar,
          contours: { z: { show: true, color: 'rgba(0,0,0,0.22)', width: 1, start: 0, end: surf.zMax, size: 8 } },
          name: `z = ${c}`,
          hovertemplate: `x=%{x:.2f}  y=%{y:.2f}<br>f=%{z:.2f}<extra>fatia z=${c}</extra>`,
        });
      });
      if (best) traces.push(traceMelhor(bestFx ?? 0));
      if (otimoVisivel) traces.push(traceOtimo);
      // Filhos projetados em (x, y, f) — por cima, pra ver onde caíram.
      if (mostrarPop && pop.length > 0) {
        traces.push({
          type: 'scatter3d', mode: 'markers', name: 'filhos (em x,y,f)',
          x: pop.map(p => p.x[0]), y: pop.map(p => p.x[1]), z: pop.map(p => p.fx),
          marker: { size: 4, color: '#ffffff', opacity: 0.98, line: { color: '#000', width: 1 } },
          text: pop.map((_, i) => `filho ${i + 1}`),
          hovertemplate: '%{text}<br>x=%{x:.3f}  y=%{y:.3f}<br>f=%{z:.3f}<extra></extra>',
        });
      }
    }

    return traces;
  }, [step, modo, surf, sliceGrids, lattice, slices, opacidade, mostrarPop, mostrarMinimos, mostrarOtimo, vizMin, vizMax]);

  const plotLayout = useMemo(() => {
    const espaco = modo === 'espaco';
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 10, b: 0 },
      paper_bgcolor: '#0a0a0a',
      plot_bgcolor: '#0a0a0a',
      uirevision: 'rast3d', // preserva rotação/zoom do usuário entre gerações
      scene: {
        xaxis: { title: { text: 'x', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        yaxis: { title: { text: 'y', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        zaxis: espaco
          ? { title: { text: 'z', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } }
          : { title: { text: 'f', font: { color: '#aaa' } }, range: [0, surf.zMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        bgcolor: '#0a0a0a',
        aspectmode: 'manual',
        aspectratio: espaco ? { x: 1, y: 1, z: 1 } : { x: 1, y: 1, z: 0.6 },
        camera: { eye: espaco ? { x: 1.6, y: 1.6, z: 1.3 } : { x: 1.7, y: 1.7, z: 0.9 } },
      },
      legend: { font: { color: '#aaa', size: 11 }, x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' },
      showlegend: true,
    };
  }, [modo, vizMin, vizMax, surf.zMax]);

  const isTorneio = selecao === 'torneio';
  const melhorAtual = step?.melhorGlobalFx;
  const distOtimo = step?.melhorGlobalX
    ? Math.sqrt(step.melhorGlobalX.reduce((a, b) => a + b * b, 0))
    : null;

  const opacLabel = modo === 'espaco' ? 'opacidade (mínimos)' : 'opacidade (superfície)';

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
          <div className="imgreg-select-label">Domínio <span style={{ color: 'var(--muted)', fontWeight: 400 }}>(x, y, z)</span></div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontFamily: 'JetBrains Mono', fontSize: 13, color: 'var(--muted)' }}>
            <span>[</span>
            <input
              type="number" step={0.5} value={domMinStr}
              onChange={e => setDomMinStr(e.target.value)}
              style={{ width: 70, background: 'var(--surface-2)', border: '1px solid #333', borderRadius: 4, color: 'var(--cyan)', padding: '6px 8px', fontFamily: 'JetBrains Mono', fontSize: 13 }}
            />
            <span>,</span>
            <input
              type="number" step={0.5} value={domMaxStr}
              onChange={e => setDomMaxStr(e.target.value)}
              style={{ width: 70, background: 'var(--surface-2)', border: '1px solid #333', borderRadius: 4, color: 'var(--cyan)', padding: '6px 8px', fontFamily: 'JetBrains Mono', fontSize: 13 }}
            />
            <span>]</span>
          </div>
          <div style={{ fontSize: 11, color: '#777', marginTop: 6 }}>padrão do Rastrigin: [-5.12, 5.12]</div>
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
          label="‖(x,y,z) − (0,0,0)‖"
        />
      </div>

      {/* Visualização 3D — AO VIVO durante a evolução (aparece já na 1ª geração) */}
      {step && (
        <Card title={`Visualização 3D — população${training ? ' (ao vivo)' : ''} e função Rastrigin`} style={{ marginBottom: 16 }}>
          <div style={{ padding: 8 }}>
            {/* Toggle de modo */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
              <button className="btn" onClick={() => setModo('superficie')} aria-pressed={modo === 'superficie'}
                style={{ fontSize: 12, padding: '7px 14px', borderColor: modo === 'superficie' ? 'var(--cyan)' : '#333', opacity: modo === 'superficie' ? 1 : 0.55 }}>
                Superfície f(x, y)
              </button>
              <button className="btn" onClick={() => setModo('espaco')} aria-pressed={modo === 'espaco'}
                style={{ fontSize: 12, padding: '7px 14px', borderColor: modo === 'espaco' ? 'var(--cyan)' : '#333', opacity: modo === 'espaco' ? 1 : 0.55 }}>
                Cubo de busca (x, y, z)
              </button>
            </div>

            {/* Controles por modo */}
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
              marginBottom: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 6,
            }}>
              {modo === 'espaco' ? (
                <>
                  <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>MOSTRAR:</span>
                  <button className="btn" onClick={() => setMostrarPop(v => !v)} aria-pressed={mostrarPop}
                    style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarPop ? 1 : 0.4 }}>filhos (população)</button>
                  <button className="btn" onClick={() => setMostrarMinimos(v => !v)} aria-pressed={mostrarMinimos}
                    style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarMinimos ? 1 : 0.4 }}>mínimos locais</button>
                  <button className="btn" onClick={() => setMostrarOtimo(v => !v)} aria-pressed={mostrarOtimo}
                    style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarOtimo ? 1 : 0.4 }}>mín. teórico</button>
                </>
              ) : (
                <>
                  <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>CAMADAS (fatias z):</span>
                  {SLICE_VALUES.map(c => (
                    <button key={c} className="btn" onClick={() => toggleSlice(c)} aria-pressed={slices[c]}
                      style={{ fontSize: 11, padding: '5px 12px', opacity: slices[c] ? 1 : 0.4, borderColor: slices[c] ? 'var(--cyan)' : '#333' }}>
                      z = {c}
                    </button>
                  ))}
                  <div style={{ width: 1, height: 22, background: '#333', margin: '0 4px' }} />
                  <button className="btn" onClick={() => setMostrarPop(v => !v)} aria-pressed={mostrarPop}
                    style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarPop ? 1 : 0.4 }}>filhos</button>
                  <button className="btn" onClick={() => setMostrarOtimo(v => !v)} aria-pressed={mostrarOtimo}
                    style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarOtimo ? 1 : 0.4 }}>mín. teórico</button>
                </>
              )}
            </div>

            {/* Opacidade */}
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center',
              marginBottom: 10, padding: '8px 12px', fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)',
            }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {opacLabel}
                <input
                  type="range" min={0} max={1} step={0.02}
                  value={opacidade}
                  onChange={e => setOpacidade(parseFloat(e.target.value))}
                  style={{ width: 160 }}
                />
                <span style={{ color: 'var(--cyan)' }}>{opacidade.toFixed(2)}</span>
              </label>
            </div>

            {/* Legenda textual por modo */}
            <div style={{
              fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
              padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: 'JetBrains Mono',
            }}>
              {modo === 'espaco' ? (
                <>
                  Cubo do <b>espaço de busca (x, y, z)</b> — cada <b>filho</b> é um ponto na <b>posição real</b>,
                  cor = aptidão (<span style={{ color: '#2222ff' }}>azul</span> = ótimo · <span style={{ color: '#ff2200' }}>vermelho</span> = ruim).
                  Os <b>mínimos locais</b> (rede de pontos) são as armadilhas onde o AG pode emperrar.{' '}
                  <b style={{ color: '#ff2d9b' }}>○ rosa</b> = melhor · <b style={{ color: '#00e5ff' }}>◇ ciano</b> = mínimo teórico (0,0,0).
                </>
              ) : (
                <>
                  Superfície <b>f(x, y)</b> num plano <b>z = c</b> fixo — a clássica "caixa de ovos" com dezenas de
                  mínimos locais e o global no centro. Os <b style={{ color: '#fff' }}>filhos</b> (pontos brancos) caem
                  ao vivo nos vales conforme evoluem. Empilhe fatias e baixe a opacidade pra ver a 3ª dimensão.{' '}
                  <b style={{ color: '#ff2d9b' }}>○ rosa</b> = melhor · <b style={{ color: '#00e5ff' }}>◇ ciano</b> = mínimo teórico.
                </>
              )}
              {' '}Arraste pra girar, scroll pra zoom.
            </div>

            <div style={{ width: '100%', height: 560 }}>
              <Plot
                data={plotData as unknown as Plotly.Data[]}
                layout={plotLayout as unknown as Partial<Plotly.Layout>}
                config={{ displaylogo: false, responsive: true, displayModeBar: false } as unknown as Partial<Plotly.Config>}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          </div>
        </Card>
      )}

      {/* Convergência — visível durante e depois do treino */}
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
                (ótimo teórico: <b style={{ color: '#ffff00' }}>0</b> em (0,0,0))
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
          Tem <b>um único mínimo global em (0, 0, 0)</b> com f = 0, e <b>centenas de mínimos locais</b>
          perto de pontos inteiros como (±1, 0, 0), (±1, ±1, 0), (2, 1, −1)… (levemente puxados rumo à origem
          pelo termo x²) — é a <b>rede de pontos</b> do modo "Cubo de busca". O desafio do AG é não ficar preso
          num desses mínimos locais pelo caminho até a origem.
        </div>
      </Card>
    </div>
  );
}
