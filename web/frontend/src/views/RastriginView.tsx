import { useState, useRef, useMemo, useEffect, useCallback, memo, type ComponentType, type CSSProperties } from 'react';
import * as factoryNS from 'react-plotly.js/factory';
// @ts-expect-error — plotly.js-dist-min não tem types separados; o factory aceita.
import * as PlotlyNS from 'plotly.js-dist-min';

// Desembrulha CJS↔ESM caçando algo callable (a função factory).
function unwrapFactory(mod: unknown): (p: unknown) => ComponentType<{
  data: unknown; layout: unknown; config: unknown; style?: CSSProperties; useResizeHandler?: boolean;
  onInitialized?: (figure: unknown, graphDiv: unknown) => void;
  onUpdate?: (figure: unknown, graphDiv: unknown) => void;
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

// Acesso tipado-frouxo ao restyle imperativo (anima sem Plotly.react → orbit livre).
const plotlyRestyle = (gd: unknown, update: Record<string, unknown[]>, traces: number[]) =>
  (Plotly as { restyle: (gd: unknown, u: unknown, t: number[]) => unknown }).restyle(gd, update, traces);
const plotlyRelayout = (gd: unknown, update: Record<string, unknown>) =>
  (Plotly as { relayout: (gd: unknown, u: unknown) => unknown }).relayout(gd, update);

// <Plot> isolado em memo: só re-renderiza (→ Plotly.react) quando data/layout/onInit
// mudam de referência (mudanças ESTRUTURAIS). Trocar de frame NÃO re-renderiza isto,
// então o arrasto da câmera nunca é interrompido durante o replay.
interface CanvasProps { data: unknown; layout: unknown; onInit: (gd: unknown) => void; }
const PlotCanvas = memo(function PlotCanvas({ data, layout, onInit }: CanvasProps) {
  return (
    <Plot
      data={data as unknown as Plotly.Data[]}
      layout={layout as unknown as Partial<Plotly.Layout>}
      config={{ displaylogo: false, responsive: true, displayModeBar: false } as unknown as Partial<Plotly.Config>}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onInitialized={(_f, gd) => onInit(gd)}
      onUpdate={(_f, gd) => onInit(gd)}
    />
  );
});

import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip, ReferenceLine,
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

const NX = 100;
const CMAX_F = 50;
const TRAIL = 6;

function termR(v: number): number {
  return v * v - 10 * Math.cos(2 * Math.PI * v);
}
function rastrigin3(x: number, y: number, z: number): number {
  return 30 + termR(x) + termR(y) + termR(z);
}
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

const SLICE_VALUES = [-2, -1, 0, 1, 2];
type Modo = 'superficie' | 'espaco';

interface Frame {
  gen: number;
  xs: number[]; ys: number[]; zs: number[]; fxs: number[];
  bestX: number[]; bestFx: number;
}
// Índices dos traces dinâmicos no array de dados (pra restyle por geração).
interface DynIdx { ghosts: number[]; comet: number | null; best: number | null; otimo: number | null; pop: number | null; }

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
  const [domMinStr, setDomMinStr] = useState('-5.12');
  const [domMaxStr, setDomMaxStr] = useState('5.12');
  const domMin = parseFloat(domMinStr);
  const domMax = parseFloat(domMaxStr);

  // Estado de treino / replay
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<RastResult | null>(null);
  const [chartData, setChartData] = useState<{ gen: number; melhor: number; melhorAcum: number; media: number }[]>([]);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState('1');
  const [tween, setTween] = useState(false);   // movimento fluido (interpola entre gerações)
  const [autoRot, setAutoRot] = useState(false); // câmera girando sozinha
  const framesRef = useRef<Frame[]>([]);
  const tFloatRef = useRef(0);

  // Controles do 3D
  const [modo, setModo] = useState<Modo>('superficie');
  const [slices, setSlices] = useState<Record<number, boolean>>({ [-2]: false, [-1]: false, 0: true, 1: false, 2: false });
  const [opacidade, setOpacidade] = useState(0.85);
  const [mostrarPop, setMostrarPop] = useState(true);
  const [mostrarMinimos, setMostrarMinimos] = useState(true);
  const [mostrarOtimo, setMostrarOtimo] = useState(true);
  const [mostrarContorno, setMostrarContorno] = useState(true); // mapa de calor no chão (modo superfície)

  // Laboratório dos operadores (Aula 15) — mostra a geometria do cruzamento real.
  const [labOp, setLabOp] = useState<RastCruzamento>('radcliff');
  const [labBeta, setLabBeta] = useState(0.3);
  const [labA, setLabA] = useState<number[]>([-3.4, 2.6, -1.4]);
  const [labB, setLabB] = useState<number[]>([3.6, -2.2, 2.4]);

  const closeSSE = useRef<(() => void) | null>(null);
  const gdRef = useRef<unknown>(null);
  const giRef = useRef(0);
  const playTimerRef = useRef<number | null>(null);
  const dynIdxRef = useRef<DynIdx>({ ghosts: [], comet: null, best: null, otimo: null, pop: null });

  const onInit = useCallback((gd: unknown) => { gdRef.current = gd; }, []);

  useEffect(() => () => {
    if (closeSSE.current) closeSSE.current();
  }, []);

  // Loop de replay (passo inteiro) — playTimerRef garante UM único intervalo.
  // Desligado quando "tween" está ativo (aí o loop em rAF abaixo assume).
  useEffect(() => {
    if (playTimerRef.current) { clearInterval(playTimerRef.current); playTimerRef.current = null; }
    if (!playing || tween || frames.length === 0) return;
    const ms = speed === '2' ? 40 : speed === '0.5' ? 160 : 80;
    playTimerRef.current = window.setInterval(() => {
      setFrameIdx(i => {
        if (i >= frames.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, ms);
    return () => {
      if (playTimerRef.current) { clearInterval(playTimerRef.current); playTimerRef.current = null; }
    };
  }, [playing, tween, speed, frames.length]);

  function limparEstado() {
    setResult(null);
    setChartData([]);
    framesRef.current = [];
    setFrames([]);
    setFrameIdx(0);
    setPlaying(false);
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
    closeSSE.current = apiSSE('/agrastrigin/train', {
      onMessage(data) {
        const s = data as RastStep;
        const pop = s.populacao ?? [];
        const frame: Frame = {
          gen: s.geracao,
          xs: pop.map(p => p.x[0]), ys: pop.map(p => p.x[1]), zs: pop.map(p => p.x[2]),
          fxs: pop.map(p => p.fx),
          bestX: s.melhorGlobalX, bestFx: s.melhorGlobalFx,
        };
        framesRef.current.push(frame);
        setFrames(framesRef.current.slice());
        setFrameIdx(framesRef.current.length - 1);
        setChartData(prev => [...prev, {
          gen: s.geracao, melhor: s.melhorFx, melhorAcum: s.melhorGlobalFx, media: s.mediaFx,
        }]);
      },
      onDone(data) {
        const r = data as RastResult;
        setResult(r);
        setTraining(false);
        closeSSE.current = null;
        const [x, y, z] = r.melhorX;
        show(`Melhor: f(${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}) = ${r.melhorFx.toFixed(4)} · ▶ pra reviver`);
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

  function handlePlayPause() {
    if (frameIdx >= frames.length - 1) setFrameIdx(0);
    setPlaying(p => !p);
  }

  const vizMin = result ? result.cfg.dominioMin : (Number.isFinite(domMin) ? domMin : -5.12);
  const vizMax = result ? result.cfg.dominioMax : (Number.isFinite(domMax) ? domMax : 5.12);

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
    const zFloor = -Math.round(zMax * 0.18); // plano do contorno, flutuando abaixo do relevo
    return { axis, base, zMax, zFloor };
  }, [vizMin, vizMax]);

  const sliceGrids = useMemo(() => {
    const m: Record<number, number[][]> = {};
    for (const c of SLICE_VALUES) {
      const stepC = 30 + termR(c);
      m[c] = surf.base.map(row => row.map(v => v + stepC));
    }
    return m;
  }, [surf]);

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

  const gi = Math.min(frameIdx, frames.length - 1);
  giRef.current = gi; // lido pelo plotData memo (que NÃO depende de gi)
  const cur = gi >= 0 ? frames[gi] : null;
  const hasFrames = frames.length > 0; // gatilho p/ (re)montar os traces quando surge a 1ª geração
  const sliceKey = SLICE_VALUES.map(c => (slices[c] ? 1 : 0)).join('');

  // DADOS estruturais do Plot — NÃO dependem da geração (gi). Coordenadas iniciais
  // dos traces dinâmicos vêm do frame atual via refs; o resto é via restyle.
  const plotData = useMemo(() => {
    const fr = framesRef.current;
    const g = giRef.current;
    const c0 = g >= 0 && g < fr.length ? fr[g] : null;
    const espaco = modo === 'espaco';
    const traces: unknown[] = [];
    const di: DynIdx = { ghosts: [], comet: null, best: null, otimo: null, pop: null };
    if (!c0) { dynIdxRef.current = di; return traces; }

    const zPop = (f: Frame) => (espaco ? f.zs : f.fxs);
    const zBest = (f: Frame) => (espaco ? (f.bestX?.[2] ?? 0) : f.bestFx);
    const colorbar = {
      title: { text: 'f(x,y,z)', side: 'right', font: { color: '#aaa', size: 10 } },
      len: 0.7, x: 1.02, tickfont: { color: '#aaa', size: 9 },
    };
    const otimoVisivel = mostrarOtimo && vizMin <= 0 && vizMax >= 0;

    // FUNÇÃO (fundo)
    if (espaco) {
      if (mostrarMinimos && lattice.xs.length > 0) {
        traces.push({
          type: 'scatter3d', mode: 'markers', name: 'mínimos locais',
          x: lattice.xs, y: lattice.ys, z: lattice.zs,
          marker: { size: 3, color: lattice.fs, colorscale: 'Jet', cmin: 0, cmax: CMAX_F, opacity: opacidade, showscale: true, colorbar, symbol: 'circle' },
          hovertemplate: 'mínimo local<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<br>f=%{marker.color:.3f}<extra></extra>',
        });
      }
    } else {
      // Contorno projetado no chão (z=0): tapete plano colorido por f(x,y,0) — a
      // "sombra" topo-down da Rastrigin, vista junto com o relevo 3D acima.
      if (mostrarContorno) {
        const floor = surf.base.map(r => r.map(() => surf.zFloor));
        traces.push({
          type: 'surface', x: surf.axis, y: surf.axis, z: floor,
          surfacecolor: sliceGrids[0], colorscale: 'Jet', cmin: 0, cmax: surf.zMax,
          showscale: false, opacity: 0.85, hoverinfo: 'skip',
          contours: { z: { show: false } },
          // iluminação plana → mapa de calor limpo, sem o granulado/jitter do sombreamento.
          lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 },
          name: 'contorno (chão)', showlegend: false,
        });
      }
      SLICE_VALUES.filter(c => slices[c]).forEach((c, idx) => {
        traces.push({
          type: 'surface', x: surf.axis, y: surf.axis, z: sliceGrids[c],
          colorscale: 'Jet', cmin: 0, cmax: surf.zMax, opacity: opacidade, showscale: idx === 0, colorbar,
          name: `z = ${c}`, showlegend: false, hoverinfo: 'skip',
        });
      });
    }

    // RASTROS (trilha) — fantasmas das gerações anteriores + cometa do melhor.
    // Liga/desliga pela LEGENDA do Plotly (grupo "rastros"), sem botão externo.
    {
      for (let j = 0; j < TRAIL; j++) {
        const fidx = g - (j + 1);
        const gf = fidx >= 0 ? fr[fidx] : null;
        const op = 0.28 * (1 - j / TRAIL);
        di.ghosts.push(traces.length);
        traces.push({
          type: 'scatter3d', mode: 'markers', name: 'rastros', legendgroup: 'rastros', showlegend: false,
          x: gf ? gf.xs : [], y: gf ? gf.ys : [], z: gf ? zPop(gf) : [],
          marker: { size: 2.5, color: '#7fe8ff', opacity: Math.max(op, 0.05) },
          hoverinfo: 'skip',
        });
      }
      const bx: number[] = [], by: number[] = [], bz: number[] = [];
      for (let f = 0; f <= g; f++) { const ff = fr[f]; if (ff?.bestX) { bx.push(ff.bestX[0]); by.push(ff.bestX[1]); bz.push(zBest(ff)); } }
      di.comet = traces.length;
      traces.push({ type: 'scatter3d', mode: 'lines', name: 'rastros (trilha + caminho do melhor)', legendgroup: 'rastros', showlegend: true, x: bx, y: by, z: bz, line: { color: '#ff2d9b', width: 4 }, opacity: 0.8, hoverinfo: 'skip' });
    }

    // MARCADORES
    if (c0.bestX) {
      di.best = traces.length;
      traces.push({
        type: 'scatter3d', mode: 'markers', name: 'melhor (até agora)',
        x: [c0.bestX[0]], y: [c0.bestX[1]], z: [zBest(c0)],
        marker: { size: 12, color: '#ff2d9b', symbol: 'circle-open', line: { color: '#ff2d9b', width: 3 } },
        hovertemplate: 'melhor<br>x=%{x:.3f}  y=%{y:.3f}<extra></extra>',
      });
    }
    if (otimoVisivel) {
      di.otimo = traces.length;
      traces.push({
        type: 'scatter3d', mode: 'markers', name: 'mín. teórico (0,0,0)',
        x: [0], y: [0], z: [0],
        marker: { size: 11, color: '#00e5ff', symbol: 'diamond-open', line: { color: '#00e5ff', width: 3 } },
        hovertemplate: 'mínimo teórico<br>f(0,0,0)=0<extra></extra>',
      });
    }

    // FILHOS (geração atual) — por cima
    if (mostrarPop) {
      di.pop = traces.length;
      traces.push(espaco ? {
        type: 'scatter3d', mode: 'markers', name: 'filhos (geração atual)',
        x: c0.xs, y: c0.ys, z: c0.zs,
        marker: { size: 5, color: c0.fxs, colorscale: 'Jet', cmin: 0, cmax: CMAX_F, opacity: 0.98, showscale: !mostrarMinimos, colorbar, line: { color: '#fff', width: 0.5 } },
        hovertemplate: 'filho<br>x=%{x:.3f}  y=%{y:.3f}  z=%{z:.3f}<br>f=%{marker.color:.4f}<extra></extra>',
      } : {
        type: 'scatter3d', mode: 'markers', name: 'filhos (geração atual)',
        x: c0.xs, y: c0.ys, z: c0.fxs,
        marker: { size: 4, color: '#ffffff', opacity: 0.98, line: { color: '#000', width: 1 } },
        hovertemplate: 'filho<br>x=%{x:.3f}  y=%{y:.3f}<br>f=%{z:.3f}<extra></extra>',
      });
    }

    dynIdxRef.current = di;
    return traces;
  }, [hasFrames, modo, surf, sliceGrids, lattice, sliceKey, opacidade, mostrarPop, mostrarMinimos, mostrarOtimo, mostrarContorno, vizMin, vizMax]); // eslint-disable-line react-hooks/exhaustive-deps

  // Aplica o frame via restyle (sem Plotly.react → orbit livre). frac∈[0,1)
  // interpola pop/melhor entre o frame `lo` e o seguinte (usado no tween).
  const applyRestyle = useCallback((lo: number, frac: number) => {
    const gd = gdRef.current;
    const fr = framesRef.current;
    const c = lo >= 0 && lo < fr.length ? fr[lo] : null;
    if (!gd || !c) return;
    const hi = Math.min(lo + 1, fr.length - 1);
    const c2 = fr[hi];
    const f = frac > 0 ? frac : 0;
    const lerp = (a: number, b: number) => a + (b - a) * f;
    const di = dynIdxRef.current;
    const espaco = modo === 'espaco';
    const zPop = (fr2: Frame) => (espaco ? fr2.zs : fr2.fxs);
    const zBest = (fr2: Frame) => (espaco ? (fr2.bestX?.[2] ?? 0) : fr2.bestFx);

    const xs: number[][] = [], ys: number[][] = [], zs: number[][] = [], inds: number[] = [];
    di.ghosts.forEach((ti, j) => {
      const fidx = lo - (j + 1);
      const gf = fidx >= 0 ? fr[fidx] : null;
      xs.push(gf ? gf.xs : []); ys.push(gf ? gf.ys : []); zs.push(gf ? zPop(gf) : []); inds.push(ti);
    });
    if (di.comet != null) {
      const bx: number[] = [], by: number[] = [], bz: number[] = [];
      for (let k = 0; k <= lo; k++) { const ff = fr[k]; if (ff?.bestX) { bx.push(ff.bestX[0]); by.push(ff.bestX[1]); bz.push(zBest(ff)); } }
      xs.push(bx); ys.push(by); zs.push(bz); inds.push(di.comet);
    }
    if (di.best != null && c.bestX && c2.bestX) {
      xs.push([lerp(c.bestX[0], c2.bestX[0])]); ys.push([lerp(c.bestX[1], c2.bestX[1])]); zs.push([lerp(zBest(c), zBest(c2))]); inds.push(di.best);
    }
    if (di.pop != null) {
      const z1 = zPop(c), z2 = zPop(c2);
      if (f === 0) { xs.push(c.xs); ys.push(c.ys); zs.push(z1); }
      else {
        const n = c.xs.length, px = new Array(n), py = new Array(n), pz = new Array(n);
        for (let i = 0; i < n; i++) { px[i] = lerp(c.xs[i], c2.xs[i] ?? c.xs[i]); py[i] = lerp(c.ys[i], c2.ys[i] ?? c.ys[i]); pz[i] = lerp(z1[i], z2[i] ?? z1[i]); }
        xs.push(px); ys.push(py); zs.push(pz);
      }
      inds.push(di.pop);
    }
    if (inds.length) plotlyRestyle(gd, { x: xs, y: ys, z: zs }, inds);
    if (espaco && di.pop != null) plotlyRestyle(gd, { 'marker.color': [c.fxs] }, [di.pop]);
  }, [modo]);

  // Sincroniza o frame exibido (passo inteiro / scrub / play normal). Em tween+play
  // quem comanda é o rAF abaixo.
  useEffect(() => {
    if (playing && tween) return;
    applyRestyle(gi, 0);
  }, [gi, hasFrames, modo, sliceKey, mostrarPop, mostrarMinimos, mostrarOtimo, vizMin, vizMax, playing, tween, applyRestyle]);

  // Tween: loop em requestAnimationFrame interpolando entre gerações (fluido).
  useEffect(() => {
    if (!(playing && tween) || frames.length === 0) return;
    tFloatRef.current = gi;
    // No tween, cada geração dura ~3× mais → dá pra VER a interpolação (gliding).
    const frameMs = (speed === '2' ? 40 : speed === '0.5' ? 160 : 80) * 3;
    let raf = 0;
    let last = 0;
    const stepFn = (now: number) => {
      if (last === 0) last = now;
      const dt = now - last; last = now;
      tFloatRef.current += dt / frameMs;
      const end = frames.length - 1;
      if (tFloatRef.current >= end) {
        applyRestyle(end, 0);
        setFrameIdx(end);
        setPlaying(false);
        return;
      }
      const t = tFloatRef.current;
      const lo = Math.floor(t);
      applyRestyle(lo, t - lo);
      const rounded = Math.round(t);
      setFrameIdx(prev => (prev !== rounded ? rounded : prev));
      raf = requestAnimationFrame(stepFn);
    };
    raf = requestAnimationFrame(stepFn);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, tween, speed, frames.length, applyRestyle]);

  // Auto-rotação: gira a câmera em torno do eixo vertical. Throttle a ~25fps +
  // velocidade por tempo (dt) → não briga tanto com o redraw da animação (menos trava).
  useEffect(() => {
    if (!autoRot) return;
    let raf = 0;
    let angle: number | null = null;
    let last = 0, acc = 0;
    const tick = (now: number) => {
      if (last === 0) last = now;
      acc += now - last; last = now;
      if (acc >= 40) {
        const gd = gdRef.current as { layout?: { scene?: { camera?: { eye?: { x: number; y: number; z: number } } } } } | null;
        const eye = gd?.layout?.scene?.camera?.eye;
        if (eye) {
          if (angle === null) angle = Math.atan2(eye.y, eye.x);
          const r = Math.sqrt(eye.x * eye.x + eye.y * eye.y);
          angle += 0.5 * (acc / 1000); // ~0.5 rad/s, independente do FPS
          plotlyRelayout(gd, { 'scene.camera.eye': { x: r * Math.cos(angle), y: r * Math.sin(angle), z: eye.z } });
        }
        acc = 0;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [autoRot]);

  const plotLayout = useMemo(() => {
    const espaco = modo === 'espaco';
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 10, b: 0 },
      paper_bgcolor: '#0a0a0a',
      plot_bgcolor: '#0a0a0a',
      uirevision: modo,
      scene: {
        xaxis: { title: { text: 'x', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        yaxis: { title: { text: 'y', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        zaxis: espaco
          ? { title: { text: 'z', font: { color: '#aaa' } }, range: [vizMin, vizMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } }
          : { title: { text: 'f', font: { color: '#aaa' } }, range: [mostrarContorno ? surf.zFloor : 0, surf.zMax], gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 10 } },
        bgcolor: '#0a0a0a',
        aspectmode: 'manual',
        aspectratio: espaco ? { x: 1, y: 1, z: 1 } : { x: 1, y: 1, z: 0.7 },
        camera: { eye: espaco ? { x: 1.5, y: 1.5, z: 1.5 } : { x: 1.25, y: 1.25, z: 1.85 } },
      },
      legend: { font: { color: '#aaa', size: 11 }, x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)', groupclick: 'togglegroup', itemclick: 'toggle', itemdoubleclick: 'toggleothers' },
      showlegend: true,
    };
  }, [modo, vizMin, vizMax, surf.zMax, surf.zFloor, mostrarContorno]);

  // Laboratório dos operadores: os filhos sempre caem na RETA dos pais.
  // P(t) = (1−t)·A + t·B → A em t=0, B em t=1.
  function sortearPais() {
    const r = () => +(vizMin + Math.random() * (vizMax - vizMin)).toFixed(2);
    setLabA([r(), r(), r()]);
    setLabB([r(), r(), r()]);
  }
  const labData = useMemo(() => {
    const A = labA, B = labB;
    const P = (t: number) => [A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])];
    const inDom = (p: number[]) => p.every(v => v >= vizMin && v <= vizMax);
    const traces: unknown[] = [];
    const l0 = P(-0.7), l1 = P(1.7);
    traces.push({ type: 'scatter3d', mode: 'lines', x: [l0[0], l1[0]], y: [l0[1], l1[1]], z: [l0[2], l1[2]], line: { color: '#444', width: 2, dash: 'dot' }, hoverinfo: 'skip', showlegend: false });
    traces.push({ type: 'scatter3d', mode: 'lines', name: 'segmento A–B', x: [A[0], B[0]], y: [A[1], B[1]], z: [A[2], B[2]], line: { color: '#999', width: 5 }, hoverinfo: 'skip', showlegend: false });
    traces.push({
      type: 'scatter3d', mode: 'markers+text', name: 'pais', x: [A[0], B[0]], y: [A[1], B[1]], z: [A[2], B[2]],
      text: ['A', 'B'], textposition: 'top center', textfont: { color: '#fff', size: 13 },
      marker: { size: 8, color: ['#3b82f6', '#22c55e'], line: { color: '#fff', width: 1 } },
      hovertemplate: '%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
    });
    if (labOp === 'radcliff') {
      const pts = [labBeta, 1 - labBeta].map(P);
      traces.push({
        type: 'scatter3d', mode: 'markers', name: 'filhos RADCLIFF', x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
        marker: { size: 8, color: '#00e5ff', symbol: 'diamond', line: { color: '#fff', width: 1 } },
        hovertemplate: 'filho RADCLIFF (entre os pais)<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
      });
    } else {
      const defs = [{ t: -0.5, l: 'xb = 1.5·A − 0.5·B' }, { t: 0.5, l: 'xa = 0.5·A + 0.5·B' }, { t: 1.5, l: 'xc = −0.5·A + 1.5·B' }];
      const pts = defs.map(d => P(d.t));
      const valid = pts.map(inDom);
      traces.push({
        type: 'scatter3d', mode: 'markers', name: 'filhos WRIGHT', x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
        marker: { size: 9, color: valid.map(v => (v ? '#ff9d2e' : '#ff3b3b')), symbol: valid.map(v => (v ? 'circle' : 'x')), line: { color: '#fff', width: 1 } },
        text: defs.map((d, i) => d.l + (valid[i] ? '' : '  (fora do domínio → clamp)')),
        hovertemplate: '%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
      });
    }
    return traces;
  }, [labA, labB, labBeta, labOp, vizMin, vizMax]);

  const labLayout = useMemo(() => {
    // Range alargado pra caber os filhos WRIGHT extrapolados (e a reta), senão
    // eles ficariam clipados fora do cubo do domínio.
    const A = labA, B = labB;
    const P = (t: number) => [A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])];
    let lo = Infinity, hi = -Infinity;
    [A, B, P(-0.7), P(1.7)].forEach(p => p.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    const pad = (hi - lo) * 0.12 || 1;
    const range = [lo - pad, hi + pad];
    const ax = (t: string) => ({ title: { text: t, font: { color: '#aaa' } }, range, gridcolor: '#222', zerolinecolor: '#444', tickfont: { color: '#888', size: 9 } });
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 6, b: 0 },
      paper_bgcolor: '#0a0a0a', plot_bgcolor: '#0a0a0a',
      uirevision: 'lab',
      scene: {
        xaxis: ax('x'), yaxis: ax('y'), zaxis: ax('z'),
        bgcolor: '#0a0a0a', aspectmode: 'cube',
        camera: { eye: { x: 1.5, y: 1.5, z: 1.3 } },
      },
      legend: { font: { color: '#aaa', size: 11 }, x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' },
      showlegend: true,
    };
  }, [labA, labB]);

  const isTorneio = selecao === 'torneio';
  const lastGen = frames.length ? frames[frames.length - 1].gen : 0;
  const melhorAtual = cur?.bestFx;
  const distOtimo = cur?.bestX ? Math.sqrt(cur.bestX.reduce((a, b) => a + b * b, 0)) : null;
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
            <input type="text" inputMode="decimal" value={domMinStr} onChange={e => setDomMinStr(e.target.value)}
              style={{ width: 76, textAlign: 'center', background: 'var(--surface-2)', border: '1px solid #333', borderRadius: 4, color: 'var(--cyan)', padding: '6px 8px', fontFamily: 'JetBrains Mono', fontSize: 13 }} />
            <span>,</span>
            <input type="text" inputMode="decimal" value={domMaxStr} onChange={e => setDomMaxStr(e.target.value)}
              style={{ width: 76, textAlign: 'center', background: 'var(--surface-2)', border: '1px solid #333', borderRadius: 4, color: 'var(--cyan)', padding: '6px 8px', fontFamily: 'JetBrains Mono', fontSize: 13 }} />
            <span>]</span>
          </div>
          <div style={{ fontSize: 11, color: '#777', marginTop: 6 }}>padrão do Rastrigin: [-5.12, 5.12]</div>
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard title="Geração" value={cur ? cur.gen.toLocaleString() : '—'} label={`de ${parseInt(maxGeracoes).toLocaleString()}`} color="green" pulse={training} />
        <MetricCard title="Melhor f(x,y,z)" value={melhorAtual !== undefined ? melhorAtual.toFixed(4) : '—'} label={melhorAtual !== undefined && melhorAtual < 0.01 ? '≈ ótimo global (0)' : 'menor = melhor (alvo: 0)'} color="cyan" />
        <MetricCard title="Distância ao ótimo" value={distOtimo !== null ? `‖x‖ = ${distOtimo.toFixed(3)}` : '—'} label="‖(x,y,z) − (0,0,0)‖" />
      </div>

      {/* Visualização 3D */}
      {frames.length > 0 && (
        <Card title={`Visualização 3D — população${training ? ' (ao vivo)' : ' · replay'} e função Rastrigin`} style={{ marginBottom: 16 }}>
          <div style={{ padding: 8 }}>
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

            {/* Player */}
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center',
              marginBottom: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 6,
              fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--muted)',
            }}>
              <button className="btn" onClick={() => { setPlaying(false); setFrameIdx(0); }} disabled={training}
                style={{ fontSize: 14, padding: 0, width: 38, height: 32, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }} title="reiniciar">⟲</button>
              <button className="btn btn-primary" onClick={handlePlayPause} disabled={training}
                style={{ fontSize: 13, padding: 0, width: 44, height: 32, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }} title={playing ? 'pausar' : 'reproduzir'}>{playing ? '⏸' : '▶'}</button>
              <input type="range" min={0} max={Math.max(frames.length - 1, 0)} value={gi}
                onChange={e => { setPlaying(false); setFrameIdx(parseInt(e.target.value)); }} disabled={training}
                style={{ flex: 1, minWidth: 180 }} />
              <span style={{ color: 'var(--cyan)', minWidth: 92, textAlign: 'right' }}>geração {cur?.gen ?? 0}/{lastGen}</span>
              <span style={{ display: 'flex', gap: 4 }}>
                {['0.5', '1', '2'].map(s => (
                  <button key={s} className="btn" onClick={() => setSpeed(s)} aria-pressed={speed === s}
                    style={{ fontSize: 11, padding: '4px 8px', opacity: speed === s ? 1 : 0.45, borderColor: speed === s ? 'var(--cyan)' : '#333' }}>{s}×</button>
                ))}
              </span>
              <button className="btn" onClick={() => setTween(v => !v)} aria-pressed={tween} title="movimento fluido entre gerações (mais lento, pra dar pra ver)"
                style={{ fontSize: 11, padding: '5px 10px', opacity: tween ? 1 : 0.45, borderColor: tween ? 'var(--cyan)' : '#333' }}>suave</button>
              <button className="btn" onClick={() => setAutoRot(v => !v)} aria-pressed={autoRot} title="câmera girando sozinha"
                style={{ fontSize: 11, padding: '5px 10px', opacity: autoRot ? 1 : 0.45, borderColor: autoRot ? 'var(--cyan)' : '#333' }}>girar</button>
              {training && <span style={{ color: 'var(--green)' }}>● ao vivo</span>}
            </div>

            {/* Controles por modo */}
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
              marginBottom: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 6,
            }}>
              {modo === 'espaco' ? (
                <>
                  <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>MOSTRAR:</span>
                  <button className="btn" onClick={() => setMostrarPop(v => !v)} aria-pressed={mostrarPop} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarPop ? 1 : 0.4 }}>filhos</button>
                  <button className="btn" onClick={() => setMostrarMinimos(v => !v)} aria-pressed={mostrarMinimos} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarMinimos ? 1 : 0.4 }}>mínimos locais</button>
                  <button className="btn" onClick={() => setMostrarOtimo(v => !v)} aria-pressed={mostrarOtimo} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarOtimo ? 1 : 0.4 }}>mín. teórico</button>
                </>
              ) : (
                <>
                  <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>CAMADAS (fatias z):</span>
                  {SLICE_VALUES.map(c => (
                    <button key={c} className="btn" onClick={() => toggleSlice(c)} aria-pressed={slices[c]}
                      style={{ fontSize: 11, padding: '5px 12px', opacity: slices[c] ? 1 : 0.4, borderColor: slices[c] ? 'var(--cyan)' : '#333' }}>z = {c}</button>
                  ))}
                  <div style={{ width: 1, height: 22, background: '#333', margin: '0 4px' }} />
                  <button className="btn" onClick={() => setMostrarPop(v => !v)} aria-pressed={mostrarPop} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarPop ? 1 : 0.4 }}>filhos</button>
                  <button className="btn" onClick={() => setMostrarOtimo(v => !v)} aria-pressed={mostrarOtimo} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarOtimo ? 1 : 0.4 }}>mín. teórico</button>
                  <button className="btn" onClick={() => setMostrarContorno(v => !v)} aria-pressed={mostrarContorno} style={{ fontSize: 11, padding: '5px 10px', opacity: mostrarContorno ? 1 : 0.4 }}>contorno (chão)</button>
                </>
              )}
              <div style={{ width: 1, height: 22, background: '#333', margin: '0 4px' }} />
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
                {opacLabel}
                <input type="range" min={0} max={1} step={0.02} value={opacidade} onChange={e => setOpacidade(parseFloat(e.target.value))} style={{ width: 120 }} />
                <span style={{ color: 'var(--cyan)' }}>{opacidade.toFixed(2)}</span>
              </label>
            </div>

            <div style={{
              fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
              padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: 'JetBrains Mono',
            }}>
              {modo === 'espaco' ? (
                <>Cubo do <b>espaço de busca (x, y, z)</b> — cada <b>filho</b> na posição real, cor = aptidão.
                  Os <b>mínimos locais</b> (rede) são as armadilhas. </>
              ) : (
                <>Superfície <b>f(x, y)</b> (a "caixa de ovos") — os <b style={{ color: '#fff' }}>filhos</b> caem nos vales.{' '}</>
              )}
              Os <b style={{ color: '#7fe8ff' }}>rastros azuis</b> são as gerações anteriores e a{' '}
              <b style={{ color: '#ff2d9b' }}>linha rosa</b> é o caminho do melhor.{' '}
              Use <b>▶</b> / arraste o tempo pra reviver. Gire o 3D livremente até durante o play.
            </div>

            <div style={{ width: '100%', height: 560 }}>
              <PlotCanvas data={plotData} layout={plotLayout} onInit={onInit} />
            </div>
          </div>
        </Card>
      )}

      {/* Convergência */}
      {chartData.length > 0 && (
        <Card title="Convergência — f(x,y,z) por geração (escala log; alvo = 0)" style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="gen" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} scale="log" domain={[0.001, 'auto']} allowDataOverflow />
                <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }} labelFormatter={(v) => `geração ${v}`} formatter={(v) => Number(v).toFixed(4)} />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                {cur && <ReferenceLine x={cur.gen} stroke="#fff" strokeOpacity={0.6} strokeDasharray="2 2" />}
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
              <span style={{ color: '#888', marginLeft: 8 }}>(ótimo teórico: <b style={{ color: '#ffff00' }}>0</b> em (0,0,0))</span>
            </div>
          </div>
        </Card>
      )}

      {/* Laboratório dos operadores */}
      <Card title="Laboratório dos operadores (Aula 15) — geometria do cruzamento real" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
            marginBottom: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 6,
          }}>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>OPERADOR:</span>
            <button className="btn" onClick={() => setLabOp('radcliff')} aria-pressed={labOp === 'radcliff'}
              style={{ fontSize: 11, padding: '5px 12px', borderColor: labOp === 'radcliff' ? 'var(--cyan)' : '#333', opacity: labOp === 'radcliff' ? 1 : 0.5 }}>RADCLIFF</button>
            <button className="btn" onClick={() => setLabOp('wright')} aria-pressed={labOp === 'wright'}
              style={{ fontSize: 11, padding: '5px 12px', borderColor: labOp === 'wright' ? 'var(--cyan)' : '#333', opacity: labOp === 'wright' ? 1 : 0.5 }}>WRIGHT</button>
            <div style={{ width: 1, height: 22, background: '#333', margin: '0 4px' }} />
            <button className="btn" onClick={sortearPais} style={{ fontSize: 11, padding: '5px 12px' }}>↻ sortear pais</button>
            {labOp === 'radcliff' && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginLeft: 8 }}>
                β
                <input type="range" min={0.05} max={0.95} step={0.05} value={labBeta} onChange={e => setLabBeta(parseFloat(e.target.value))} style={{ width: 140 }} />
                <span style={{ color: 'var(--cyan)' }}>{labBeta.toFixed(2)}</span>
              </label>
            )}
          </div>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: 'JetBrains Mono',
          }}>
            Os filhos sempre caem na <b>reta</b> que passa pelos pais <b style={{ color: '#3b82f6' }}>A</b> e <b style={{ color: '#22c55e' }}>B</b>.{' '}
            {labOp === 'radcliff' ? (
              <>
                <b>RADCLIFF</b> (combinação convexa): os 2 filhos ficam <b>entre</b> A e B (em t = β e t = 1−β) —
                <b> nunca extrapolam</b>, sempre dentro do domínio. Mexa no β pra ver eles deslizarem no segmento.
              </>
            ) : (
              <>
                <b>WRIGHT</b> gera 3: o do meio (0,5·A+0,5·B) e dois que <b>extrapolam</b> além de A e B
                (1,5·A−0,5·B e −0,5·A+1,5·B). Os que saem do domínio (<b style={{ color: '#ff3b3b' }}>×</b>) são clampados.
              </>
            )}
          </div>
          <div style={{ width: '100%', height: 380 }}>
            <Plot
              data={labData as unknown as Plotly.Data[]}
              layout={labLayout as unknown as Partial<Plotly.Layout>}
              config={{ displaylogo: false, responsive: true, displayModeBar: false } as unknown as Partial<Plotly.Config>}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          </div>
        </div>
      </Card>

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
