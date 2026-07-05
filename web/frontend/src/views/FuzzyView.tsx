import { useState, useEffect, useMemo, memo, type ComponentType, type CSSProperties } from 'react';
import * as factoryNS from 'react-plotly.js/factory';
// @ts-expect-error — plotly.js-dist-min não tem types separados; o factory aceita.
import * as PlotlyNS from 'plotly.js-dist-min';
import {
  ComposedChart, Line, Area, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Tooltip, ReferenceLine, ReferenceDot,
} from 'recharts';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost } from '../api/client';
import type {
  FuzzyMeta, FuzzyResultado, FuzzySuperficie, FuzzyVariavel, FuzzyRegraAtivada,
} from '../api/types';

// =============================================================================
// Trabalho 16 — Qualidade da Água com Lógica Fuzzy (Aulas 17–19).
// Pipeline Mamdani AO VIVO: sliders → fuzzificação → 45 regras → agregação →
// centroide. Toda a matemática roda no backend Go (pacote fuzzy); esta view só
// DESENHA o trace devolvido por /api/fuzzy/evaluate + a meta de /api/fuzzy/meta.
// =============================================================================

// ---- Plotly via factory (mesmo padrão do RastriginView) ---------------------
function unwrapFactory(mod: unknown): (p: unknown) => ComponentType<{
  data: unknown; layout: unknown; config: unknown; style?: CSSProperties; useResizeHandler?: boolean;
}> {
  type Maybe = { default?: unknown; [k: string]: unknown };
  const m = mod as Maybe;
  if (typeof mod === 'function') return mod as never;
  if (typeof m.default === 'function') return m.default as never;
  if (m.default && typeof (m.default as Maybe).default === 'function') return (m.default as Maybe).default as never;
  // eslint-disable-next-line no-console
  console.error('react-plotly.js/factory shape inesperada:', mod);
  throw new Error('react-plotly.js factory não encontrado');
}
function unwrapPlotly(mod: unknown): unknown {
  type Maybe = { default?: unknown; newPlot?: unknown };
  const m = mod as Maybe;
  if (m && typeof m.newPlot === 'function') return mod;
  if (m && m.default && typeof (m.default as Maybe).newPlot === 'function') return m.default;
  return mod;
}
const Plot = unwrapFactory(factoryNS)(unwrapPlotly(PlotlyNS));

interface PlotProps { data: unknown; layout: unknown; }
const SurfacePlot = memo(function SurfacePlot({ data, layout }: PlotProps) {
  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displaylogo: false, responsive: true, displayModeBar: false }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
    />
  );
});

// ---- helpers de desenho (a inferência em si fica toda no backend) -----------
type Trap = [number, number, number, number];
function trapMu(x: number, [a, b, c, d]: Trap): number {
  if (x < a || x > d) return 0;
  if (x >= b && x <= c) return 1;
  if (x < b) return (x - a) / (b - a);
  return (d - x) / (d - c);
}

// Amostra os trapézios de uma variável numa grade comum (inclui os cantos, pra
// os vértices ficarem exatos no gráfico).
function sampleVariavel(v: FuzzyVariavel, n = 121): Record<string, number>[] {
  const xs = new Set<number>();
  for (let i = 0; i < n; i++) xs.add(v.min + (v.max - v.min) * i / (n - 1));
  v.termos.forEach(t => t.trap.forEach(p => { if (p >= v.min && p <= v.max) xs.add(p); }));
  return [...xs].sort((a, b) => a - b).map(x => {
    const row: Record<string, number> = { x };
    v.termos.forEach(t => { row[t.id] = trapMu(x, t.trap); });
    return row;
  });
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// "inadequado baixo" → "inad. baixo" (rótulos curtos nas tabelas de regras)
function abreviarTermo(nome: string): string {
  return nome.replace('inadequado', 'inad.').replace('adequado', 'adeq.')
    .replace('inadequada', 'inad.').replace('adequada', 'adeq.');
}

const MONO = 'JetBrains Mono';
const CYAN = '#00e5ff';

// ---- presets ----------------------------------------------------------------
interface Preset { nome: string; cor: number; ph: number; turbidez: number; hint: string; }
const PRESETS: Preset[] = [
  { nome: '📖 Apostila', cor: 15, ph: 7, turbidez: 0, hint: 'o exemplo canônico: deve dar Q = 0.60 (adequada)' },
  { nome: '🚰 Torneira', cor: 2, ph: 7.2, turbidez: 0.3, hint: 'água tratada dentro dos limites SABESP' },
  { nome: '🌧 Pós-chuva', cor: 12, ph: 6.8, turbidez: 3.5, hint: 'cor e turbidez sobem depois da chuva' },
  { nome: '🏊 Piscina', cor: 3, ph: 9.5, turbidez: 0.8, hint: 'límpida, mas pH alto demais' },
  { nome: '🏭 Rio poluído', cor: 25, ph: 5, turbidez: 8, hint: 'tudo fora dos limites' },
];

const EIXOS_OPTIONS = [
  { value: 'ph|turbidez', label: 'pH × Turbidez (cor fixa)' },
  { value: 'cor|ph', label: 'Cor × pH (turbidez fixa)' },
  { value: 'cor|turbidez', label: 'Cor × Turbidez (pH fixo)' },
];

export default function FuzzyView() {
  const { show } = useToast();

  const [meta, setMeta] = useState<FuzzyMeta | null>(null);
  const [cor, setCor] = useState(15);
  const [ph, setPh] = useState(7);
  const [turbidez, setTurbidez] = useState(0);
  const [resultado, setResultado] = useState<FuzzyResultado | null>(null);
  const [regraSel, setRegraSel] = useState<FuzzyRegraAtivada | null>(null);
  const [areaUnica, setAreaUnica] = useState(true); // saída: área agregada única vs. recortes por termo
  const [eixos, setEixos] = useState('ph|turbidez');
  const [superficie, setSuperficie] = useState<FuzzySuperficie | null>(null);

  useEffect(() => {
    apiGet<FuzzyMeta>('/fuzzy/meta').then(setMeta).catch(() => show('Erro ao carregar a definição do sistema fuzzy'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Inferência ao vivo — debounce curto pra acompanhar o arrasto do slider.
  useEffect(() => {
    const t = setTimeout(() => {
      apiPost<FuzzyResultado>('/fuzzy/evaluate', { cor, ph, turbidez })
        .then(setResultado)
        .catch(() => { /* transitório (ex: servidor reiniciando) — próximo tick resolve */ });
    }, 80);
    return () => clearTimeout(t);
  }, [cor, ph, turbidez]);

  // Superfície 3D — mais pesada (41×41 inferências), debounce maior.
  useEffect(() => {
    const [ex, ey] = eixos.split('|');
    const t = setTimeout(() => {
      apiGet<FuzzySuperficie>(`/fuzzy/surface?eixoX=${ex}&eixoY=${ey}&cor=${cor}&ph=${ph}&turbidez=${turbidez}`)
        .then(setSuperficie)
        .catch(() => { /* idem */ });
    }, 250);
    return () => clearTimeout(t);
  }, [eixos, cor, ph, turbidez]);

  // ---- derivados -------------------------------------------------------------
  const entradaAtual: Record<string, number> = { cor, ph, turbidez };
  const varCor = meta?.entradas[0];

  const chartsEntrada = useMemo(() => {
    if (!meta) return null;
    return meta.entradas.map(v => ({ v, data: sampleVariavel(v) }));
  }, [meta]);

  // Curva de saída: junta o recorte vindo do backend com os trapézios "cheios"
  // (desenhados a partir da meta, tracejados, só pra referência visual).
  const curvaSaida = useMemo(() => {
    if (!meta || !resultado) return [];
    return resultado.curva.map(p => {
      const row: Record<string, number> = {
        x: p.x, inadequada: p.inadequada, adequada: p.adequada, boa: p.boa, agregada: p.agregada,
      };
      meta.saida.termos.forEach(t => { row['ref_' + t.id] = trapMu(p.x, t.trap); });
      return row;
    });
  }, [meta, resultado]);

  // Índice força-por-célula pras tabelas de regras.
  const regraPorChave = useMemo(() => {
    const m = new Map<string, FuzzyRegraAtivada>();
    resultado?.regras.forEach(r => m.set(`${r.aparencia}|${r.ph}|${r.turbidez}`, r));
    return m;
  }, [resultado]);

  const nomeTermo = useMemo(() => {
    const m = new Map<string, string>();
    meta?.entradas.forEach(v => v.termos.forEach(t => m.set(`${v.id}:${t.id}`, t.nome)));
    meta?.saida.termos.forEach(t => m.set(`saida:${t.id}`, t.nome));
    return m;
  }, [meta]);

  const corSaida = useMemo(() => {
    const m = new Map<string, string>();
    meta?.saida.termos.forEach(t => m.set(t.id, t.cor));
    return m;
  }, [meta]);

  const q = resultado?.centroide ?? null;
  const classe = resultado?.classe ?? null;
  const classeNome = classe ? (nomeTermo.get(`saida:${classe}`) ?? classe) : '—';
  const classeCor = classe ? (corSaida.get(classe) ?? CYAN) : CYAN;

  const ehApostila = cor === 15 && ph === 7 && turbidez === 0;
  const apostilaOk = ehApostila && q !== null && Math.abs(q - 0.6) <= 0.01;

  // Ponto atual sobre a superfície (marcador 3D).
  const plotData = useMemo(() => {
    if (!superficie) return null;
    const [ex, ey] = eixos.split('|');
    const data: unknown[] = [{
      type: 'surface',
      x: superficie.xs, y: superficie.ys, z: superficie.z,
      cmin: 0, cmax: 1,
      colorscale: [
        [0, '#ff4d6d'], [0.4, '#ff9550'], [0.6, '#ffb020'], [0.8, '#a8d34f'], [1, '#3ddc84'],
      ],
      colorbar: {
        title: { text: 'Q', font: { color: '#8a94a3', family: MONO, size: 11 } },
        tickfont: { color: '#8a94a3', family: MONO, size: 10 },
        thickness: 12, len: 0.7,
      },
      opacity: 0.96,
    }];
    if (q !== null) {
      data.push({
        type: 'scatter3d', mode: 'markers',
        x: [entradaAtual[ex]], y: [entradaAtual[ey]], z: [q + 0.015],
        marker: { size: 6, color: CYAN, symbol: 'diamond' },
        name: 'entrada atual', showlegend: false,
        hovertemplate: `Q = ${q.toFixed(3)}<extra>entrada atual</extra>`,
      });
    }
    return data;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [superficie, q, eixos]);

  const plotLayout = useMemo(() => {
    const [ex, ey] = eixos.split('|');
    const nomeEixo: Record<string, string> = { cor: 'Cor (UH)', ph: 'pH', turbidez: 'Turbidez (UT)' };
    const eixo3d = (titulo: string, range?: number[]) => ({
      title: { text: titulo, font: { color: '#8a94a3', family: MONO, size: 11 } },
      tickfont: { color: '#667', family: MONO, size: 9 },
      gridcolor: '#26303c', zerolinecolor: '#26303c', backgroundcolor: 'rgba(0,0,0,0)',
      ...(range ? { range } : {}),
    });
    return {
      autosize: true,
      uirevision: 'keep', // preserva a câmera quando a grade atualiza
      paper_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        xaxis: eixo3d(nomeEixo[ex]),
        yaxis: eixo3d(nomeEixo[ey]),
        zaxis: eixo3d('Qualidade Q', [0, 1]),
        bgcolor: 'rgba(0,0,0,0)',
        camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
      },
    };
  }, [eixos]);

  // ---- sub-renders ------------------------------------------------------------
  const sliderRow = (
    label: string, unidade: string, min: number, max: number, step: number,
    value: number, onChange: (v: number) => void, varId: string,
  ) => {
    const pert = resultado?.pertinencias[varId];
    const termos = meta?.entradas.find(v => v.id === varId)?.termos ?? [];
    return (
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontFamily: MONO, fontSize: 12, color: 'var(--muted)', width: 118, flexShrink: 0 }}>
            {label}
          </span>
          <input
            type="range" min={min} max={max} step={step} value={value}
            aria-label={label}
            onChange={e => onChange(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: CYAN }}
          />
          <span style={{ fontFamily: MONO, fontSize: 14, color: CYAN, width: 86, textAlign: 'right', flexShrink: 0 }}>
            {value.toFixed(step < 1 ? 1 : 0)}{unidade && <span style={{ color: '#667', fontSize: 11 }}> {unidade}</span>}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 5, marginLeft: 128, flexWrap: 'wrap' }}>
          {termos.map(t => {
            const mu = pert?.[t.id] ?? 0;
            return (
              <span key={t.id} style={{
                fontFamily: MONO, fontSize: 10, padding: '2px 8px', borderRadius: 10,
                border: `1px solid ${mu > 0 ? t.cor : '#333'}`,
                color: mu > 0 ? t.cor : '#556',
                background: mu > 0 ? hexToRgba(t.cor, 0.12) : 'transparent',
              }}>
                {t.nome} μ={mu.toFixed(2)}
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  const tabelaRegras = (aparenciaId: string) => {
    if (!meta) return null;
    const apNome = nomeTermo.get(`cor:${aparenciaId}`) ?? aparenciaId;
    const apCor = varCor?.termos.find(t => t.id === aparenciaId)?.cor ?? '#888';
    const muAp = resultado?.pertinencias.cor?.[aparenciaId] ?? 0;
    return (
      <Card key={aparenciaId} style={muAp > 0 ? { border: `1px solid ${hexToRgba(apCor, 0.5)}` } : undefined}>
        <div className="card-title">
          SE aparência é <span style={{ color: apCor }}>{apNome.toUpperCase()}</span>
          <span style={{ color: muAp > 0 ? apCor : '#667', fontWeight: 400 }}> · μ={muAp.toFixed(2)}</span>
        </div>
        <div style={{ padding: '4px 10px 10px', overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontFamily: MONO, fontSize: 10.5, width: '100%' }}>
            <thead>
              <tr>
                <th style={{ padding: '3px 6px', color: '#667', textAlign: 'left', fontWeight: 400 }}>pH ↓ · turb →</th>
                {meta.ordemTurb.map(tb => (
                  <th key={tb} style={{ padding: '3px 6px', color: '#8a94a3', textAlign: 'center', fontWeight: 600 }}>
                    {abreviarTermo(nomeTermo.get(`turbidez:${tb}`) ?? tb)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {meta.ordemPh.map(phId => (
                <tr key={phId}>
                  <td style={{ padding: '3px 6px', color: '#8a94a3', whiteSpace: 'nowrap' }}>
                    {abreviarTermo(nomeTermo.get(`ph:${phId}`) ?? phId)}
                  </td>
                  {meta.ordemTurb.map(tbId => {
                    const r = regraPorChave.get(`${aparenciaId}|${phId}|${tbId}`);
                    const saidaId = r?.saida ?? meta.regras.find(
                      rg => rg.aparencia === aparenciaId && rg.ph === phId && rg.turbidez === tbId,
                    )?.saida ?? '';
                    const cCor = corSaida.get(saidaId) ?? '#888';
                    const forca = r?.forca ?? 0;
                    const sel = regraSel && regraSel.aparencia === aparenciaId && regraSel.ph === phId && regraSel.turbidez === tbId;
                    return (
                      <td key={tbId}
                        onClick={() => r && setRegraSel(sel ? null : r)}
                        title={r ? `força ${forca.toFixed(2)} — clique pra inspecionar` : ''}
                        style={{
                          padding: '4px 6px', textAlign: 'center', cursor: r ? 'pointer' : 'default',
                          color: forca > 0 ? cCor : hexToRgba(cCor, 0.8),
                          background: hexToRgba(cCor, 0.06 + 0.5 * forca),
                          border: sel ? `1px solid ${CYAN}` : '1px solid transparent',
                          borderRadius: 4, fontWeight: forca > 0 ? 700 : 400,
                        }}>
                        {abreviarTermo(nomeTermo.get(`saida:${saidaId}`) ?? saidaId)}
                        {forca > 0 && <span style={{ fontSize: 9, opacity: 0.85 }}> {forca.toFixed(2)}</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    );
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Fuzzy <span>Qualidade da Água</span></div>
          <div className="page-sub">
            Sistema de inferência <b>Mamdani</b> — Trabalho 16 · Aulas 17–19 · classifica a potabilidade
            (limites SABESP) a partir de cor aparente, pH e turbidez — apostila Jafelice · Barros · Bassanezi, ex. 2.7.3
          </div>
        </div>
        {apostilaOk && (
          <div style={{
            fontFamily: MONO, fontSize: 11, color: '#3ddc84', border: '1px solid #3ddc84',
            borderRadius: 6, padding: '6px 12px', background: hexToRgba('#3ddc84', 0.08),
          }}>
            ✓ exemplo da apostila validado: Q = 0.60 → adequada
          </div>
        )}
      </div>

      {/* Controle: presets + sliders */}
      <Card title="Amostra de água — arraste os sliders ou escolha um cenário" pulse style={{ marginBottom: 16 }}>
        <div style={{ padding: '4px 14px 12px' }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
            {PRESETS.map(p => {
              const ativo = p.cor === cor && p.ph === ph && p.turbidez === turbidez;
              return (
                <button key={p.nome} className="btn" title={p.hint}
                  onClick={() => { setCor(p.cor); setPh(p.ph); setTurbidez(p.turbidez); }}
                  style={{
                    fontSize: 11, padding: '6px 12px',
                    borderColor: ativo ? CYAN : '#333', opacity: ativo ? 1 : 0.7,
                  }}>
                  {p.nome}
                </button>
              );
            })}
          </div>
          {sliderRow('Cor aparente', 'UH', 0, 30, 0.5, cor, setCor, 'cor')}
          {sliderRow('pH', '', 0, 14, 0.1, ph, setPh, 'ph')}
          {sliderRow('Turbidez', 'UT', 0, 10, 0.1, turbidez, setTurbidez, 'turbidez')}
        </div>
      </Card>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard title="Qualidade Q (centroide)" value={q !== null ? q.toFixed(3) : '—'} label="defuzzificação em [0, 1]" color="cyan" />
        <MetricCard title="Classificação" value={classeNome.toUpperCase()} label="termo com maior pertinência em Q" valueStyle={{ color: classeCor }} />
        <MetricCard title="Regras ativas" value={resultado ? String(resultado.regrasAtivas) : '—'} label="de 45 (força > 0)" color="green" />
      </div>

      {/* 1) Fuzzificação */}
      <Card title="① Fuzzificação — os trapézios de cada entrada (limites SABESP)" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: MONO,
          }}>
            Cada valor nítido vira um <b>grau de pertinência μ ∈ [0,1]</b> em cada termo linguístico.
            A linha ciano é o valor atual do slider; os pontos marcam os μ que alimentam as regras.
            Os cruzamentos em μ=0.5 caem exatamente nos limites da SABESP (cor 5 e 15 UH · pH 6, 6.5, 8.5 e 10 · turbidez 1 e 5 UT).
          </div>
          <div className="grid-3">
            {chartsEntrada?.map(({ v, data }) => (
              <div key={v.id}>
                <div style={{ fontFamily: MONO, fontSize: 11, color: '#8a94a3', margin: '2px 0 4px 8px' }}>
                  {v.nome}{v.unidade ? ` (${v.unidade})` : ''}
                </div>
                <div style={{ width: '100%', height: 170 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -26 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis dataKey="x" type="number" domain={[v.min, v.max]} stroke="#555"
                        tick={{ fill: '#555', fontSize: 9, fontFamily: MONO }} tickLine={false} />
                      <YAxis domain={[0, 1.05]} ticks={[0, 0.5, 1]} stroke="#555"
                        tick={{ fill: '#555', fontSize: 9, fontFamily: MONO }} tickLine={false} />
                      <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: MONO }}
                        labelFormatter={x => `${v.nome}: ${Number(x).toFixed(2)}`}
                        formatter={(val, name) => [Number(val).toFixed(2), nomeTermo.get(`${v.id}:${String(name)}`) ?? name]} />
                      {v.termos.map(t => (
                        <Area key={t.id} dataKey={t.id} type="linear" stroke={t.cor} strokeWidth={2}
                          fill={hexToRgba(t.cor, 0.14)} isAnimationActive={false} dot={false} />
                      ))}
                      <ReferenceLine x={entradaAtual[v.id]} stroke={CYAN} strokeWidth={1.5} strokeDasharray="4 3" />
                      {v.termos.map(t => {
                        const mu = resultado?.pertinencias[v.id]?.[t.id] ?? 0;
                        return mu > 0.001 ? (
                          <ReferenceDot key={'d' + t.id} x={entradaAtual[v.id]} y={mu} r={4}
                            fill={t.cor} stroke="#10141a" strokeWidth={1.5} />
                        ) : null;
                      })}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* 2) Regras */}
      <Card title="② Base de regras — 45 regras (Tabelas 2.6 · 2.7 · 2.8 da apostila)" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: MONO,
          }}>
            Uma tabela por termo da <b>aparência</b> (cor). Cada célula é o consequente de
            <b> SE aparência E pH E turbidez</b>; o conectivo E é o <b>mínimo</b> dos três μ.
            Células <b>acesas</b> = regras disparadas agora (intensidade ∝ força). <b>Clique numa célula</b> pra inspecionar a regra.
          </div>
          <div className="grid-3">
            {varCor?.termos.map(t => tabelaRegras(t.id))}
          </div>

          {regraSel && (
            <div style={{
              marginTop: 12, padding: '12px 16px', background: 'var(--surface-2)', borderRadius: 6,
              border: `1px solid ${hexToRgba(CYAN, 0.4)}`, fontFamily: MONO, fontSize: 12.5, lineHeight: 2,
            }}>
              <div style={{ color: CYAN, fontSize: 11, marginBottom: 4 }}>🔍 INSPETOR DE REGRA</div>
              {(() => {
                const menor = Math.min(regraSel.muAparencia, regraSel.muPh, regraSel.muTurbidez);
                const parte = (rot: string, nome: string, mu: number) => (
                  <span>
                    {rot} é <b style={{ color: '#f0f2f5' }}>{nome}</b>{' '}
                    <span style={{
                      color: mu === menor ? '#ffb020' : '#8a94a3',
                      border: mu === menor ? '1px solid #ffb020' : '1px solid transparent',
                      borderRadius: 4, padding: '0 4px',
                    }}>μ={mu.toFixed(2)}</span>
                  </span>
                );
                return (
                  <div style={{ color: 'var(--muted)' }}>
                    <b style={{ color: '#c9a3ff' }}>SE</b> aparência {parte('', nomeTermo.get(`cor:${regraSel.aparencia}`) ?? '', regraSel.muAparencia)}{' '}
                    <b style={{ color: '#c9a3ff' }}>E</b> {parte('pH', nomeTermo.get(`ph:${regraSel.ph}`) ?? '', regraSel.muPh)}{' '}
                    <b style={{ color: '#c9a3ff' }}>E</b> {parte('turbidez', nomeTermo.get(`turbidez:${regraSel.turbidez}`) ?? '', regraSel.muTurbidez)}{' '}
                    <b style={{ color: '#c9a3ff' }}>ENTÃO</b> qualidade é{' '}
                    <b style={{ color: corSaida.get(regraSel.saida) }}>{nomeTermo.get(`saida:${regraSel.saida}`)}</b>
                    <br />
                    força = min(μ) = <b style={{ color: regraSel.forca > 0 ? '#ffb020' : '#667' }}>{regraSel.forca.toFixed(3)}</b>
                    {regraSel.forca === 0 && <span style={{ color: '#667' }}> — regra dormente pra esta amostra</span>}
                    {regraSel.forca > 0 && <span style={{ color: '#667' }}> — recorta o termo "{nomeTermo.get(`saida:${regraSel.saida}`)}" nessa altura</span>}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </Card>

      {/* 3) Agregação + defuzzificação */}
      <Card title="③ Agregação (max) e defuzzificação (centroide)" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: MONO,
          }}>
            Cada termo de saída é <b>recortado</b> (implicação min) na força máxima das regras que apontam pra ele
            (tracejado = trapézio original). O envelope <b style={{ color: CYAN }}>ciano</b> é o máximo dos recortes;
            a linha vertical é o <b>centroide</b> dessa área — o valor nítido Q.
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 11, fontFamily: MONO, color: '#888' }}>ver:</span>
            <button className="btn" onClick={() => setAreaUnica(true)} aria-pressed={areaUnica}
              style={{ fontSize: 11, padding: '5px 10px', borderColor: areaUnica ? CYAN : '#333', opacity: areaUnica ? 1 : 0.5 }}>
              área agregada única
            </button>
            <button className="btn" onClick={() => setAreaUnica(false)} aria-pressed={!areaUnica}
              style={{ fontSize: 11, padding: '5px 10px', borderColor: !areaUnica ? CYAN : '#333', opacity: !areaUnica ? 1 : 0.5 }}>
              recortes por termo
            </button>
          </div>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={curvaSaida} margin={{ top: 24, right: 16, bottom: 14, left: -18 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="x" type="number" domain={[0, 1]} stroke="#555"
                  tick={{ fill: '#555', fontSize: 10, fontFamily: MONO }} tickLine={false}
                  label={{ value: 'Qualidade da água (Q)', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 11 }} />
                <YAxis domain={[0, 1.05]} ticks={[0, 0.5, 1]} stroke="#555"
                  tick={{ fill: '#555', fontSize: 10, fontFamily: MONO }} tickLine={false} />
                <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: MONO }}
                  labelFormatter={x => `Q = ${Number(x).toFixed(3)}`}
                  formatter={(val, name) => {
                    const n = String(name);
                    if (n.startsWith('ref_')) return [Number(val).toFixed(2), (nomeTermo.get(`saida:${n.slice(4)}`) ?? n) + ' (original)'];
                    if (n === 'agregada') return [Number(val).toFixed(2), 'envelope agregado'];
                    return [Number(val).toFixed(2), (nomeTermo.get(`saida:${n}`) ?? n) + ' (recortado)'];
                  }} />
                {meta?.saida.termos.map(t => (
                  <Line key={'ref' + t.id} dataKey={'ref_' + t.id} type="linear" stroke={hexToRgba(t.cor, 0.45)}
                    strokeWidth={1} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                ))}
                {!areaUnica && meta?.saida.termos.map(t => (
                  <Area key={t.id} dataKey={t.id} type="linear" stroke={t.cor} strokeWidth={2}
                    fill={hexToRgba(t.cor, 0.22)} dot={false} isAnimationActive={false} />
                ))}
                {areaUnica ? (
                  <Area dataKey="agregada" type="linear" stroke={CYAN} strokeWidth={2.5}
                    fill={hexToRgba(CYAN, 0.24)} dot={false} isAnimationActive={false} />
                ) : (
                  <Line dataKey="agregada" type="linear" stroke={CYAN} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                )}
                {q !== null && (
                  <ReferenceLine x={q} stroke={CYAN} strokeWidth={2}
                    label={{ value: `Q = ${q.toFixed(3)}`, fill: CYAN, fontSize: 12, fontFamily: MONO, position: 'top' }} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          {resultado && (
            <div style={{ display: 'flex', gap: 10, marginTop: 8, marginLeft: 8, flexWrap: 'wrap' }}>
              {meta?.saida.termos.map(t => (
                <span key={t.id} style={{
                  fontFamily: MONO, fontSize: 11, padding: '3px 10px', borderRadius: 10,
                  border: `1px solid ${t.cor}`, color: t.cor, background: hexToRgba(t.cor, 0.08),
                }}>
                  {t.nome}: força {(resultado.forcaSaida[t.id] ?? 0).toFixed(2)}
                </span>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* 4) Superfície 3D */}
      <Card title="④ Superfície de decisão — Q em função de duas entradas" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
            <Select options={EIXOS_OPTIONS} value={eixos} onChange={setEixos} style={{ width: 260 }} />
            <span style={{ fontFamily: MONO, fontSize: 11, color: '#667' }}>
              a variável que não está nos eixos fica presa ao slider · losango ciano = amostra atual
            </span>
          </div>
          <div style={{ width: '100%', height: 440 }}>
            {plotData ? (
              <SurfacePlot data={plotData} layout={plotLayout} />
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontFamily: MONO }}>
                calculando superfície…
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Educacional */}
      <Card title="Como funciona — inferência fuzzy de Mamdani">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Por que fuzzy?</b> Os limites da SABESP são nítidos (ex: turbidez ≤ 1 UT é "boa"), mas a
          realidade não vira ruim de repente em 1.01 UT. A lógica fuzzy troca o {'{0, 1}'} da lógica clássica
          por <b>graus de pertinência μ ∈ [0, 1]</b>: uma água com turbidez 1.1 UT é <i>um pouco</i> boa e
          <i> um pouco</i> adequada ao mesmo tempo.
          <br /><br />
          <b>① Fuzzificação:</b> cada entrada nítida é projetada nas funções de pertinência trapezoidais
          <code> μ(x) = trap(x; a, b, c, d)</code> — sobe de <code>a</code> a <code>b</code>, platô até
          <code> c</code>, desce até <code>d</code>. Os cruzamentos em μ = 0.5 caem nos limites da SABESP.
          <br /><br />
          <b>② Regras (Mamdani):</b> 45 regras <code>SE aparência é X E pH é Y E turbidez é Z ENTÃO qualidade é W</code>.
          O <b>E</b> é uma t-norma — aqui o <b>mínimo</b>: <code>força = min(μ_aparência, μ_pH, μ_turbidez)</code>.
          Várias regras disparam ao mesmo tempo, cada uma com sua força — é isso que suaviza a resposta.
          <br /><br />
          <b>③ Implicação + agregação:</b> cada termo de saída é <b>recortado</b> na força máxima das regras que
          apontam pra ele (implicação de Mamdani, min) e os recortes são unidos pelo <b>máximo</b> — o envelope agregado.
          <br /><br />
          <b>④ Defuzzificação (centroide):</b> o valor nítido é o centro de massa da área agregada,
          <code> Q = Σ x·μ(x) / Σ μ(x)</code> sobre [0, 1] discretizado. A classe final é o termo de saída com
          maior pertinência em Q.
          <br /><br />
          <b>O exemplo da apostila</b> (cor 15 UH · pH 7 · turbidez 0 UT): cor 15 cai exatamente no cruzamento
          adequada/inadequada (μ = 0.5 nos dois), pH 7 é <i>bom</i> (μ = 1) e turbidez 0 é <i>boa</i> (μ = 1).
          Disparam só 2 regras, ambas com força 0.5 e consequente <i>adequada</i> — o trapézio recortado é
          simétrico em torno de 0.6, então o centroide dá <b>Q = 0.60 → adequada</b>, igualzinho à apostila.
          Clique no preset 📖 e confira cada etapa aí em cima.
        </div>
      </Card>
    </div>
  );
}
