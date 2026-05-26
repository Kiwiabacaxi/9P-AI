import { useMemo, useState, useEffect } from 'react';
import type { TspCidade, TspMigracao } from '../../api/types';

// Paleta de cores por ilha — neon, distinta, casa com o tema do app.
// Usada de ponta a ponta (mini-rota, linha do gráfico, nó do anel, tabela).
export const ISLAND_COLORS = [
  '#00ccff', // ciano
  '#ff00aa', // rosa
  '#ffd400', // amarelo
  '#7cf67c', // verde
  '#b388ff', // roxo
  '#ff8a3d', // laranja
];

export function islandColor(i: number): string {
  return ISLAND_COLORS[i % ISLAND_COLORS.length];
}

// =============================================================================
// MiniRoute — desenha a rota de UMA ilha num SVG pequeno (small multiple).
// Projeta lat/lng das cidades num viewBox com padding; tour fechado + pontos.
// =============================================================================

interface MiniRouteProps {
  cidades: TspCidade[];
  tour: number[];
  color: string;
  size?: number;
  highlightTour?: number[]; // gene migrante recém-chegado (destaca em outra cor)
  highlightColor?: string;
}

export function MiniRoute({
  cidades, tour, color, size = 150, highlightTour, highlightColor,
}: MiniRouteProps) {
  const proj = useMemo(() => {
    if (cidades.length === 0) return null;
    const lats = cidades.map(c => c.lat);
    const lngs = cidades.map(c => c.lng);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
    const pad = 12;
    const w = size, h = size;
    const spanLat = maxLat - minLat || 1;
    const spanLng = maxLng - minLng || 1;
    const byId = new Map(cidades.map(c => [c.id, c]));
    // y invertido (lat maior = mais ao norte = mais acima)
    const project = (id: number): [number, number] | null => {
      const c = byId.get(id);
      if (!c) return null;
      const x = pad + ((c.lng - minLng) / spanLng) * (w - 2 * pad);
      const y = pad + ((maxLat - c.lat) / spanLat) * (h - 2 * pad);
      return [x, y];
    };
    return { project };
  }, [cidades, size]);

  const path = useMemo(() => {
    if (!proj || tour.length === 0) return '';
    const pts = tour.map(id => proj.project(id)).filter(Boolean) as [number, number][];
    if (pts.length === 0) return '';
    const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    return d + ' Z';
  }, [proj, tour]);

  const highlightPath = useMemo(() => {
    if (!proj || !highlightTour || highlightTour.length === 0) return '';
    const pts = highlightTour.map(id => proj.project(id)).filter(Boolean) as [number, number][];
    if (pts.length === 0) return '';
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') + ' Z';
  }, [proj, highlightTour]);

  const dots = useMemo(() => {
    if (!proj) return [];
    return cidades.map(c => {
      const p = proj.project(c.id);
      return p ? { id: c.id, x: p[0], y: p[1], isDepot: c.id === 0 } : null;
    }).filter(Boolean) as { id: number; x: number; y: number; isDepot: boolean }[];
  }, [proj, cidades]);

  return (
    <svg width={size} height={size} style={{ display: 'block', background: '#0a0a0a', borderRadius: 4 }}>
      {highlightPath && (
        <path d={highlightPath} fill="none" stroke={highlightColor ?? '#fff'} strokeWidth={2.5} strokeOpacity={0.9} strokeDasharray="3 3" />
      )}
      {path && <path d={path} fill="none" stroke={color} strokeWidth={1.6} strokeOpacity={0.95} />}
      {dots.map(d => (
        <circle
          key={d.id}
          cx={d.x}
          cy={d.y}
          r={d.isDepot ? 3.2 : 1.8}
          fill={d.isDepot ? '#ff00aa' : color}
          stroke={d.isDepot ? '#fff' : 'none'}
          strokeWidth={d.isDepot ? 1 : 0}
        />
      ))}
    </svg>
  );
}

// =============================================================================
// GeneStrip — desenha um cromossomo (tour = permutação) como uma fita de genes.
// Cada gene = uma cidade (mostra o id); depot (0) em rosa; gene em activeIdx
// destacado na cor da ilha. É o "desenho do cromossomo".
// =============================================================================

interface GeneStripProps {
  cidades: TspCidade[];
  tour: (number | null)[];     // null = gene ainda vazio (no laboratório)
  color: string;
  activeIdx?: number;          // gene único destacado (playhead)
  highlight?: Set<number>;     // conjunto de posições destacadas (segmento/mutação)
  highlightColor?: string;     // cor do destaque do conjunto (default = color)
}

export function GeneStrip({ cidades, tour, color, activeIdx = -1, highlight, highlightColor }: GeneStripProps) {
  const byId = useMemo(() => new Map(cidades.map(c => [c.id, c])), [cidades]);
  const hc = highlightColor ?? color;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
      {tour.map((id, i) => {
        if (id === null || id === undefined) {
          return (
            <div key={i} style={{
              minWidth: 20, height: 20, borderRadius: 3,
              border: '1px dashed #333', background: 'transparent',
            }} />
          );
        }
        const isDepot = id === 0;
        const playhead = i === activeIdx;
        const inSet = highlight?.has(i) ?? false;
        const active = playhead || inSet;
        const bg = playhead ? color : inSet ? hc : 'var(--surface-2)';
        const c = byId.get(id);
        return (
          <div
            key={i}
            title={c ? `${i + 1}º · ${c.nome}` : String(id)}
            style={{
              minWidth: 20, height: 20, padding: '0 3px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: 3, fontFamily: 'JetBrains Mono', fontSize: 9, fontWeight: 700,
              background: bg,
              color: active ? '#0a0a0a' : (isDepot ? '#ff00aa' : color),
              border: `1px solid ${isDepot ? '#ff00aa' : active ? bg : '#222'}`,
              boxShadow: active ? `0 0 8px ${bg}` : 'none',
              transition: 'background 0.15s, box-shadow 0.15s, color 0.15s',
            }}
          >
            {id}
          </div>
        );
      })}
    </div>
  );
}

// Botõezinhos de transporte reutilizados (play/pause/step/speed).
function ctrlBtn(label: string, onClick: () => void, active = false, title?: string) {
  return (
    <button
      className="btn btn-ghost"
      style={{ fontSize: 11, padding: '4px 9px', color: active ? 'var(--cyan)' : undefined, fontWeight: active ? 700 : 400 }}
      onClick={onClick}
      title={title}
    >
      {label}
    </button>
  );
}

// ChromosomeFollower — fita de genes com playhead controlável (play/pause,
// voltar/avançar passo, velocidade) percorrendo o tour na ordem de visita.
export function ChromosomeFollower({ cidades, tour, color }: { cidades: TspCidade[]; tour: number[]; color: string }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(3); // genes por segundo

  useEffect(() => { setIdx(0); }, [tour]);
  useEffect(() => {
    if (!playing || tour.length === 0) return;
    const t = setInterval(() => setIdx(p => (p + 1) % tour.length), 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed, tour]);

  const byId = useMemo(() => new Map(cidades.map(c => [c.id, c])), [cidades]);
  if (tour.length === 0) return null;
  const n = tour.length;
  const atual = byId.get(tour[idx]);
  const prox = byId.get(tour[(idx + 1) % n]);
  const stepTo = (v: number) => { setPlaying(false); setIdx((v + n) % n); };

  return (
    <div>
      <GeneStrip cidades={cidades} tour={tour} color={color} activeIdx={idx} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
        {ctrlBtn('⏮', () => stepTo(0), false, 'reiniciar')}
        {ctrlBtn('◀', () => stepTo(idx - 1), false, 'voltar')}
        {ctrlBtn(playing ? '⏸' : '▶', () => setPlaying(p => !p), playing, playing ? 'pausar' : 'tocar')}
        {ctrlBtn('▶|', () => stepTo(idx + 1), false, 'avançar')}
        <div style={{ display: 'flex', gap: 2, marginLeft: 4 }}>
          {[1, 3, 6].map(s => ctrlBtn(`${s}/s`, () => setSpeed(s), s === speed, `${s} genes por segundo`))}
        </div>
        <div style={{ marginLeft: 8, fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
          passo <span style={{ color }}>{idx + 1}</span>/{n}:{' '}
          <b style={{ color }}>{atual?.nome ?? '—'}</b>
          <span style={{ color: '#444' }}> → </span>
          {prox?.nome ?? '—'}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// OperatorLab — laboratório passo-a-passo dos operadores genéticos: pega 2 pais
// reais (melhores tours de duas ilhas) e anima uma reprodução completa —
// seleção → cruzamento (OX/PMX) → mutação (swap/inversão) — respeitando os
// operadores escolhidos. Quadros navegáveis (voltar/avançar/play).
// =============================================================================

interface MatingFrame {
  fase: 'selecao' | 'cruzamento' | 'mutacao' | 'fim';
  caption: string;
  child: (number | null)[];
  highlight: Set<number>;     // posições destacadas no filho
  highlightColor: string;
  p1Seg?: Set<number>;        // segmento destacado no pai 1
}

function oxChild(p1: number[], p2: number[], c1: number, c2: number): number[] {
  const n = p1.length;
  const child = new Array<number>(n).fill(-1);
  const inChild = new Set<number>();
  for (let i = c1; i < c2; i++) { child[i] = p1[i]; inChild.add(p1[i]); }
  const remaining: number[] = [];
  for (let i = 0; i < n; i++) { const c = p2[(c2 + i) % n]; if (!inChild.has(c)) remaining.push(c); }
  let j = 0;
  for (let i = 0; i < n; i++) { const pos = (c2 + i) % n; if (pos >= c1 && pos < c2) continue; child[pos] = remaining[j++]; }
  return child;
}

function pmxChild(p1: number[], p2: number[], c1: number, c2: number): number[] {
  const n = p1.length;
  const child = new Array<number>(n).fill(-1);
  const inSeg = new Set<number>();
  for (let i = c1; i < c2; i++) { child[i] = p1[i]; inSeg.add(p1[i]); }
  const posInP2 = new Map<number, number>();
  p2.forEach((c, i) => posInP2.set(c, i));
  for (let i = c1; i < c2; i++) {
    const c = p2[i];
    if (inSeg.has(c)) continue;
    let target = p1[i];
    let tp = posInP2.get(target)!;
    while (tp >= c1 && tp < c2) { target = p1[tp]; tp = posInP2.get(target)!; }
    child[tp] = c;
  }
  for (let i = 0; i < n; i++) if (child[i] === -1) child[i] = p2[i];
  return child;
}

function buildMating(
  p1: number[], p2: number[],
  cruzamento: 'ox' | 'pmx', mutacao: 'swap' | 'inversao', probMut: number,
): { frames: MatingFrame[] } {
  const n = p1.length;
  let c1 = Math.floor(Math.random() * n);
  let c2 = Math.floor(Math.random() * n);
  if (c1 > c2) [c1, c2] = [c2, c1];
  if (c1 === c2) c2 = Math.min(n, c2 + Math.max(2, Math.floor(n / 4)));

  const seg = new Set<number>();
  for (let i = c1; i < c2; i++) seg.add(i);

  const crossed = cruzamento === 'pmx' ? pmxChild(p1, p2, c1, c2) : oxChild(p1, p2, c1, c2);
  const fora = new Set<number>();
  for (let i = 0; i < n; i++) if (i < c1 || i >= c2) fora.add(i);

  // mutação
  const mutated = crossed.slice();
  const mutHL = new Set<number>();
  const ocorreuMut = Math.random() < probMut;
  let mutCaption = `Mutação não ocorreu (sorteio ≥ Pm=${probMut.toFixed(2)})`;
  if (ocorreuMut) {
    if (mutacao === 'swap') {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      if (j === i) j = (j + 1) % n;
      [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
      mutHL.add(i); mutHL.add(j);
      mutCaption = `Mutação SWAP: troca os genes das posições ${i + 1} e ${j + 1}`;
    } else {
      let i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      if (i > j) [i, j] = [j, i];
      for (let x = i, y = j; x < y; x++, y--) [mutated[x], mutated[y]] = [mutated[y], mutated[x]];
      for (let x = i; x <= j; x++) mutHL.add(x);
      mutCaption = `Mutação INVERSÃO: reverte o trecho [${i + 1}, ${j + 1}] (movimento 2-opt)`;
    }
  }

  const segChild: (number | null)[] = new Array(n).fill(null);
  for (let i = c1; i < c2; i++) segChild[i] = crossed[i];

  const frames: MatingFrame[] = [
    {
      fase: 'selecao',
      caption: 'Seleção: 2 pais escolhidos da população (os melhores tendem a ser sorteados).',
      child: new Array(n).fill(null),
      highlight: new Set(),
      highlightColor: '#ffff00',
    },
    {
      fase: 'cruzamento',
      caption: `Cruzamento ${cruzamento.toUpperCase()}: copia o segmento [${c1 + 1}, ${c2}] do Pai 1 pro filho.`,
      child: segChild,
      highlight: seg,
      highlightColor: '#00ccff',
      p1Seg: seg,
    },
    {
      fase: 'cruzamento',
      caption: cruzamento === 'ox'
        ? 'Cruzamento OX: preenche o resto com os genes do Pai 2, na ordem, pulando repetidos.'
        : 'Cruzamento PMX: mapeia os genes do Pai 2 seguindo a cadeia do segmento e completa o resto.',
      child: crossed,
      highlight: fora,
      highlightColor: '#ff00aa',
      p1Seg: seg,
    },
    {
      fase: 'mutacao',
      caption: mutCaption,
      child: mutated,
      highlight: mutHL,
      highlightColor: '#ff8a3d',
    },
    {
      fase: 'fim',
      caption: 'Filho pronto — entra na próxima geração (e pode virar pai depois).',
      child: mutated,
      highlight: new Set(),
      highlightColor: '#7cf67c',
    },
  ];
  return { frames };
}

interface OperatorLabProps {
  cidades: TspCidade[];
  pais: { tour: number[]; label: string; color: string }[]; // >=2 candidatos a pai
  cruzamento: 'ox' | 'pmx';
  mutacao: 'swap' | 'inversao';
  probMut: number;
}

export function OperatorLab({ cidades, pais, cruzamento, mutacao, probMut }: OperatorLabProps) {
  const p1 = pais[0];
  const p2 = pais[1] ?? pais[0];

  const [seqKey, setSeqKey] = useState(0); // muda → gera nova reprodução
  const { frames } = useMemo(
    () => buildMating(p1.tour, p2.tour, cruzamento, mutacao, probMut),
    // seqKey força regenerar; tours/operadores também
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [seqKey, p1.tour, p2.tour, cruzamento, mutacao, probMut],
  );

  const [fi, setFi] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setFi(0); }, [frames]);
  useEffect(() => {
    if (!playing) return;
    if (fi >= frames.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setFi(f => Math.min(f + 1, frames.length - 1)), 1400);
    return () => clearTimeout(t);
  }, [playing, fi, frames.length]);

  if (p1.tour.length === 0) return null;
  const frame = frames[fi];
  const stepTo = (v: number) => { setPlaying(false); setFi(Math.max(0, Math.min(v, frames.length - 1))); };

  const faseLabel: Record<MatingFrame['fase'], string> = {
    selecao: '1 · Seleção', cruzamento: '2 · Cruzamento', mutacao: '3 · Mutação', fim: '4 · Filho',
  };

  return (
    <div>
      {/* pais */}
      <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'JetBrains Mono', color: p1.color }}>
        Pai 1 — {p1.label}
      </div>
      <GeneStrip cidades={cidades} tour={p1.tour} color={p1.color} highlight={frame.p1Seg} highlightColor="#00ccff" />
      <div style={{ margin: '8px 0 4px', fontSize: 11, fontFamily: 'JetBrains Mono', color: p2.color }}>
        Pai 2 — {p2.label}
      </div>
      <GeneStrip cidades={cidades} tour={p2.tour} color={p2.color} />

      {/* filho */}
      <div style={{ margin: '12px 0 4px', fontSize: 11, fontFamily: 'JetBrains Mono', color: '#ffff00' }}>
        Filho (em construção)
      </div>
      <GeneStrip cidades={cidades} tour={frame.child} color="#ffff00" highlight={frame.highlight} highlightColor={frame.highlightColor} />

      {/* caption da fase */}
      <div style={{
        marginTop: 10, padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6,
        fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--on-surface)', lineHeight: 1.5,
      }}>
        <span style={{ color: 'var(--cyan)' }}>[{faseLabel[frame.fase]}]</span> {frame.caption}
      </div>

      {/* controles */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
        {ctrlBtn('⏮', () => stepTo(0), false, 'início')}
        {ctrlBtn('◀', () => stepTo(fi - 1), false, 'voltar etapa')}
        {ctrlBtn(playing ? '⏸' : '▶', () => setPlaying(p => !p), playing, playing ? 'pausar' : 'tocar etapas')}
        {ctrlBtn('▶|', () => stepTo(fi + 1), false, 'avançar etapa')}
        <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginLeft: 4 }}>
          etapa {fi + 1}/{frames.length}
        </span>
        <div style={{ flex: 1 }} />
        {ctrlBtn('🧬 gerar novo filho', () => { setSeqKey(k => k + 1); }, false, 'sortear novos cortes/mutação')}
      </div>
    </div>
  );
}

// =============================================================================
// RingDiagram — as ilhas dispostas em círculo, com setas i→(i+1) (a "dança de
// cadeiras"). Na geração de migração, as setas pulsam na cor da origem.
// =============================================================================

interface RingProps {
  numIlhas: number;
  ilhaVencedora: number;
  migracoes?: TspMigracao[]; // setas ativas neste instante
  size?: number;
}

export function RingDiagram({ numIlhas, ilhaVencedora, migracoes, size = 220 }: RingProps) {
  const cx = size / 2, cy = size / 2;
  const R = size / 2 - 34;
  const nodes = useMemo(() => {
    const out: { i: number; x: number; y: number }[] = [];
    for (let i = 0; i < numIlhas; i++) {
      // começa no topo, sentido horário
      const ang = -Math.PI / 2 + (2 * Math.PI * i) / numIlhas;
      out.push({ i, x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) });
    }
    return out;
  }, [numIlhas, cx, cy, R]);

  const migratingFrom = useMemo(() => {
    const s = new Set<number>();
    (migracoes ?? []).forEach(m => s.add(m.de));
    return s;
  }, [migracoes]);

  // arco curvo entre nó i e (i+1)
  function arc(a: { x: number; y: number }, b: { x: number; y: number }) {
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = b.x - a.x, dy = b.y - a.y;
    // empurra o ponto de controle pra fora do centro
    const toCx = mx - cx, toCy = my - cy;
    const len = Math.hypot(toCx, toCy) || 1;
    const k = 0.25 * Math.hypot(dx, dy);
    const px = mx + (toCx / len) * k;
    const py = my + (toCy / len) * k;
    return `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${px.toFixed(1)},${py.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`;
  }

  return (
    <svg width={size} height={size} style={{ display: 'block', margin: '0 auto' }}>
      <defs>
        {nodes.map(node => {
          const c = islandColor(node.i);
          return (
            <marker key={node.i} id={`arrow-${node.i}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={c} />
            </marker>
          );
        })}
      </defs>
      {/* setas do anel */}
      {numIlhas >= 2 && nodes.map(node => {
        const dest = nodes[(node.i + 1) % numIlhas];
        const c = islandColor(node.i);
        const ativa = migratingFrom.has(node.i);
        return (
          <path
            key={`arc-${node.i}`}
            d={arc(node, dest)}
            fill="none"
            stroke={c}
            strokeWidth={ativa ? 3 : 1.2}
            strokeOpacity={ativa ? 1 : 0.28}
            markerEnd={`url(#arrow-${node.i})`}
            style={ativa ? { filter: `drop-shadow(0 0 6px ${c})` } : undefined}
          >
            {ativa && (
              <animate attributeName="stroke-opacity" values="1;0.4;1" dur="0.6s" repeatCount="indefinite" />
            )}
          </path>
        );
      })}
      {/* nós das ilhas */}
      {nodes.map(node => {
        const c = islandColor(node.i);
        const venc = node.i === ilhaVencedora;
        return (
          <g key={`node-${node.i}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={venc ? 18 : 14}
              fill="#0a0a0a"
              stroke={c}
              strokeWidth={venc ? 3 : 2}
              style={venc ? { filter: `drop-shadow(0 0 8px ${c})` } : undefined}
            />
            <text x={node.x} y={node.y + 4} textAnchor="middle" fill={c} fontSize="12" fontFamily="JetBrains Mono" fontWeight="bold">
              {node.i + 1}
            </text>
          </g>
        );
      })}
      {/* tokens migrantes viajando (mostrados durante a migração) */}
      {(migracoes ?? []).map((m, idx) => {
        const a = nodes[m.de], b = nodes[m.para];
        if (!a || !b) return null;
        const c = islandColor(m.de);
        return (
          <circle key={`tok-${idx}`} r="5" fill={c} style={{ filter: `drop-shadow(0 0 6px ${c})` }}>
            <animateMotion dur="0.9s" repeatCount="indefinite" path={arc(a, b)} />
          </circle>
        );
      })}
    </svg>
  );
}
