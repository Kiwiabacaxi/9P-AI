import { useMemo, useState, useEffect, type CSSProperties } from 'react';
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

export type GeneOrigin = 'p1' | 'p2' | 'mut';

interface GeneStripProps {
  cidades: TspCidade[];
  tour: (number | null)[];     // null = gene ainda vazio (no laboratório)
  color: string;
  activeIdx?: number;          // gene único destacado (playhead)
  highlight?: Set<number>;     // conjunto de posições destacadas (segmento/mutação)
  highlightColor?: string;     // cor do destaque do conjunto (default = color)
  // Proveniência por gene (laboratório de operadores): pinta o fundo conforme a
  // origem — Pai 1 / Pai 2 / mutação — pra o usuário ver "de onde veio cada gene".
  provenance?: (GeneOrigin | null)[];
  p1Color?: string;
  p2Color?: string;
  mutColor?: string;
  // Diff vs referência: genes iguais à `compareTo` no mesmo índice ficam atenuados;
  // os diferentes ficam fortes — torna óbvio onde os cromossomos divergem.
  compareTo?: number[];
  dimMatching?: boolean;
}

export function GeneStrip({
  cidades, tour, color, activeIdx = -1, highlight, highlightColor,
  provenance, p1Color = '#00ccff', p2Color = '#ff00aa', mutColor = '#ff8a3d',
  compareTo, dimMatching,
}: GeneStripProps) {
  const byId = useMemo(() => new Map(cidades.map(c => [c.id, c])), [cidades]);
  const hc = highlightColor ?? color;
  const provColor = (p: GeneOrigin | null | undefined): string | null => {
    if (p === 'p1') return p1Color;
    if (p === 'p2') return p2Color;
    if (p === 'mut') return mutColor;
    return null;
  };
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
        const pc = provenance ? provColor(provenance[i]) : null;
        // prioridade: playhead > highlight > proveniência > default
        let bg = 'var(--surface-2)';
        let active = false;
        if (playhead) { bg = color; active = true; }
        else if (inSet) { bg = hc; active = true; }
        else if (pc) { bg = pc; active = true; }
        const matchesRef = !!(compareTo && compareTo[i] !== undefined && compareTo[i] === id);
        const opacity = dimMatching && matchesRef && !active ? 0.18 : 1;
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
              opacity,
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

// Botõezinhos pequenos (velocidade etc).
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

// PlayerControls — player limpo: ◀ (anterior) · botão central play/pause · ▶ (próximo).
function PlayerControls({ playing, onPlayPause, onPrev, onNext, prevTitle, nextTitle }: {
  playing: boolean; onPlayPause: () => void; onPrev: () => void; onNext: () => void;
  prevTitle?: string; nextTitle?: string;
}) {
  const side: CSSProperties = {
    width: 38, height: 34, borderRadius: 6, cursor: 'pointer', fontSize: 14,
    background: 'var(--surface-2)', color: 'var(--cyan)', border: '1px solid #2a2a2a',
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0,
  };
  const center: CSSProperties = {
    width: 48, height: 34, borderRadius: 6, cursor: 'pointer', fontSize: 16,
    background: 'var(--cyan)', color: '#06121a', border: 'none', fontWeight: 700,
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 0,
    boxShadow: '0 0 10px rgba(0,204,255,0.35)',
  };
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
      <button style={side} onClick={onPrev} title={prevTitle} aria-label="anterior">⏮</button>
      <button style={center} onClick={onPlayPause} title={playing ? 'pausar' : 'tocar'} aria-label="play/pause">
        {playing ? '⏸' : '▶'}
      </button>
      <button style={side} onClick={onNext} title={nextTitle} aria-label="próximo">⏭</button>
    </div>
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
      <div style={{ marginTop: 12 }}>
        <PlayerControls
          playing={playing}
          onPlayPause={() => setPlaying(p => !p)}
          onPrev={() => stepTo(idx - 1)}
          onNext={() => stepTo(idx + 1)}
          prevTitle="gene anterior"
          nextTitle="próximo gene"
        />
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 3 }}>
            {[1, 3, 6].map(s => ctrlBtn(`${s}/s`, () => setSpeed(s), s === speed, `${s} genes por segundo`))}
          </div>
          <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
            passo <span style={{ color }}>{idx + 1}</span>/{n}:{' '}
            <b style={{ color }}>{atual?.nome ?? '—'}</b>
            <span style={{ color: '#444' }}> → </span>
            {prox?.nome ?? '—'}
          </div>
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
  highlight: Set<number>;                   // posições destacadas no filho (ênfase)
  highlightColor: string;
  provenance: (GeneOrigin | null)[];        // origem por gene no filho — colore o fundo
  p1Seg?: Set<number>;                      // segmento destacado no pai 1
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
): { frames: MatingFrame[]; c1: number; c2: number } {
  const n = p1.length;

  // Corte GRANDE: segmento de 30 a 60% do cromossomo, posicionado aleatoriamente.
  // Sem isso o operador faz quase nada visível (segmento de 2-3 genes de 50 = invisível).
  const segMin = Math.max(2, Math.floor(n * 0.3));
  const segMax = Math.max(segMin + 1, Math.floor(n * 0.6));
  const segLen = segMin + Math.floor(Math.random() * (segMax - segMin + 1));
  const c1 = Math.floor(Math.random() * (n - segLen + 1));
  const c2 = c1 + segLen; // exclusivo

  const seg = new Set<number>();
  for (let i = c1; i < c2; i++) seg.add(i);

  const crossed = cruzamento === 'pmx' ? pmxChild(p1, p2, c1, c2) : oxChild(p1, p2, c1, c2);

  // Proveniência base do cruzamento: posições do segmento = vieram do Pai 1;
  // demais posições = vieram do Pai 2 (preenchidas pelo OX/PMX a partir de P2).
  const baseProv: (GeneOrigin | null)[] = new Array(n).fill(null);
  for (let i = 0; i < n; i++) baseProv[i] = seg.has(i) ? 'p1' : 'p2';

  // mutação
  const mutated = crossed.slice();
  const mutHL = new Set<number>();
  const ocorreuMut = Math.random() < probMut;
  let mutCaption = `Mutação não ocorreu (sorteio ≥ Pm=${probMut.toFixed(2)}). O filho vai pra próxima geração como está.`;
  if (ocorreuMut) {
    if (mutacao === 'swap') {
      const i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      if (j === i) j = (j + 1) % n;
      [mutated[i], mutated[j]] = [mutated[j], mutated[i]];
      mutHL.add(i); mutHL.add(j);
      mutCaption = `Mutação SWAP: troca os genes das posições ${i + 1} e ${j + 1}.`;
    } else {
      let i = Math.floor(Math.random() * n);
      let j = Math.floor(Math.random() * n);
      if (i > j) [i, j] = [j, i];
      if (i === j) j = Math.min(n - 1, j + 1);
      for (let x = i, y = j; x < y; x++, y--) [mutated[x], mutated[y]] = [mutated[y], mutated[x]];
      for (let x = i; x <= j; x++) mutHL.add(x);
      mutCaption = `Mutação INVERSÃO: reverte o trecho [${i + 1}, ${j + 1}] do filho (movimento 2-opt).`;
    }
  }

  // Proveniência pós-mutação: posições mutadas viram 'mut'; demais mantêm.
  const mutProv = baseProv.slice();
  mutHL.forEach(i => { mutProv[i] = 'mut'; });

  // Frame 2 (segmento copiado): só o segmento aparece no filho, marcado como 'p1'.
  const segChild: (number | null)[] = new Array(n).fill(null);
  const segProv: (GeneOrigin | null)[] = new Array(n).fill(null);
  for (let i = c1; i < c2; i++) { segChild[i] = crossed[i]; segProv[i] = 'p1'; }

  const segLabel = `[${c1 + 1}, ${c2}] (${segLen} genes = ${Math.round(100 * segLen / n)}% do cromossomo)`;

  const frames: MatingFrame[] = [
    {
      fase: 'selecao',
      caption: 'Seleção: 2 pais escolhidos da população. No torneio, sorteamos k indivíduos e ficamos com os melhores; na roleta, a chance é proporcional à aptidão. Os melhores tendem a se reproduzir mais.',
      child: new Array(n).fill(null),
      highlight: new Set(),
      highlightColor: '#ffff00',
      provenance: new Array(n).fill(null),
    },
    {
      fase: 'cruzamento',
      caption: `Cruzamento ${cruzamento.toUpperCase()} — passo 1: copia o segmento ${segLabel} do Pai 1 pro filho (as células azuis = vieram do Pai 1).`,
      child: segChild,
      highlight: seg,
      highlightColor: '#00ccff',
      provenance: segProv,
      p1Seg: seg,
    },
    {
      fase: 'cruzamento',
      caption: cruzamento === 'ox'
        ? 'Cruzamento OX — passo 2: preenche o resto com os genes do Pai 2 na ordem em que aparecem, pulando os que já estão no filho. (Células rosa = vieram do Pai 2.)'
        : 'Cruzamento PMX — passo 2: para cada gene do Pai 2 fora do segmento, segue a "cadeia de mapeamento" do segmento e coloca no lugar correto; o que sobra vem direto do Pai 2. (Células rosa = origem Pai 2.)',
      child: crossed,
      highlight: new Set(),
      highlightColor: '#ff00aa',
      provenance: baseProv,
      p1Seg: seg,
    },
    {
      fase: 'mutacao',
      caption: mutCaption,
      child: mutated,
      highlight: mutHL,
      highlightColor: '#ff8a3d',
      provenance: mutProv,
    },
    {
      fase: 'fim',
      caption: 'Filho pronto — entra na próxima geração (e pode virar pai depois). Repare na composição: parte azul (Pai 1) + parte rosa (Pai 2)' + (ocorreuMut ? ' + cantos laranja (mutação).' : ' (sem mutação desta vez).'),
      child: mutated,
      highlight: new Set(),
      highlightColor: '#7cf67c',
      provenance: mutProv,
    },
  ];
  return { frames, c1, c2 };
}

interface Pai { tour: number[]; label: string; color: string }

interface OperatorLabProps {
  cidades: TspCidade[];
  pais: Pai[]; // candidatos a pai (todas as ilhas, ordenadas por aptidão)
  cruzamento: 'ox' | 'pmx';
  mutacao: 'swap' | 'inversao';
  probMut: number;
}

// pickParents — garante DOIS pais distintos. Pega o melhor (pais[0]) e o primeiro
// com tour diferente; se todas as ilhas convergiram pro mesmo tour, deriva o 2º
// pai perturbando o 1º (alguns swaps) pra o cruzamento ter o que fazer.
function pickParents(pais: Pai[]): { p1: Pai; p2: Pai } {
  const p1 = pais[0];
  const sig = (t: number[]) => t.join(',');
  const diff = pais.slice(1).find(p => sig(p.tour) !== sig(p1.tour));
  if (diff) return { p1, p2: diff };
  // todas iguais → perturba uma cópia do melhor
  const t = p1.tour.slice();
  const n = t.length;
  const swaps = Math.max(3, Math.floor(n / 8));
  for (let s = 0; s < swaps; s++) {
    const i = Math.floor(Math.random() * n);
    const j = Math.floor(Math.random() * n);
    [t[i], t[j]] = [t[j], t[i]];
  }
  return { p1, p2: { tour: t, label: 'variação (ilhas convergiram → 2º pai perturbado)', color: '#ff00aa' } };
}

export function OperatorLab({ cidades, pais, cruzamento, mutacao, probMut }: OperatorLabProps) {
  const [seqKey, setSeqKey] = useState(0); // muda → gera nova reprodução

  const paisSig = pais.map(p => p.tour.join(',')).join('#');
  const { p1, p2 } = useMemo(
    () => pickParents(pais),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paisSig, seqKey],
  );
  const { frames } = useMemo(
    () => buildMating(p1.tour, p2.tour, cruzamento, mutacao, probMut),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [p1, p2, cruzamento, mutacao, probMut, seqKey],
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

  // Cores DIDÁTICAS fixas: Pai 1 = ciano, Pai 2 = rosa, mutação = laranja.
  // O filho herda essas mesmas cores conforme a proveniência de cada gene, então
  // a "stripe" azul no filho mapeia direto pro Pai 1 e a região rosa pro Pai 2.
  const P1C = '#00ccff', P2C = '#ff00aa', MUTC = '#ff8a3d';

  // Cor do swatch da legenda
  const Swatch = ({ c }: { c: string }) => (
    <span style={{
      display: 'inline-block', width: 12, height: 12, borderRadius: 3,
      background: c, marginRight: 4, verticalAlign: 'middle',
    }} />
  );

  return (
    <div>
      {/* Legenda de cores — explica a proveniência ANTES das fitas */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center', justifyContent: 'center',
        marginBottom: 12, padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6,
        fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)',
      }}>
        <span><Swatch c={P1C} />gene veio do <b style={{ color: P1C }}>Pai 1</b></span>
        <span><Swatch c={P2C} />gene veio do <b style={{ color: P2C }}>Pai 2</b></span>
        <span><Swatch c={MUTC} />gene <b style={{ color: MUTC }}>mutado</b></span>
        <span><Swatch c="#0a0a0a" />gene vazio (ainda não preenchido)</span>
      </div>

      {/* pais */}
      <div style={{ marginBottom: 4, fontSize: 11, fontFamily: 'JetBrains Mono', color: P1C }}>
        Pai 1 — {p1.label}
      </div>
      <GeneStrip cidades={cidades} tour={p1.tour} color={P1C} highlight={frame.p1Seg} highlightColor={P1C} />
      <div style={{ margin: '8px 0 4px', fontSize: 11, fontFamily: 'JetBrains Mono', color: P2C }}>
        Pai 2 — {p2.label}
      </div>
      <GeneStrip cidades={cidades} tour={p2.tour} color={P2C} />

      {/* filho */}
      <div style={{ margin: '12px 0 4px', fontSize: 11, fontFamily: 'JetBrains Mono', color: '#ffff00' }}>
        Filho (em construção)
      </div>
      <GeneStrip
        cidades={cidades}
        tour={frame.child}
        color="#ffff00"
        provenance={frame.provenance}
        p1Color={P1C}
        p2Color={P2C}
        mutColor={MUTC}
      />

      {/* caption da fase */}
      <div style={{
        marginTop: 10, padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6,
        fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--on-surface)', lineHeight: 1.5,
      }}>
        <span style={{ color: 'var(--cyan)' }}>[{faseLabel[frame.fase]}]</span> {frame.caption}
      </div>

      {/* controles */}
      <div style={{ marginTop: 12 }}>
        <PlayerControls
          playing={playing}
          onPlayPause={() => { if (fi >= frames.length - 1) setFi(0); setPlaying(p => !p); }}
          onPrev={() => stepTo(fi - 1)}
          onNext={() => stepTo(fi + 1)}
          prevTitle="etapa anterior"
          nextTitle="próxima etapa"
        />
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
            etapa {fi + 1}/{frames.length}
          </span>
          {ctrlBtn('🧬 gerar novo filho', () => setSeqKey(k => k + 1), false, 'sortear novos pais/cortes/mutação')}
        </div>
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
