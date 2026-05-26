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
  tour: number[];
  color: string;
  activeIdx?: number; // gene destacado (-1 = nenhum)
}

export function GeneStrip({ cidades, tour, color, activeIdx = -1 }: GeneStripProps) {
  const byId = useMemo(() => new Map(cidades.map(c => [c.id, c])), [cidades]);
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
      {tour.map((id, i) => {
        const isDepot = id === 0;
        const active = i === activeIdx;
        const c = byId.get(id);
        return (
          <div
            key={i}
            title={c ? `${i + 1}º · ${c.nome}` : String(id)}
            style={{
              minWidth: 20, height: 20, padding: '0 3px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: 3, fontFamily: 'JetBrains Mono', fontSize: 9, fontWeight: 700,
              background: active ? color : 'var(--surface-2)',
              color: active ? '#0a0a0a' : (isDepot ? '#ff00aa' : color),
              border: `1px solid ${isDepot ? '#ff00aa' : active ? color : '#222'}`,
              boxShadow: active ? `0 0 8px ${color}` : 'none',
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

// ChromosomeFollower — fita de genes com um "playhead" que percorre o tour na
// ordem (animação seguindo o melhor), mostrando cidade atual → próxima.
export function ChromosomeFollower({ cidades, tour, color }: { cidades: TspCidade[]; tour: number[]; color: string }) {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    setIdx(0);
    if (tour.length === 0) return;
    const t = setInterval(() => setIdx(p => (p + 1) % tour.length), 320);
    return () => clearInterval(t);
  }, [tour]);

  const byId = useMemo(() => new Map(cidades.map(c => [c.id, c])), [cidades]);
  if (tour.length === 0) return null;
  const atual = byId.get(tour[idx]);
  const prox = byId.get(tour[(idx + 1) % tour.length]);

  return (
    <div>
      <GeneStrip cidades={cidades} tour={tour} color={color} activeIdx={idx} />
      <div style={{ marginTop: 8, fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
        passo <span style={{ color }}>{idx + 1}</span>/{tour.length}:{' '}
        <b style={{ color }}>{atual?.nome ?? '—'}</b>
        <span style={{ color: '#444' }}> → </span>
        {prox?.nome ?? '—'}
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
