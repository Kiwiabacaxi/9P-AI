import { useMemo, useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Tooltip as LMTooltip, useMap } from 'react-leaflet';
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip,
} from 'recharts';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { TspCidade, TspLegGeometry, TspRouteGeometry } from '../../api/types';

interface Props {
  cidades: TspCidade[];
  tour: number[];                       // ordem atual exibida (best da geração displayada)
  globalTour?: number[];                // melhor global acumulado (em fade)
  routeGeometry?: TspRouteGeometry;     // OSRM: polyline + legs (estradas reais)
  height?: number;
}

// Marcador "truck" (círculo amarelo brilhante com halo rosa) que percorre o tour.
function truckIcon(): L.DivIcon {
  return L.divIcon({
    className: 'tsp-truck-marker',
    html: `<div style="
      width: 18px; height: 18px;
      border-radius: 50%;
      background: #ffff00;
      border: 2px solid #ff00aa;
      box-shadow: 0 0 16px 6px rgba(255, 0, 170, 0.7);
    "></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

// Marker custom estilo "círculo numerado" (não depende dos PNGs do Leaflet —
// que tem problema com imports do Vite — e fica mais limpo visualmente).
function cityIcon(num: number, isStart: boolean): L.DivIcon {
  const bg = isStart ? '#ff00aa' : '#0a0a0a';
  const border = isStart ? '#ff00aa' : '#00ccff';
  const color = isStart ? '#fff' : '#00ccff';
  return L.divIcon({
    className: 'tsp-city-marker',
    html: `<div style="
      width: 24px; height: 24px;
      border-radius: 50%;
      background: ${bg};
      border: 2px solid ${border};
      color: ${color};
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: bold;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 8px rgba(0, 204, 255, 0.4);
    ">${num}</div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

// Componente interno que ajusta os bounds do mapa às cidades.
function FitBounds({ cidades }: { cidades: TspCidade[] }) {
  const map = useMap();
  const lastKey = useRef('');
  useEffect(() => {
    if (cidades.length === 0) return;
    const key = cidades.map(c => `${c.lat},${c.lng}`).join('|');
    if (key === lastKey.current) return;
    lastKey.current = key;
    const bounds = L.latLngBounds(cidades.map(c => [c.lat, c.lng]));
    map.fitBounds(bounds, { padding: [40, 40] });
  }, [cidades, map]);
  return null;
}

// Helper: converte tour (lista de IDs) em sequência de [lat, lng] fechada (volta ao 1º).
function tourToLatLngs(tour: number[], cidades: TspCidade[]): [number, number][] {
  if (tour.length === 0) return [];
  const cidadeById = new Map(cidades.map(c => [c.id, c]));
  const pts: [number, number][] = [];
  for (const id of tour) {
    const c = cidadeById.get(id);
    if (c) pts.push([c.lat, c.lng]);
  }
  // fecha o tour
  if (pts.length > 0) pts.push(pts[0]);
  return pts;
}

export default function TspMap({ cidades, tour, globalTour, routeGeometry, height = 480 }: Props) {
  const tourLatLngs = useMemo(() => tourToLatLngs(tour, cidades), [tour, cidades]);
  const globalLatLngs = useMemo(
    () => (globalTour && globalTour.length > 0 ? tourToLatLngs(globalTour, cidades) : []),
    [globalTour, cidades],
  );

  // Centro inicial — Brasília (mais ou menos centro do BR). FitBounds depois corrige.
  const center: [number, number] = useMemo(() => {
    if (cidades.length === 0) return [-15.0, -55.0];
    const sumLat = cidades.reduce((acc, c) => acc + c.lat, 0);
    const sumLng = cidades.reduce((acc, c) => acc + c.lng, 0);
    return [sumLat / cidades.length, sumLng / cidades.length];
  }, [cidades]);

  // ===== Animação do "caminhão" percorrendo a rota GANHADORA =====
  // Por design, a animação roda sempre no melhor global (globalTour) — não na
  // rota exibida pelo slider. Faz mais sentido pedagogicamente: o usuário quer
  // ver o melhor tour sendo percorrido, não tours intermediários ruidosos.
  //
  // Unificamos a animação em torno de "legs" (uma perna = ida de uma cidade
  // pra próxima). No modo OSRM, cada leg traz sua polyline curvada pelas
  // estradas reais. Nos modos Haversine/Euclidiana, sintetizamos legs com
  // 2 pontos cada (cidade A → cidade B em linha reta). Em ambos os casos,
  // o truck percorre as polylines dos legs em sequência — segue as curvas
  // quando estão lá.
  const animatedTour = useMemo(() => globalTour ?? [], [globalTour]);
  const legs = useMemo<TspLegGeometry[]>(() => {
    if (routeGeometry?.legs && routeGeometry.legs.length > 0) {
      return routeGeometry.legs;
    }
    if (animatedTour.length < 2) return [];
    const cidadeById = new Map(cidades.map(c => [c.id, c]));
    const out: TspLegGeometry[] = [];
    for (let i = 0; i < animatedTour.length; i++) {
      const aId = animatedTour[i];
      const bId = animatedTour[(i + 1) % animatedTour.length];
      const a = cidadeById.get(aId);
      const b = cidadeById.get(bId);
      if (!a || !b) continue;
      out.push({
        polyline: [[a.lat, a.lng], [b.lat, b.lng]],
        distancia: 0,
        duracao: 0,
        deId: aId,
        paraId: bId,
      });
    }
    return out;
  }, [routeGeometry, animatedTour, cidades]);

  // Polyline completa do tour (concatenação dos legs com remoção de pontos
  // duplicados nas junções). Usada como "pista" desenhada no mapa.
  const animatedFullLine = useMemo<[number, number][]>(() => {
    if (legs.length === 0) return [];
    const out: [number, number][] = [];
    legs.forEach((leg, i) => {
      const startIdx = i === 0 ? 0 : 1;
      for (let k = startIdx; k < leg.polyline.length; k++) {
        out.push(leg.polyline[k]);
      }
    });
    return out;
  }, [legs]);

  const segCount = legs.length;

  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);                 // posição contínua, em "legs"
  const [speed, setSpeed] = useState(2);          // legs por segundo
  const [loop, setLoop] = useState(true);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // reset ao trocar a rota ganhadora (não ao trocar o tour exibido pelo slider)
  useEffect(() => {
    setPlaying(false);
    setT(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [globalTour?.join(','), cidades, routeGeometry?.distancia]);

  // RAF loop (usa lambda fresca a cada frame via refs/state)
  useEffect(() => {
    if (!playing || segCount === 0) return;
    lastTimeRef.current = performance.now();
    const tick = (now: number) => {
      const dt = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;
      setT(prev => {
        if (segCount <= 0) return 0;
        const next = prev + dt * speed;
        if (next >= segCount) {
          if (loop) return next % segCount;
          setPlaying(false);
          return segCount;
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, speed, loop, segCount]);

  // Interpola um ponto dentro da polyline de um leg, em fração [0, 1].
  // Distribui uniforme por índice de pontos (não por distância) — visual fica
  // bom na prática e é O(1) por frame.
  function pointInLeg(leg: TspLegGeometry, frac: number): [number, number] | null {
    const pts = leg.polyline;
    if (pts.length === 0) return null;
    if (pts.length === 1) return pts[0];
    const f = Math.max(0, Math.min(1, frac));
    const idx = f * (pts.length - 1);
    const i = Math.floor(idx);
    const localFrac = idx - i;
    if (i >= pts.length - 1) return pts[pts.length - 1];
    const a = pts[i];
    const b = pts[i + 1];
    return [a[0] + (b[0] - a[0]) * localFrac, a[1] + (b[1] - a[1]) * localFrac];
  }

  // Posição interpolada do truck — segue a polyline curvada do leg atual.
  const truckPos = useMemo<[number, number] | null>(() => {
    if (legs.length === 0 || segCount === 0) return null;
    const ti = Math.min(Math.max(t, 0), segCount);
    const legIdx = Math.max(0, Math.min(Math.floor(ti), legs.length - 1));
    const leg = legs[legIdx];
    if (!leg) return null;
    const legFrac = ti - legIdx;
    return pointInLeg(leg, legFrac);
  }, [t, legs, segCount]);

  // Polyline já percorrida = legs completos + parcial do leg atual.
  const drawnLine = useMemo<[number, number][]>(() => {
    if (legs.length === 0 || segCount === 0 || t <= 0) return [];
    const out: [number, number][] = [];
    const ti = Math.min(Math.max(t, 0), segCount);
    const legIdx = Math.max(0, Math.min(Math.floor(ti), legs.length - 1));
    const legFrac = ti - legIdx;

    // legs já completos
    for (let i = 0; i < legIdx; i++) {
      const leg = legs[i];
      if (!leg) continue;
      const pts = leg.polyline;
      if (!pts || pts.length === 0) continue;
      const startIdx = out.length === 0 ? 0 : 1; // pula duplicata da junção
      for (let k = startIdx; k < pts.length; k++) {
        out.push(pts[k]);
      }
    }

    // parcial do leg atual
    const curLeg = legs[legIdx];
    const cur = curLeg?.polyline;
    if (cur && cur.length >= 2) {
      const startIdx = out.length === 0 ? 0 : 1;
      const upTo = legFrac * (cur.length - 1);
      const intIdx = Math.floor(upTo);
      const localFrac = upTo - intIdx;

      for (let i = startIdx; i <= Math.min(intIdx, cur.length - 1); i++) {
        out.push(cur[i]);
      }
      if (intIdx < cur.length - 1 && localFrac > 0) {
        const a = cur[intIdx];
        const b = cur[intIdx + 1];
        out.push([a[0] + (b[0] - a[0]) * localFrac, a[1] + (b[1] - a[1]) * localFrac]);
      }
    }
    return out;
  }, [t, legs, segCount]);

  const showingPlayback = playing || t > 0;
  const currentSegIdx = Math.min(Math.floor(t), segCount);

  return (
    <div>
      <div style={{ height, width: '100%', borderRadius: 6, overflow: 'hidden' }}>
        <MapContainer
          center={center}
          zoom={4}
          style={{ height: '100%', width: '100%', background: '#0a0a0a' }}
          attributionControl={false}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
            maxZoom={18}
          />
          <FitBounds cidades={cidades} />

          {/*
            Quando há `routeGeometry` (modo OSRM com tour resolvido), a polyline
            principal é a versão curvada pelas estradas. Caso contrário (modos
            Haversine/Euclidiana), as polylines são linhas retas city-to-city.
            Em ambos os casos, durante playback, escurecemos a pista pra deixar
            o trecho percorrido (drawnLine) sobressair.
          */}
          {routeGeometry && routeGeometry.polyline.length > 1 && (
            <Polyline
              positions={routeGeometry.polyline}
              pathOptions={{
                color: '#ff00aa',
                weight: 3,
                opacity: showingPlayback ? 0.3 : 0.95,
              }}
            />
          )}

          {/* === Modo "linha reta" (Haversine/Euclidiana — sem OSRM) === */}
          {!routeGeometry && !showingPlayback && globalLatLngs.length > 0 && (
            <Polyline
              positions={globalLatLngs}
              pathOptions={{
                color: '#ffff00',
                weight: 2,
                opacity: 0.35,
                dashArray: '4 6',
              }}
            />
          )}

          {!routeGeometry && !showingPlayback && tourLatLngs.length > 0 && (
            <Polyline
              positions={tourLatLngs}
              pathOptions={{
                color: '#ff00aa',
                weight: 3,
                opacity: 0.9,
              }}
            />
          )}

          {/* Pista do animatedFullLine (= legs concatenados) durante playback —
              usada quando NÃO temos OSRM polyline (caso contrário, ela é a pista). */}
          {!routeGeometry && showingPlayback && animatedFullLine.length > 1 && (
            <Polyline
              positions={animatedFullLine}
              pathOptions={{
                color: '#ff00aa',
                weight: 3,
                opacity: 0.25,
              }}
            />
          )}

          {/* Polyline percorrida (durante playback) */}
          {drawnLine.length > 1 && (
            <Polyline
              positions={drawnLine}
              pathOptions={{
                color: '#ff00aa',
                weight: 4,
                opacity: 1,
              }}
            />
          )}

          {/* Truck marker */}
          {truckPos && showingPlayback && (
            <Marker position={truckPos} icon={truckIcon()} interactive={false} />
          )}

          {/* Markers numerados — número = ordem no tour atual; isStart se for o c0 */}
          {cidades.map(c => {
            const orderInTour = tour.indexOf(c.id);
            const num = orderInTour >= 0 ? orderInTour + 1 : c.id + 1;
            const isStart = tour.length > 0 && tour[0] === c.id;
            return (
              <Marker
                key={c.id}
                position={[c.lat, c.lng]}
                icon={cityIcon(num, isStart)}
              >
                <LMTooltip
                  direction="top"
                  offset={[0, -14]}
                  opacity={0.95}
                  className="tsp-tooltip"
                >
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11 }}>
                    <b>{c.nome}</b>{c.uf ? ` / ${c.uf}` : ''}
                    {orderInTour >= 0 && <div style={{ color: '#888' }}>posição no tour: {orderInTour + 1}</div>}
                  </div>
                </LMTooltip>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      {/* Controles de playback — só aparecem quando já há rota ganhadora */}
      {segCount >= 2 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '8px 12px', marginTop: 8,
          background: 'var(--surface-2)', borderRadius: 6,
          fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--muted)',
          flexWrap: 'wrap',
        }}>
          <button
            className="btn btn-ghost"
            style={{ fontSize: 11, padding: '4px 12px', minWidth: 70 }}
            onClick={() => {
              if (!playing && t >= segCount) setT(0);
              setPlaying(p => !p);
            }}
            title={playing ? 'pausar animação' : 'animar tour'}
          >
            {playing ? '⏸ pause' : '▶ play'}
          </button>

          <button
            className="btn btn-ghost"
            style={{ fontSize: 11, padding: '4px 10px' }}
            onClick={() => { setT(0); }}
            title="voltar ao início"
          >
            ⏮
          </button>

          <span>velocidade:</span>
          {[0.5, 1, 2, 4, 8].map(s => (
            <button
              key={s}
              className="btn btn-ghost"
              style={{
                fontSize: 11, padding: '4px 8px',
                color: s === speed ? 'var(--cyan)' : 'var(--muted)',
                fontWeight: s === speed ? 700 : 400,
              }}
              onClick={() => setSpeed(s)}
            >
              {s}x
            </button>
          ))}

          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={loop}
              onChange={e => setLoop(e.target.checked)}
              style={{ accentColor: 'var(--cyan)' }}
            />
            loop
          </label>

          <div style={{ flex: 1, minWidth: 80 }} />

          {showingPlayback && animatedTour.length > 0 && currentSegIdx < animatedTour.length && (
            <span>
              passo <span style={{ color: 'var(--cyan)' }}>{currentSegIdx + 1}</span> /{' '}
              {animatedTour.length}
              {' — '}
              <span style={{ color: 'var(--pink)' }}>
                {cidades.find(c => c.id === animatedTour[Math.min(currentSegIdx, animatedTour.length - 1)])?.nome ?? ''}
              </span>
              {' → '}
              <span style={{ color: 'var(--cyan)' }}>
                {cidades.find(c => c.id === animatedTour[(currentSegIdx + 1) % animatedTour.length])?.nome ?? ''}
              </span>
            </span>
          )}
          {!showingPlayback && (
            <span style={{ fontSize: 10 }}>
              {segCount >= 2
                ? '▶ play anima a rota ganhadora (melhor global encontrado)'
                : 'rode OTIMIZAR pra liberar a animação'}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Evolução do tour ao longo das gerações (melhor / médio / melhor acumulado).
// =============================================================================

interface EvoProps {
  histMelhor: number[];
  histMedia: number[];
  unidade?: string; // "km" ou "graus"
  height?: number;
}

export function TspEvoChart({ histMelhor, histMedia, unidade = 'km', height = 220 }: EvoProps) {
  const data = useMemo(() => {
    const N = Math.max(histMelhor.length, histMedia.length);
    if (N === 0) return [];
    let bestSoFar = Infinity;
    type Row = {
      gen: number;
      melhor: number | null;
      media: number | null;
      bestSoFar: number | null;
    };
    const raw: Row[] = new Array(N);
    for (let i = 0; i < N; i++) {
      const m = histMelhor[i];
      if (m !== undefined) bestSoFar = Math.min(bestSoFar, m);
      raw[i] = {
        gen: i + 1,
        melhor: m ?? null,
        media: histMedia[i] ?? null,
        bestSoFar: Number.isFinite(bestSoFar) ? bestSoFar : null,
      };
    }
    const maxPoints = 250;
    if (N <= maxPoints) return raw;
    const out: Row[] = new Array(maxPoints);
    const step = (N - 1) / (maxPoints - 1);
    for (let i = 0; i < maxPoints; i++) {
      out[i] = raw[Math.round(i * step)];
    }
    return out;
  }, [histMelhor, histMedia]);

  if (data.length === 0) return <div className="chart-wrap" style={{ height }} />;
  const xInterval = Math.max(0, Math.floor(data.length / 8) - 1);

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="gen"
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            interval={xInterval}
          />
          <YAxis
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              background: '#111',
              border: '1px solid #333',
              fontSize: 11,
              fontFamily: 'JetBrains Mono',
            }}
            labelFormatter={(v) => `geracao ${v}`}
            formatter={(v) => `${Number(v).toFixed(1)} ${unidade}`}
          />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
          <Line
            name="media"
            type="monotone"
            dataKey="media"
            stroke="#00ccff"
            strokeWidth={1}
            strokeOpacity={0.6}
            strokeDasharray="3 3"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            name="melhor da geracao"
            type="monotone"
            dataKey="melhor"
            stroke="#ff00aa"
            strokeWidth={1}
            strokeOpacity={0.45}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            name="melhor acumulado"
            type="monotone"
            dataKey="bestSoFar"
            stroke="#ffff00"
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
