import { useMemo, useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Tooltip as LMTooltip, useMap } from 'react-leaflet';
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip,
} from 'recharts';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { TspCidade } from '../../api/types';

interface Props {
  cidades: TspCidade[];
  tour: number[];        // ordem atual exibida (best da geração displayada)
  globalTour?: number[]; // melhor global acumulado (em fade)
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

export default function TspMap({ cidades, tour, globalTour, height = 480 }: Props) {
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

  // ===== Animação do "caminhão" percorrendo o tour =====
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);                 // posição contínua no tour, em "segmentos"
  const [speed, setSpeed] = useState(2);          // segmentos por segundo
  const [loop, setLoop] = useState(true);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const segCount = Math.max(0, tourLatLngs.length - 1);

  // reset ao trocar tour
  useEffect(() => {
    setPlaying(false);
    setT(0);
  }, [tour, cidades]);

  // RAF loop (usa lambda fresca a cada frame via refs/state)
  useEffect(() => {
    if (!playing || segCount === 0) return;
    lastTimeRef.current = performance.now();
    const tick = (now: number) => {
      const dt = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;
      setT(prev => {
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

  // Posição interpolada do truck.
  const truckPos = useMemo<[number, number] | null>(() => {
    if (tourLatLngs.length < 2) return null;
    const ti = Math.min(t, segCount);
    const i = Math.min(Math.floor(ti), segCount - 1);
    const frac = ti - i;
    const a = tourLatLngs[i];
    const b = tourLatLngs[i + 1];
    return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac];
  }, [t, tourLatLngs, segCount]);

  // Polyline percorrida até a posição atual (drawn so far).
  const drawnLine = useMemo<[number, number][]>(() => {
    if (tourLatLngs.length < 2 || t <= 0) return [];
    const ti = Math.min(t, segCount);
    const i = Math.min(Math.floor(ti), segCount - 1);
    const frac = ti - i;
    const out = tourLatLngs.slice(0, i + 1);
    if (frac > 0) {
      const a = tourLatLngs[i];
      const b = tourLatLngs[i + 1];
      out.push([a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]);
    } else if (i >= segCount) {
      out.push(tourLatLngs[segCount]);
    }
    return out;
  }, [t, tourLatLngs, segCount]);

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

          {/* Tour global em ghost (amarelo translúcido) */}
          {globalLatLngs.length > 0 && (
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

          {/* Tour atual — escurecido durante playback pra a parte percorrida sobressair */}
          {tourLatLngs.length > 0 && (
            <Polyline
              positions={tourLatLngs}
              pathOptions={{
                color: '#ff00aa',
                weight: 3,
                opacity: showingPlayback ? 0.25 : 0.9,
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

      {/* Controles de playback */}
      {tourLatLngs.length >= 2 && (
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

          {showingPlayback && currentSegIdx < tour.length && (
            <span>
              passo <span style={{ color: 'var(--cyan)' }}>{currentSegIdx + 1}</span> /{' '}
              {tour.length}
              {' — '}
              <span style={{ color: 'var(--pink)' }}>
                {cidades.find(c => c.id === tour[Math.min(currentSegIdx, tour.length - 1)])?.nome ?? ''}
              </span>
              {' → '}
              <span style={{ color: 'var(--cyan)' }}>
                {cidades.find(c => c.id === tour[(currentSegIdx + 1) % tour.length])?.nome ?? ''}
              </span>
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
