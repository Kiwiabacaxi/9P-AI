import { useMemo, useEffect, useRef } from 'react';
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

  return (
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

        {/* Tour atual em rosa-fluor */}
        {tourLatLngs.length > 0 && (
          <Polyline
            positions={tourLatLngs}
            pathOptions={{
              color: '#ff00aa',
              weight: 3,
              opacity: 0.9,
            }}
          />
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
