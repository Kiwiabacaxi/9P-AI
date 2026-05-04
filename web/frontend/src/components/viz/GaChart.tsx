import { useMemo } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, ZAxis,
  ResponsiveContainer, CartesianGrid, Legend, Tooltip, ReferenceLine,
} from 'recharts';
import type { GAIndividuo } from '../../api/types';

interface Props {
  populacao: GAIndividuo[];
  melhor: GAIndividuo | null;
  globalBest?: GAIndividuo | null;
  dominioMin: number;
  dominioMax: number;
  optimoX?: number;
  optimoFx?: number;
  height?: number;
}

// f(x) = -|x · sin(√|x|)|  (mesma do backend, replicada para desenhar a curva)
function f(x: number): number {
  return -Math.abs(x * Math.sin(Math.sqrt(Math.abs(x))));
}

export default function GaChart({
  populacao, melhor, globalBest, dominioMin, dominioMax,
  optimoX, optimoFx, height = 340,
}: Props) {
  // amostra densa da curva da função (calculada uma vez por intervalo)
  const curvaData = useMemo(() => {
    const N = 600;
    const data: { x: number; fx: number }[] = new Array(N);
    const step = (dominioMax - dominioMin) / (N - 1);
    for (let i = 0; i < N; i++) {
      const x = dominioMin + i * step;
      data[i] = { x, fx: f(x) };
    }
    return data;
  }, [dominioMin, dominioMax]);

  const popData = useMemo(
    () => populacao.map(ind => ({ x: ind.x, fx: ind.fx })),
    [populacao],
  );
  const melhorData = useMemo(
    () => (melhor ? [{ x: melhor.x, fx: melhor.fx }] : []),
    [melhor],
  );
  const globalBestData = useMemo(
    () => (globalBest ? [{ x: globalBest.x, fx: globalBest.fx }] : []),
    [globalBest],
  );

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[dominioMin, dominioMax]}
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            tickCount={9}
            allowDuplicatedCategory={false}
          />
          <YAxis
            dataKey="fx"
            type="number"
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
          />
          <ZAxis range={[40, 40]} />
          <Tooltip
            shared={false}
            cursor={{ stroke: '#444', strokeWidth: 1, strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const item = payload[0];
              const data = item.payload as { x: number; fx: number };
              const colorMap: Record<string, string> = {
                'f(x)': '#00ff88',
                'populacao': '#00ccff',
                'melhor (geracao)': '#ff00aa',
                'melhor global': '#ffff00',
              };
              const name = String(item.name ?? '');
              return (
                <div style={{
                  background: '#111',
                  border: '1px solid #333',
                  padding: '6px 10px',
                  fontSize: 11,
                  fontFamily: 'JetBrains Mono',
                  color: '#ccc',
                }}>
                  <div style={{ color: colorMap[name] ?? '#888', marginBottom: 4 }}>{name}</div>
                  <div>x &nbsp;&nbsp;= {Number(data.x).toFixed(4)}</div>
                  <div>f(x) = {Number(data.fx).toFixed(4)}</div>
                </div>
              );
            }}
          />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
          <Line
            data={curvaData}
            name="f(x)"
            type="monotone"
            dataKey="fx"
            stroke="#00ff88"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {optimoX !== undefined && (
            <ReferenceLine
              x={optimoX}
              stroke="#ffaa00"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: `x* ≈ ${optimoX.toFixed(2)}`,
                fill: '#ffaa00',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
                position: 'insideTopRight',
              }}
            />
          )}
          {optimoFx !== undefined && (
            <ReferenceLine
              y={optimoFx}
              stroke="#ffaa00"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{
                value: `f(x*) ≈ ${optimoFx.toFixed(2)}`,
                fill: '#ffaa00',
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
                position: 'insideBottomLeft',
              }}
            />
          )}
          <Scatter
            data={popData}
            name="populacao"
            fill="#00ccff"
            isAnimationActive={false}
          />
          <Scatter
            data={melhorData}
            name="melhor (geracao)"
            fill="#ff00aa"
            shape="diamond"
            isAnimationActive={false}
          />
          <Scatter
            data={globalBestData}
            name="melhor global"
            fill="#ffff00"
            shape="star"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// =============================================================================
// Evolução do f(x) ao longo das gerações — melhor + média.
// =============================================================================

interface EvoProps {
  histMelhor: number[];
  histMedia: number[];
  height?: number;
}

export function GaEvoChart({ histMelhor, histMedia, height = 240 }: EvoProps) {
  const data = useMemo(() => {
    const N = Math.max(histMelhor.length, histMedia.length);
    if (N === 0) return [];

    // calcula bestSoFar (mínimo acumulado de histMelhor) — linha monotônica
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
    // downsample para no máx ~250 pontos (mantém início e fim)
    const maxPoints = 250;
    if (N <= maxPoints) return raw;
    const out: Row[] = new Array(maxPoints);
    const step = (N - 1) / (maxPoints - 1);
    for (let i = 0; i < maxPoints; i++) {
      out[i] = raw[Math.round(i * step)];
    }
    return out;
  }, [histMelhor, histMedia]);

  if (data.length === 0) {
    return <div className="chart-wrap" style={{ height }} />;
  }

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
            formatter={(v) => Number(v).toFixed(4)}
          />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
          {/* fundo: ruído per-gen, mais sutis */}
          <Line
            name="media f(x)"
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
          {/* destaque: linha monotônica do melhor acumulado — a história da convergência */}
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
