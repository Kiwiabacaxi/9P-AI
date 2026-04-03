import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, ReferenceLine, Tooltip, Legend, Area } from 'recharts';
import type { TsPoint, TsForecastPoint } from '../../api/types';

interface Props {
  pontos: TsPoint[];
  forecast?: TsForecastPoint[];
  validStart?: number;
  height?: number;
}

export default function TimeSeriesChart({ pontos, forecast, validStart, height = 300 }: Props) {
  if (!pontos || pontos.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const chartData = useMemo(() => {
    // Historical + validation points
    const pts = pontos.map((p) => ({
      data: p.data.slice(5), // MM-DD
      real: p.preco,
      predito: p.predito,
      upper: undefined as number | undefined,
      lower: undefined as number | undefined,
      forecast: undefined as number | undefined,
    }));

    // Append forecast points
    if (forecast && forecast.length > 0) {
      for (const f of forecast) {
        pts.push({
          data: `D+${f.dia}`,
          real: undefined as unknown as number,
          predito: undefined as unknown as number,
          forecast: f.predito,
          upper: f.upper,
          lower: f.lower,
        });
      }
    }

    return pts;
  }, [pontos, forecast]);

  const validDate = validStart != null && validStart < pontos.length
    ? pontos[validStart].data.slice(5) : null;

  // Forecast start reference line
  const forecastDate = forecast && forecast.length > 0 ? 'D+1' : null;

  const interval = Math.max(0, Math.floor(chartData.length / 10) - 1);

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="data" stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            tickLine={false} interval={interval} />
          <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            tickLine={false} domain={['auto', 'auto']} />

          {validDate && (
            <ReferenceLine x={validDate} stroke="#ff6ec7" strokeDasharray="4 4" strokeWidth={1.5}
              label={{ value: 'VALID', position: 'top', fill: '#ff6ec7', fontSize: 8, fontFamily: 'JetBrains Mono' }} />
          )}
          {forecastDate && (
            <ReferenceLine x={forecastDate} stroke="#ffaa00" strokeDasharray="4 4" strokeWidth={1.5}
              label={{ value: 'FUTURO', position: 'top', fill: '#ffaa00', fontSize: 8, fontFamily: 'JetBrains Mono' }} />
          )}

          <Tooltip
            contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }}
            labelStyle={{ color: '#888' }}
          />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10 }} />

          {/* Confidence interval area */}
          {forecast && forecast.length > 0 && (
            <Area type="monotone" dataKey="upper" stroke="none" fill="#ff6ec720" isAnimationActive={false} name="intervalo" />
          )}
          {forecast && forecast.length > 0 && (
            <Area type="monotone" dataKey="lower" stroke="none" fill="#1c2026" isAnimationActive={false} legendType="none" />
          )}

          {/* Historical lines */}
          <Line type="monotone" dataKey="real" stroke="#00fbfb" strokeWidth={2} dot={false}
            name="preço real" isAnimationActive={false} connectNulls={false} />
          <Line type="monotone" dataKey="predito" stroke="#00ff00" strokeWidth={1.5} dot={false}
            name="predição treino" isAnimationActive={false} strokeDasharray="4 2" connectNulls={false} />

          {/* Forecast line */}
          {forecast && forecast.length > 0 && (
            <Line type="monotone" dataKey="forecast" stroke="#ffaa00" strokeWidth={2.5}
              dot={{ r: 3, fill: '#ffaa00' }} name="previsão futura" isAnimationActive={false} connectNulls={false} />
          )}
          {forecast && forecast.length > 0 && (
            <Line type="monotone" dataKey="upper" stroke="#ff6ec7" strokeWidth={1} dot={false}
              strokeDasharray="3 3" name="limite superior" isAnimationActive={false} connectNulls={false} />
          )}
          {forecast && forecast.length > 0 && (
            <Line type="monotone" dataKey="lower" stroke="#ff6ec7" strokeWidth={1} dot={false}
              strokeDasharray="3 3" name="limite inferior" isAnimationActive={false} connectNulls={false} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
