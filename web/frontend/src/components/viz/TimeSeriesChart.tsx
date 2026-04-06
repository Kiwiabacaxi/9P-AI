import { useMemo, useState, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, ReferenceLine, Tooltip, Legend, Area } from 'recharts';
import type { TsPoint, TsForecastPoint } from '../../api/types';

interface Props {
  pontos: TsPoint[];
  forecast?: TsForecastPoint[];
  validStart?: number;
  height?: number;
  showConfidence?: boolean; // default true — set false for baselines
}

export default function TimeSeriesChart({ pontos, forecast, validStart, height = 300, showConfidence = true }: Props) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const toggleSeries = useCallback((dataKey: string) => {
    setHidden(prev => {
      const next = new Set(prev);
      if (next.has(dataKey)) next.delete(dataKey);
      else next.add(dataKey);
      return next;
    });
  }, []);

  if (!pontos || pontos.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const hasForecast = forecast && forecast.length > 0;
  const hasConfidence = showConfidence && hasForecast;

  const chartData = useMemo(() => {
    const pts = pontos.map((p) => ({
      data: p.data?.slice(5) || '',
      real: p.preco,
      predito: p.predito,
      upper: undefined as number | undefined,
      lower: undefined as number | undefined,
      forecast: undefined as number | undefined,
    }));

    if (hasForecast) {
      for (const f of forecast!) {
        pts.push({
          data: `D+${f.dia}`,
          real: undefined as unknown as number,
          predito: undefined as unknown as number,
          forecast: f.predito,
          upper: showConfidence ? f.upper : undefined,
          lower: showConfidence ? f.lower : undefined,
        });
      }
    }
    return pts;
  }, [pontos, forecast, hasForecast, showConfidence]);

  const validDate = validStart != null && validStart < pontos.length
    ? pontos[validStart].data?.slice(5) : null;
  const forecastDate = hasForecast ? 'D+1' : null;
  const interval = Math.max(0, Math.floor(chartData.length / 10) - 1);

  // Legend click handler
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleLegendClick = (e: any) => {
    if (e?.dataKey) toggleSeries(String(e.dataKey));
  };

  const isHidden = (key: string) => hidden.has(key);

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
          <Legend
            wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, cursor: 'pointer' }}
            onClick={handleLegendClick}
          />

          {/* Confidence interval area */}
          {hasConfidence && (
            <Area type="monotone" dataKey="upper" stroke="none" fill={isHidden('upper') ? 'none' : '#ff6ec720'}
              isAnimationActive={false} name="intervalo" legendType="none" />
          )}

          {/* Historical lines */}
          <Line type="monotone" dataKey="real" stroke={isHidden('real') ? '#333' : '#00fbfb'}
            strokeWidth={isHidden('real') ? 0.5 : 2} dot={false}
            name="preço real" isAnimationActive={false} connectNulls={false} />
          <Line type="monotone" dataKey="predito" stroke={isHidden('predito') ? '#333' : '#00ff00'}
            strokeWidth={isHidden('predito') ? 0.5 : 1.5} dot={false}
            name="predição" isAnimationActive={false} strokeDasharray="4 2" connectNulls={false} />

          {/* Forecast line */}
          {hasForecast && (
            <Line type="monotone" dataKey="forecast" stroke={isHidden('forecast') ? '#333' : '#ffaa00'}
              strokeWidth={isHidden('forecast') ? 0.5 : 2.5}
              dot={isHidden('forecast') ? false : { r: 3, fill: '#ffaa00' }}
              name="previsão futura" isAnimationActive={false} connectNulls={false} />
          )}
          {hasConfidence && (
            <Line type="monotone" dataKey="upper" stroke={isHidden('upper') ? '#333' : '#ff6ec7'}
              strokeWidth={isHidden('upper') ? 0 : 1} dot={false}
              strokeDasharray="3 3" name="limite superior" isAnimationActive={false} connectNulls={false} />
          )}
          {hasConfidence && (
            <Line type="monotone" dataKey="lower" stroke={isHidden('lower') ? '#333' : '#ff6ec7'}
              strokeWidth={isHidden('lower') ? 0 : 1} dot={false}
              strokeDasharray="3 3" name="limite inferior" isAnimationActive={false} connectNulls={false} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
