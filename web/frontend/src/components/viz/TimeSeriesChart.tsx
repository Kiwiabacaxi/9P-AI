import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, ReferenceLine, Tooltip, Legend } from 'recharts';
import type { TsPoint } from '../../api/types';

interface Props {
  pontos: TsPoint[];
  validStart?: number; // index where validation starts
  height?: number;
}

export default function TimeSeriesChart({ pontos, validStart, height = 260 }: Props) {
  if (!pontos || pontos.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const chartData = useMemo(() =>
    pontos.map((p, i) => ({
      data: p.data.slice(5), // MM-DD
      real: p.preco,
      predito: p.predito,
      isValid: validStart != null && i >= validStart,
    }))
  , [pontos, validStart]);

  // Find the validation start date for reference line
  const validDate = validStart != null && validStart < chartData.length
    ? chartData[validStart].data : null;

  // Show ~8 ticks on X axis
  const interval = Math.max(0, Math.floor(chartData.length / 8) - 1);

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
              label={{ value: 'VALIDAÇÃO', position: 'top', fill: '#ff6ec7', fontSize: 9, fontFamily: 'JetBrains Mono' }} />
          )}
          <Tooltip
            contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 10 }}
            labelStyle={{ color: '#888' }}
          />
          <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10 }} />
          <Line type="monotone" dataKey="real" stroke="#00fbfb" strokeWidth={2} dot={false}
            name="preço real" isAnimationActive={false} />
          <Line type="monotone" dataKey="predito" stroke="#00ff00" strokeWidth={2} dot={false}
            name="predição MLP" isAnimationActive={false} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
