import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from 'recharts';

interface Props {
  data: number[];
  color?: string;
  height?: number;
}

// Downsample large arrays to at most maxPoints entries, keeping first and last
function downsample(data: number[], maxPoints: number): { ciclo: number; erro: number }[] {
  const n = data.length;
  if (n <= maxPoints) {
    return data.map((v, i) => ({ ciclo: i + 1, erro: v }));
  }
  const result: { ciclo: number; erro: number }[] = [];
  const step = (n - 1) / (maxPoints - 1);
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.round(i * step);
    result.push({ ciclo: idx + 1, erro: data[idx] });
  }
  return result;
}

export default function LogChart({ data, color = '#00ff88', height = 160 }: Props) {
  if (!data || data.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const chartData = useMemo(() => downsample(data, 150), [data]);
  // Show ~6 evenly spaced tick labels on X axis
  const xInterval = Math.max(0, Math.floor(chartData.length / 6) - 1);

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="ciclo"
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            interval={xInterval}
          />
          <YAxis
            scale="log"
            domain={['auto', 'auto']}
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            allowDataOverflow
          />
          <Line
            type="monotone"
            dataKey="erro"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
