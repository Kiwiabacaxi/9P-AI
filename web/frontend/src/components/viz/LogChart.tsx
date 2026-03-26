import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from 'recharts';

interface Props {
  data: number[];
  color?: string;
  height?: number;
}

export default function LogChart({ data, color = '#00ff88', height = 160 }: Props) {
  if (!data || data.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const chartData = data.map((v, i) => ({ ciclo: i + 1, erro: v }));

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
