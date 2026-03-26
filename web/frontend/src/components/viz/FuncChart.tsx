import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import type { FuncPoint } from '../../api/types';

interface Props {
  pontos: FuncPoint[];
  height?: number;
}

export default function FuncChart({ pontos, height = 300 }: Props) {
  if (!pontos || pontos.length === 0) return <div className="chart-wrap" style={{ height }} />;

  const data = pontos.map(p => ({
    x: p.x,
    original: p.y,
    rede: p.yPred,
  }));

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="x"
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            type="number"
            domain={[-1, 1]}
            tickCount={5}
          />
          <YAxis
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
          />
          <Legend
            wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }}
          />
          <Line
            name="funcao original"
            type="monotone"
            dataKey="original"
            stroke="#00ff88"
            strokeWidth={3}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            name="aproximacao da rede"
            type="monotone"
            dataKey="rede"
            stroke="#00ccff"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
