import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Cell, Tooltip } from 'recharts';

interface BenchEntry {
  metodo: string;
  tempoMs: number;
  loss: number;
}

interface Props {
  data: BenchEntry[];
  dataKey: 'tempoMs' | 'loss';
  height?: number;
}

const COLORS = ['#00ff88', '#00ccff', '#ff6ec7', '#77ff61'];

export default function BenchChart({ data, dataKey, height = 200 }: Props) {
  if (!data || data.length === 0) return <div className="chart-wrap" style={{ height }} />;

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="metodo"
            stroke="#555"
            tick={{ fill: '#aaa', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
          />
          <YAxis
            stroke="#555"
            tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{ background: '#1c2026', border: '1px solid #333', fontFamily: 'JetBrains Mono', fontSize: 11 }}
            labelStyle={{ color: '#eaffde' }}
          />
          <Bar dataKey={dataKey} radius={[0, 0, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
