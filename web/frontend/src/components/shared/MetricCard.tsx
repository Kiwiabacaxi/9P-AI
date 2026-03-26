interface Props {
  title: string;
  value: string;
  label: string;
  color?: 'green' | 'cyan' | 'pink';
  pulse?: boolean;
  valueStyle?: React.CSSProperties;
}

export default function MetricCard({ title, value, label, color, pulse, valueStyle }: Props) {
  return (
    <div className="card">
      {pulse && <div className="card-pulse" />}
      <div className="card-title">{title}</div>
      <div className={`metric-val${color ? ' ' + color : ''}`} style={valueStyle}>{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
