interface Option {
  value: string;
  label: string;
}

interface Props {
  label?: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  style?: React.CSSProperties;
}

export default function Select({ label, options, value, onChange, style }: Props) {
  return (
    <div>
      {label && <div className="imgreg-select-label">{label}</div>}
      <select
        className="imgreg-select"
        value={value}
        onChange={e => onChange(e.target.value)}
        style={style}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}
