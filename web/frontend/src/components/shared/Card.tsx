import type { ReactNode } from 'react';

interface Props {
  title?: string;
  pulse?: boolean;
  children: ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export default function Card({ title, pulse, children, className, style }: Props) {
  return (
    <div className={`card${className ? ' ' + className : ''}`} style={style}>
      {pulse && <div className="card-pulse" />}
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}
