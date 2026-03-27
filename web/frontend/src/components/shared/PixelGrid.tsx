import { useState, useCallback, useRef, useEffect } from 'react';

interface Props {
  rows: number;
  cols: number;
  cellSize?: number;
  gap?: number;
  values?: number[];
  onChange?: (values: number[]) => void;
  readOnly?: boolean;
  showClear?: boolean;
}

export default function PixelGrid({ rows, cols, cellSize = 28, gap = 4, values: externalValues, onChange, readOnly, showClear = true }: Props) {
  const total = rows * cols;
  const [internalValues, setInternalValues] = useState<number[]>(() => new Array(total).fill(-1));
  const values = externalValues ?? internalValues;

  // Keep a mutable ref of current values so drag doesn't use stale closures
  const valuesRef = useRef(values);
  useEffect(() => { valuesRef.current = values; }, [values]);

  const painting = useRef(false);
  const paintVal = useRef(1);

  const emit = useCallback((next: number[]) => {
    valuesRef.current = next;
    if (onChange) onChange(next);
    else setInternalValues(next);
  }, [onChange]);

  const handleMouseDown = useCallback((idx: number, e: React.MouseEvent) => {
    if (readOnly) return;
    e.preventDefault();
    painting.current = true;
    const newVal = valuesRef.current[idx] === 1 ? -1 : 1;
    paintVal.current = newVal;
    const next = [...valuesRef.current];
    next[idx] = newVal;
    emit(next);
  }, [readOnly, emit]);

  const handleMouseEnter = useCallback((idx: number) => {
    if (readOnly || !painting.current) return;
    const cur = valuesRef.current;
    if (cur[idx] === paintVal.current) return; // already set
    const next = [...cur];
    next[idx] = paintVal.current;
    emit(next);
  }, [readOnly, emit]);

  const handleMouseUp = useCallback(() => {
    painting.current = false;
  }, []);

  const clear = useCallback(() => {
    const next = new Array(total).fill(-1);
    emit(next);
  }, [total, emit]);

  return (
    <div onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
      <div
        className="pixel-grid"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
          gap: `${gap}px`,
          cursor: readOnly ? 'default' : 'crosshair',
          userSelect: 'none',
        }}
      >
        {values.slice(0, total).map((v, i) => (
          <div
            key={i}
            className={`pixel${v === 1 ? ' on' : ''}`}
            style={{ width: cellSize, height: cellSize }}
            onMouseDown={(e) => handleMouseDown(i, e)}
            onMouseEnter={() => handleMouseEnter(i)}
          />
        ))}
      </div>
      {!readOnly && showClear && (
        <div style={{ marginTop: 6, display: 'flex', gap: 6 }}>
          <button className="btn" style={{ fontSize: 11, padding: '3px 8px' }} onClick={clear}>LIMPAR</button>
        </div>
      )}
    </div>
  );
}
