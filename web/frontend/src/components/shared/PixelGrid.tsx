import { useState, useCallback, useRef } from 'react';

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
  const painting = useRef(false);
  const paintVal = useRef(1);

  const setVal = useCallback((idx: number, val: number) => {
    const next = [...values];
    next[idx] = val;
    if (onChange) onChange(next);
    else setInternalValues(next);
  }, [values, onChange]);

  const toggle = useCallback((idx: number) => {
    if (readOnly) return;
    const newVal = values[idx] === 1 ? -1 : 1;
    paintVal.current = newVal;
    setVal(idx, newVal);
  }, [values, readOnly, setVal]);

  const handleMouseDown = useCallback((idx: number) => {
    if (readOnly) return;
    painting.current = true;
    toggle(idx);
  }, [readOnly, toggle]);

  const handleMouseEnter = useCallback((idx: number) => {
    if (readOnly || !painting.current) return;
    setVal(idx, paintVal.current);
  }, [readOnly, setVal]);

  const handleMouseUp = useCallback(() => {
    painting.current = false;
  }, []);

  const clear = useCallback(() => {
    const next = new Array(total).fill(-1);
    if (onChange) onChange(next);
    else setInternalValues(next);
  }, [total, onChange]);

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
            onMouseDown={(e) => { e.preventDefault(); handleMouseDown(i); }}
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
