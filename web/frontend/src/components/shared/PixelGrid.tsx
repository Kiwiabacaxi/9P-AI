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
  const [localValues, setLocalValues] = useState<number[]>(() => new Array(total).fill(-1));

  // Sync external values into local state when not painting
  const painting = useRef(false);
  useEffect(() => {
    if (externalValues && !painting.current) {
      setLocalValues(externalValues);
    }
  }, [externalValues]);

  const valuesRef = useRef(localValues);
  useEffect(() => { valuesRef.current = localValues; }, [localValues]);

  const paintVal = useRef(1);

  const handleMouseDown = useCallback((idx: number, e: React.MouseEvent) => {
    if (readOnly) return;
    e.preventDefault();
    painting.current = true;
    const newVal = valuesRef.current[idx] === 1 ? -1 : 1;
    paintVal.current = newVal;
    const next = [...valuesRef.current];
    next[idx] = newVal;
    // Only update local state during paint — no parent notification yet
    valuesRef.current = next;
    setLocalValues(next);
  }, [readOnly]);

  const handleMouseEnter = useCallback((idx: number) => {
    if (readOnly || !painting.current) return;
    const cur = valuesRef.current;
    if (cur[idx] === paintVal.current) return;
    const next = [...cur];
    next[idx] = paintVal.current;
    valuesRef.current = next;
    setLocalValues(next);
  }, [readOnly]);

  const handleMouseUp = useCallback(() => {
    if (!painting.current) return;
    painting.current = false;
    // Notify parent ONCE with final state
    if (onChange) onChange(valuesRef.current);
  }, [onChange]);

  const clear = useCallback(() => {
    const next = new Array(total).fill(-1);
    valuesRef.current = next;
    setLocalValues(next);
    if (onChange) onChange(next);
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
        {localValues.slice(0, total).map((v, i) => (
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
