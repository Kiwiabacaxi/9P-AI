import { useState, useRef, useEffect, useCallback } from 'react';
import type { CnnVisualizeResp } from '../../api/types';

// =============================================================================
// Mega-Animação CNN — Visualização completa do processo de convolução
//
// Mostra kernel deslizando posição por posição com passo-a-passo manual
// ou auto-play. Cada fase pode ser navegada com ANTERIOR/PRÓXIMO.
// =============================================================================

interface Props {
  vizData: CnnVisualizeResp;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function reshapeInput(flat: number[]): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < 28; r++) grid.push(flat.slice(r * 28, (r + 1) * 28));
  return grid;
}

function computeConvAt(input: number[][], kernel: number[][], r: number, c: number): {
  patch: number[][]; products: number[][]; sum: number; output: number;
} {
  const patch: number[][] = [];
  const products: number[][] = [];
  let sum = 0;
  for (let kh = 0; kh < 3; kh++) {
    const pRow: number[] = [], prRow: number[] = [];
    for (let kw = 0; kw < 3; kw++) {
      const inp = input[r + kh]?.[c + kw] ?? 0;
      const ker = kernel[kh][kw];
      pRow.push(inp); prRow.push(inp * ker);
      sum += inp * ker;
    }
    patch.push(pRow); products.push(prRow);
  }
  return { patch, products, sum, output: Math.max(0, sum) };
}

function computePoolAt(map: number[][], r: number, c: number): {
  values: number[]; maxVal: number; maxIdx: number;
} {
  const values = [
    map[r * 2]?.[c * 2] ?? 0, map[r * 2]?.[c * 2 + 1] ?? 0,
    map[r * 2 + 1]?.[c * 2] ?? 0, map[r * 2 + 1]?.[c * 2 + 1] ?? 0,
  ];
  let maxIdx = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[maxIdx]) maxIdx = i;
  return { values, maxVal: values[maxIdx], maxIdx };
}

// ---------------------------------------------------------------------------
// MiniGrid — feature map renderer with kernel overlay
// ---------------------------------------------------------------------------

function MiniGrid({ data, cellSize, highlightR, highlightC, highlightSize, label, builtUpTo, showGrid }: {
  data: number[][]; cellSize: number; highlightR?: number; highlightC?: number;
  highlightSize?: number; label?: string; builtUpTo?: number; showGrid?: boolean;
}) {
  let min = Infinity, max = -Infinity;
  for (const row of data) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
  const range = max - min || 1;
  const w = data[0]?.length || 0;
  const hs = highlightSize ?? 3;
  const gap = showGrid ? 1 : 0;

  let cellIdx = 0;
  return (
    <div style={{ display: 'inline-block', textAlign: 'center' }}>
      <div style={{ position: 'relative', display: 'inline-block', background: showGrid ? '#1a1f2a' : 'transparent' }}>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${w}, ${cellSize}px)`, gap }}>
          {data.flatMap((row, r) =>
            row.map((v, c) => {
              const norm = (v - min) / range;
              const brightness = Math.round(norm * 255);
              const idx = cellIdx++;
              const show = builtUpTo == null || idx <= builtUpTo;
              return (
                <div key={`${r}-${c}`} style={{
                  width: cellSize, height: cellSize,
                  background: show ? `rgb(0, ${brightness}, ${Math.round(brightness * 0.8)})` : '#0a0a0a',
                  transition: 'background 0.15s',
                }} />
              );
            })
          )}
        </div>
        {highlightR != null && highlightC != null && (
          <div style={{
            position: 'absolute',
            top: highlightR * (cellSize + gap) - 1,
            left: highlightC * (cellSize + gap) - 1,
            width: hs * cellSize + (hs - 1) * gap + 2,
            height: hs * cellSize + (hs - 1) * gap + 2,
            border: '2px solid #00fbfb',
            boxShadow: '0 0 14px rgba(0,251,251,0.7), inset 0 0 10px rgba(0,251,251,0.15)',
            transition: 'top 200ms ease-out, left 200ms ease-out',
            pointerEvents: 'none', borderRadius: 1,
          }} />
        )}
      </div>
      {label && <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginTop: 4 }}>{label}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Math panel
// ---------------------------------------------------------------------------

function MathPanel({ patch, kernel, products, sum, output }: {
  patch: number[][]; kernel: number[][]; products: number[][]; sum: number; output: number;
}) {
  return (
    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, textAlign: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>patch 3×3</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 38px)', gap: 1 }}>
            {patch.flat().map((v, i) => (
              <div key={i} style={{ background: v > 0 ? '#1a2a1a' : '#111', padding: '4px 2px', textAlign: 'center',
                color: v > 0 ? 'var(--cyan)' : '#444', fontSize: 9 }}>{v.toFixed(2)}</div>
            ))}
          </div>
        </div>
        <span style={{ color: '#555', fontSize: 18 }}>×</span>
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>kernel 3×3</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 38px)', gap: 1 }}>
            {kernel.flat().map((v, i) => (
              <div key={i} style={{ background: v >= 0 ? '#0a2a0a' : '#2a0a1a', padding: '4px 2px', textAlign: 'center',
                color: v >= 0 ? 'var(--primary-glow)' : 'var(--pink)', fontSize: 9 }}>{v.toFixed(2)}</div>
            ))}
          </div>
        </div>
        <span style={{ color: '#555', fontSize: 18 }}>=</span>
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>produtos</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 38px)', gap: 1 }}>
            {products.flat().map((v, i) => (
              <div key={i} style={{ background: v >= 0 ? '#0a1a0a' : '#1a0a0a', padding: '4px 2px', textAlign: 'center',
                color: v >= 0 ? '#8f8' : '#f88', fontSize: 9 }}>{v.toFixed(3)}</div>
            ))}
          </div>
        </div>
      </div>
      <div style={{ marginTop: 10, fontSize: 13 }}>
        <span style={{ color: '#888' }}>soma = </span>
        <span style={{ color: 'var(--cyan)', fontWeight: 700, fontSize: 15 }}>{sum.toFixed(4)}</span>
        <span style={{ color: '#555' }}> → ReLU → </span>
        <span style={{ color: 'var(--primary-glow)', fontWeight: 700, fontSize: 15 }}>{output.toFixed(4)}</span>
      </div>
    </div>
  );
}

function PoolPanel({ values, maxIdx }: { values: number[]; maxIdx: number }) {
  return (
    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, textAlign: 'center' }}>
      <div style={{ fontSize: 9, color: '#888', marginBottom: 6 }}>janela 2×2 — MAX em verde</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 50px)', gap: 2, justifyContent: 'center' }}>
        {values.map((v, i) => (
          <div key={i} style={{
            padding: '8px 4px', textAlign: 'center', fontSize: 12, fontWeight: 700,
            background: i === maxIdx ? '#002a00' : '#1a1a1a',
            color: i === maxIdx ? 'var(--primary-glow)' : '#555',
            border: i === maxIdx ? '2px solid var(--primary-glow)' : '1px solid #333',
            boxShadow: i === maxIdx ? '0 0 10px rgba(0,255,0,0.4)' : 'none',
          }}>{v.toFixed(3)}</div>
        ))}
      </div>
    </div>
  );
}

// Phase labels for pipeline
const PHASE_NAMES = ['Intro', 'Conv1 + ReLU', 'MaxPool 2×2', 'Conv2 + ReLU', 'MaxPool 2×2', 'Flatten → Dense → Softmax'];

// =============================================================================
// Main component
// =============================================================================

export default function CnnMegaAnimation({ vizData, onClose }: Props) {
  const [phase, setPhase] = useState(0);
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [filter1, setFilter1] = useState(0); // which conv1 filter (0-7)
  const [filter2, setFilter2] = useState(0); // which conv2 filter (0-15)
  const autoRef = useRef<number>(0);
  const inputGrid = useRef(reshapeInput(vizData.input));

  // Max steps per phase
  const maxSteps = [
    1,   // phase 0: intro (auto-advance)
    676, // phase 1: conv1 = 26×26 positions
    169, // phase 2: pool1 = 13×13 positions
    121, // phase 3: conv2 = 11×11 positions
    25,  // phase 4: pool2 = 5×5 positions
    5,   // phase 5: finale substeps
  ];

  // Current kernel position from step index
  const getPos = (ph: number, s: number): [number, number] => {
    if (ph === 1) return [Math.floor(s / 26), s % 26];       // conv1: 26 cols
    if (ph === 2) return [Math.floor(s / 13), s % 13];       // pool1: 13 cols
    if (ph === 3) return [Math.floor(s / 11), s % 11];       // conv2: 11 cols
    if (ph === 4) return [Math.floor(s / 5), s % 5];         // pool2: 5 cols
    return [0, 0];
  };

  const clearAuto = useCallback(() => {
    if (autoRef.current) { clearInterval(autoRef.current); autoRef.current = 0; }
  }, []);

  useEffect(() => () => clearAuto(), [clearAuto]);

  // Auto-play timer
  useEffect(() => {
    clearAuto();
    if (!autoPlay) return;

    if (phase === 0) {
      autoRef.current = window.setTimeout(() => {
        setPhase(1); setStep(0);
      }, 2000 / speed);
      return;
    }

    if (phase === 5) {
      if (step < maxSteps[5] - 1) {
        autoRef.current = window.setTimeout(() => setStep(s => s + 1), 1200 / speed);
      }
      return;
    }

    // Conv/Pool phases: fast auto-step
    const interval = phase === 1 ? 60 : phase === 2 ? 80 : phase === 3 ? 80 : 100;
    autoRef.current = window.setInterval(() => {
      setStep(s => {
        if (s + 1 >= maxSteps[phase]) {
          clearAuto();
          // Auto-advance to next phase after brief pause
          window.setTimeout(() => {
            setPhase(p => Math.min(p + 1, 5));
            setStep(0);
          }, 600 / speed);
          return s;
        }
        return s + 1;
      });
    }, interval / speed);
  }, [phase, autoPlay, speed, step, clearAuto]);

  // Manual controls
  const goNext = () => {
    clearAuto(); setAutoPlay(false);
    if (step + 1 < maxSteps[phase]) setStep(step + 1);
    else if (phase < 5) { setPhase(phase + 1); setStep(0); }
  };
  const goPrev = () => {
    clearAuto(); setAutoPlay(false);
    if (step > 0) setStep(step - 1);
    else if (phase > 1) { setPhase(phase - 1); setStep(maxSteps[phase - 1] - 1); }
  };
  const skipPhase = () => {
    clearAuto();
    if (phase < 5) { setPhase(phase + 1); setStep(0); }
  };
  const toggleAuto = () => {
    if (autoPlay) clearAuto();
    setAutoPlay(!autoPlay);
  };

  // Compute current state using selected filter
  const [posR, posC] = getPos(phase, step);
  const convResult = phase === 1 ? computeConvAt(inputGrid.current, vizData.filters1[filter1][0], posR, posC) : null;
  const poolResult = phase === 2 ? computePoolAt(vizData.conv1Maps[filter1], posR, posC) : null;
  const conv2Result = phase === 3 ? computeConvAt(vizData.pool1Maps[0], vizData.filters2[filter2][0], posR, posC) : null;

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', padding: 20, marginBottom: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--cyan)', fontWeight: 700 }}>
          ANIMAÇÃO PASSO A PASSO — Fase {phase}/5: {PHASE_NAMES[phase]}
        </div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Speed */}
          {[0.5, 1, 2, 4].map(s => (
            <button key={s} className={`porta-chip${speed === s ? ' selected' : ''}`}
              style={{ padding: '2px 6px', fontSize: 9 }} onClick={() => setSpeed(s)}>{s}x</button>
          ))}
          <span style={{ color: '#333' }}>|</span>
          {/* Step controls */}
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }} onClick={goPrev}>⏮</button>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }} onClick={toggleAuto}>
            {autoPlay ? '⏸' : '▶'}
          </button>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }} onClick={goNext}>⏭</button>
          <span style={{ color: '#333' }}>|</span>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }} onClick={skipPhase}>PULAR FASE →</button>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px', color: 'var(--pink)' }} onClick={onClose}>✕</button>
        </div>
      </div>

      {/* Phase selector + filter selector */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
        {/* Phase jump buttons */}
        <div style={{ display: 'flex', gap: 2 }}>
          {PHASE_NAMES.map((name, i) => (
            <button key={i} className={`porta-chip${phase === i ? ' selected' : ''}`}
              style={{ padding: '2px 8px', fontSize: 8 }}
              onClick={() => { clearAuto(); setPhase(i); setStep(0); }}>
              {i}: {name.split(' ')[0]}
            </button>
          ))}
        </div>

        {/* Filter selector for Conv1 */}
        {(phase === 1 || phase === 2) && (
          <div style={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: '#888' }}>filtro:</span>
            {Array.from({ length: 8 }, (_, i) => (
              <button key={i} className={`porta-chip${filter1 === i ? ' selected' : ''}`}
                style={{ padding: '1px 5px', fontSize: 8, minWidth: 0 }}
                onClick={() => { setFilter1(i); setStep(0); }}>{i + 1}</button>
            ))}
          </div>
        )}

        {/* Filter selector for Conv2 */}
        {(phase === 3 || phase === 4) && (
          <div style={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: '#888' }}>filtro:</span>
            {Array.from({ length: 16 }, (_, i) => (
              <button key={i} className={`porta-chip${filter2 === i ? ' selected' : ''}`}
                style={{ padding: '1px 4px', fontSize: 7, minWidth: 0 }}
                onClick={() => { setFilter2(i); setStep(0); }}>{i + 1}</button>
            ))}
          </div>
        )}
      </div>

      {/* Step counter */}
      {phase >= 1 && phase <= 4 && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', marginBottom: 8 }}>
          Posição ({posR}, {posC}) — passo {step + 1}/{maxSteps[phase]}
          {!autoPlay && <span style={{ color: 'var(--pink)', marginLeft: 8 }}>MANUAL — use ⏮ ⏭ para navegar</span>}
        </div>
      )}

      {/* Phase 0: Intro */}
      {phase === 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 32, justifyContent: 'center', padding: '20px 0' }}>
          <MiniGrid data={inputGrid.current} cellSize={8} label="Input 28×28" />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--on-surface)', maxWidth: 300, lineHeight: 1.8 }}>
            Vamos acompanhar como a CNN processa esta imagem...
            <div style={{ color: 'var(--cyan)', marginTop: 8, fontSize: 11 }}>
              Conv1 → Pool1 → Conv2 → Pool2 → Flatten → Dense → Softmax
            </div>
          </div>
        </div>
      )}

      {/* Phase 1: Conv1 — kernel sliding over full 28×28 input */}
      {phase === 1 && (
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', justifyContent: 'center' }}>
          <MiniGrid data={inputGrid.current} cellSize={8} showGrid
            highlightR={posR} highlightC={posC} highlightSize={3}
            label={`Input 28×28 — kernel 3×3 em (${posR},${posC})`} />
          <div style={{ minWidth: 300, maxWidth: 420 }}>
            {convResult && <MathPanel patch={convResult.patch} kernel={vizData.filters1[filter1][0]}
              products={convResult.products} sum={convResult.sum} output={convResult.output} />}
          </div>
          <MiniGrid data={vizData.conv1Maps[filter1]} cellSize={4}
            builtUpTo={step} highlightR={posR} highlightC={posC} highlightSize={1}
            label={`Conv1 Feature Map (filtro ${filter1 + 1}/8)`} />
        </div>
      )}

      {/* Phase 2: Pool1 — 2×2 window sliding over conv1 output */}
      {phase === 2 && (
        <div style={{ display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
          <MiniGrid data={vizData.conv1Maps[filter1]} cellSize={5} showGrid
            highlightR={posR * 2} highlightC={posC * 2} highlightSize={2}
            label={`Conv1 26×26 (filtro ${filter1 + 1}) — janela 2×2`} />
          <div style={{ minWidth: 130 }}>
            {poolResult && <PoolPanel values={poolResult.values} maxIdx={poolResult.maxIdx} />}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--pink)', marginTop: 12, textAlign: 'center' }}>
              Reduz pela metade<br />mantendo o MAX
            </div>
          </div>
          <MiniGrid data={vizData.pool1Maps[filter1]} cellSize={7}
            builtUpTo={step} highlightR={posR} highlightC={posC} highlightSize={1}
            label={`Pool1 13×13 (filtro ${filter1 + 1})`} />
        </div>
      )}

      {/* Phase 3: Conv2 — kernel sliding over pool1 */}
      {phase === 3 && (
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', justifyContent: 'center' }}>
          <MiniGrid data={vizData.pool1Maps[0]} cellSize={8} showGrid
            highlightR={posR} highlightC={posC} highlightSize={3}
            label={`Pool1 13×13 — kernel 3×3 em (${posR},${posC})`} />
          <div style={{ minWidth: 300, maxWidth: 420 }}>
            {conv2Result && <MathPanel patch={conv2Result.patch} kernel={vizData.filters2[filter2][0]}
              products={conv2Result.products} sum={conv2Result.sum} output={conv2Result.output} />}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-surface)', marginTop: 8, textAlign: 'center' }}>
              Conv2 filtro {filter2 + 1}/16 — soma <b style={{ color: 'var(--pink)' }}>8 canais</b> (mostrando ch0)
            </div>
          </div>
          <MiniGrid data={vizData.conv2Maps[filter2]} cellSize={6}
            builtUpTo={step} highlightR={posR} highlightC={posC} highlightSize={1}
            label={`Conv2 11×11 (filtro ${filter2 + 1}/16)`} />
        </div>
      )}

      {/* Phase 4: Pool2 */}
      {phase === 4 && (
        <div style={{ display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
          <MiniGrid data={vizData.conv2Maps[filter2]} cellSize={7} showGrid
            highlightR={posR * 2} highlightC={posC * 2} highlightSize={2}
            label={`Conv2 11×11 (filtro ${filter2 + 1}) — janela 2×2`} />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--pink)', alignSelf: 'center' }}>→ MaxPool →</div>
          <MiniGrid data={vizData.pool2Maps[filter2]} cellSize={12}
            builtUpTo={step} highlightR={posR} highlightC={posC} highlightSize={1}
            label={`Pool2 5×5 (filtro ${filter2 + 1})`} />
        </div>
      )}

      {/* Phase 5: Flatten + Dense + Softmax */}
      {phase === 5 && (
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          <div style={{ transition: 'opacity 0.5s', opacity: step >= 0 ? 1 : 0 }}>
            <div style={{ display: 'flex', gap: 4, justifyContent: 'center', marginBottom: 12 }}>
              {vizData.pool2Maps.slice(0, 16).map((map, i) => (
                <MiniGrid key={i} data={map} cellSize={4} />
              ))}
            </div>
          </div>
          <div style={{ transition: 'opacity 0.5s', opacity: step >= 1 ? 1 : 0 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--cyan)', marginBottom: 4 }}>↓ FLATTEN ↓</div>
            <div style={{ width: 400, height: 8, background: 'linear-gradient(90deg, #00fbfb30, #00fbfb60, #00fbfb30)',
              border: '1px solid #00fbfb40', margin: '0 auto 4px' }} />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: '#888' }}>vetor de 400 valores</div>
          </div>
          <div style={{ transition: 'opacity 0.5s', opacity: step >= 2 ? 1 : 0, marginTop: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
              <div style={{ width: 200, height: 8, background: 'linear-gradient(90deg, #00ff0020, #00ff0040)', border: '1px solid #00ff0040' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)' }}>Dense 400→64 (ReLU)</span>
              <span style={{ color: '#555' }}>→</span>
              <div style={{ width: 60, height: 8, background: 'linear-gradient(90deg, #00ff0020, #00ff0040)', border: '1px solid #00ff0040' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)' }}>Dense 64→26</span>
            </div>
          </div>
          <div style={{ transition: 'opacity 0.5s', opacity: step >= 3 ? 1 : 0, marginTop: 16 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--primary-glow)' }}>↓ SOFTMAX ↓</div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 64, fontWeight: 700, color: 'var(--primary-glow)',
              textShadow: '0 0 20px rgba(0,255,0,0.5), 0 0 40px rgba(0,255,0,0.3)',
            }}>{vizData.letra}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--on-surface)' }}>
              confiança: {(vizData.top5[0].score * 100).toFixed(1)}%
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 8 }}>
              {vizData.top5.map((c, i) => (
                <span key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 10,
                  color: i === 0 ? 'var(--primary-glow)' : '#666' }}>{c.letra}: {(c.score * 100).toFixed(1)}%</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Progress bar */}
      <div style={{ marginTop: 16, height: 3, background: '#1a1a1a', overflow: 'hidden' }}>
        <div style={{
          height: '100%', background: 'var(--cyan)', transition: 'width 0.2s',
          width: `${((phase + (step / (maxSteps[phase] || 1))) / 6) * 100}%`,
        }} />
      </div>
    </div>
  );
}
