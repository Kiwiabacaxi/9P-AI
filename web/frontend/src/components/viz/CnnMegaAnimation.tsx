import { useState, useRef, useEffect, useCallback } from 'react';
import type { CnnVisualizeResp } from '../../api/types';

// =============================================================================
// Mega-Animação CNN — Visualização cinematográfica do processo de convolução
//
// 5 fases: Intro → Conv1 → Pool1 → Conv2+Pool2 → Flatten+Dense+Softmax
// Mostra kernel deslizando, multiplicação, feature maps construindo pixel a pixel
// =============================================================================

interface Props {
  vizData: CnnVisualizeResp;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Helpers — computação client-side
// ---------------------------------------------------------------------------

function reshapeInput(flat: number[]): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < 28; r++) {
    grid.push(flat.slice(r * 28, (r + 1) * 28));
  }
  return grid;
}

function computeConvAt(input: number[][], kernel: number[][], r: number, c: number): {
  patch: number[][]; products: number[][]; sum: number; output: number;
} {
  const patch: number[][] = [];
  const products: number[][] = [];
  let sum = 0;
  for (let kh = 0; kh < 3; kh++) {
    const pRow: number[] = [];
    const prRow: number[] = [];
    for (let kw = 0; kw < 3; kw++) {
      const inp = input[r + kh]?.[c + kw] ?? 0;
      const ker = kernel[kh][kw];
      pRow.push(inp);
      prRow.push(inp * ker);
      sum += inp * ker;
    }
    patch.push(pRow);
    products.push(prRow);
  }
  return { patch, products, sum, output: Math.max(0, sum) };
}

function computePoolAt(map: number[][], r: number, c: number): {
  values: number[]; maxVal: number; maxIdx: number;
} {
  const values = [
    map[r * 2]?.[c * 2] ?? 0,
    map[r * 2]?.[c * 2 + 1] ?? 0,
    map[r * 2 + 1]?.[c * 2] ?? 0,
    map[r * 2 + 1]?.[c * 2 + 1] ?? 0,
  ];
  let maxIdx = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[maxIdx]) maxIdx = i;
  return { values, maxVal: values[maxIdx], maxIdx };
}

// Feature map mini-grid renderer
function MiniGrid({ data, cellSize, highlightR, highlightC, highlightSize, label, builtCells }: {
  data: number[][]; cellSize: number; highlightR?: number; highlightC?: number;
  highlightSize?: number; label?: string; builtCells?: Set<string>;
}) {
  let min = Infinity, max = -Infinity;
  for (const row of data) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
  const range = max - min || 1;
  const w = data[0]?.length || 0;
  const hs = highlightSize ?? 3;

  return (
    <div style={{ display: 'inline-block', textAlign: 'center' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${w}, ${cellSize}px)`, gap: 0 }}>
          {data.flatMap((row, r) =>
            row.map((v, c) => {
              const norm = (v - min) / range;
              const brightness = Math.round(norm * 255);
              const isBuilt = !builtCells || builtCells.has(`${r},${c}`);
              return (
                <div key={`${r}-${c}`} style={{
                  width: cellSize, height: cellSize,
                  background: isBuilt ? `rgb(0, ${brightness}, ${Math.round(brightness * 0.8)})` : '#111',
                  transition: 'background 0.3s',
                }} />
              );
            })
          )}
        </div>
        {/* Sliding overlay box */}
        {highlightR != null && highlightC != null && (
          <div style={{
            position: 'absolute',
            top: highlightR * cellSize - 1,
            left: highlightC * cellSize - 1,
            width: hs * cellSize + 2,
            height: hs * cellSize + 2,
            border: '2px solid #00fbfb',
            boxShadow: '0 0 12px rgba(0,251,251,0.6), inset 0 0 8px rgba(0,251,251,0.2)',
            transition: 'top 300ms ease-out, left 300ms ease-out',
            pointerEvents: 'none',
          }} />
        )}
      </div>
      {label && <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginTop: 4 }}>{label}</div>}
    </div>
  );
}

// Math panel — shows patch × kernel = sum
function MathPanel({ patch, kernel, products, sum, output }: {
  patch: number[][]; kernel: number[][]; products: number[][]; sum: number; output: number;
}) {
  return (
    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, textAlign: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
        {/* Patch */}
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>patch</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 36px)', gap: 1 }}>
            {patch.flat().map((v, i) => (
              <div key={i} style={{ background: '#1a2a1a', padding: '3px 2px', textAlign: 'center', color: 'var(--cyan)', fontSize: 9 }}>
                {v.toFixed(2)}
              </div>
            ))}
          </div>
        </div>
        <span style={{ color: '#555', fontSize: 16 }}>×</span>
        {/* Kernel */}
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>kernel</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 36px)', gap: 1 }}>
            {kernel.flat().map((v, i) => (
              <div key={i} style={{
                background: v >= 0 ? '#0a2a0a' : '#2a0a1a', padding: '3px 2px', textAlign: 'center',
                color: v >= 0 ? 'var(--primary-glow)' : 'var(--pink)', fontSize: 9,
              }}>
                {v.toFixed(2)}
              </div>
            ))}
          </div>
        </div>
        <span style={{ color: '#555', fontSize: 16 }}>=</span>
        {/* Products */}
        <div>
          <div style={{ fontSize: 8, color: '#888', marginBottom: 2 }}>produtos</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 36px)', gap: 1 }}>
            {products.flat().map((v, i) => (
              <div key={i} style={{
                background: v >= 0 ? '#0a1a0a' : '#1a0a0a', padding: '3px 2px', textAlign: 'center',
                color: v >= 0 ? '#8f8' : '#f88', fontSize: 9,
              }}>
                {v.toFixed(3)}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div style={{ marginTop: 8, fontSize: 12 }}>
        <span style={{ color: '#888' }}>soma = </span>
        <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>{sum.toFixed(4)}</span>
        <span style={{ color: '#555' }}> → ReLU → </span>
        <span style={{ color: 'var(--primary-glow)', fontWeight: 700 }}>{output.toFixed(4)}</span>
      </div>
    </div>
  );
}

// Pool math panel
function PoolPanel({ values, maxIdx }: { values: number[]; maxIdx: number }) {
  return (
    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, textAlign: 'center' }}>
      <div style={{ fontSize: 8, color: '#888', marginBottom: 4 }}>janela 2×2 — MAX destacado</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 44px)', gap: 2, justifyContent: 'center' }}>
        {values.map((v, i) => (
          <div key={i} style={{
            padding: '6px 4px', textAlign: 'center', fontSize: 11, fontWeight: 700,
            background: i === maxIdx ? '#002a00' : '#1a1a1a',
            color: i === maxIdx ? 'var(--primary-glow)' : '#666',
            border: i === maxIdx ? '2px solid var(--primary-glow)' : '1px solid #333',
            boxShadow: i === maxIdx ? '0 0 8px rgba(0,255,0,0.3)' : 'none',
          }}>
            {v.toFixed(3)}
          </div>
        ))}
      </div>
    </div>
  );
}

// Demo positions for each phase
const CONV1_POSITIONS = [[0, 0], [0, 12], [0, 23], [12, 0], [12, 12], [12, 23], [23, 0], [23, 12]];
const POOL1_POSITIONS = [[0, 0], [0, 6], [6, 6], [12, 12]];
const CONV2_POSITIONS = [[0, 0], [0, 8], [5, 5], [8, 8]];


// =============================================================================
// Main component
// =============================================================================

export default function CnnMegaAnimation({ vizData, onClose }: Props) {
  const [phase, setPhase] = useState(0);
  const [substep, setSubstep] = useState(0);
  const [subphase, setSubphase] = useState<'move' | 'compute' | 'result' | 'ff' | 'summary'>('move');
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const timeouts = useRef<number[]>([]);
  const [builtCells, setBuiltCells] = useState<Set<string>>(new Set());
  const [convResult, setConvResult] = useState<ReturnType<typeof computeConvAt> | null>(null);
  const [poolResult, setPoolResult] = useState<ReturnType<typeof computePoolAt> | null>(null);

  const inputGrid = useRef(reshapeInput(vizData.input));
  const kernel1 = vizData.filters1[0][0]; // filter 0, channel 0

  const clearTimeouts = useCallback(() => {
    timeouts.current.forEach(clearTimeout);
    timeouts.current = [];
  }, []);

  const t = useCallback((ms: number) => ms / speed, [speed]);

  // Cleanup on unmount
  useEffect(() => () => clearTimeouts(), [clearTimeouts]);

  // Schedule next step
  const schedule = useCallback((fn: () => void, ms: number) => {
    if (!playing) return;
    const id = window.setTimeout(fn, t(ms));
    timeouts.current.push(id);
  }, [playing, t]);

  // Phase sequencer
  useEffect(() => {
    if (!playing) return;
    clearTimeouts();

    if (phase === 0) {
      // Intro
      schedule(() => { setPhase(1); setSubstep(0); setSubphase('move'); setBuiltCells(new Set()); }, 2000);
    } else if (phase === 1) {
      // Conv1 demo
      const positions = CONV1_POSITIONS;
      if (subphase === 'move' && substep < positions.length) {
        const [r, c] = positions[substep];
        const result = computeConvAt(inputGrid.current, kernel1, r, c);
        setConvResult(result);
        schedule(() => setSubphase('compute'), 400);
      } else if (subphase === 'compute') {
        schedule(() => setSubphase('result'), 500);
      } else if (subphase === 'result') {
        const [r, c] = positions[substep];
        setBuiltCells(prev => new Set([...prev, `${r},${c}`]));
        if (substep + 1 < positions.length) {
          schedule(() => { setSubstep(substep + 1); setSubphase('move'); }, 400);
        } else {
          schedule(() => setSubphase('ff'), 400);
        }
      } else if (subphase === 'ff') {
        // Fast-forward: fill all cells
        const all = new Set<string>();
        for (let r = 0; r < 26; r++) for (let c = 0; c < 26; c++) all.add(`${r},${c}`);
        setBuiltCells(all);
        schedule(() => setSubphase('summary'), 800);
      } else if (subphase === 'summary') {
        schedule(() => {
          setPhase(2); setSubstep(0); setSubphase('move');
          setBuiltCells(new Set()); setConvResult(null);
        }, 2000);
      }
    } else if (phase === 2) {
      // Pool1 demo
      const positions = POOL1_POSITIONS;
      if (subphase === 'move' && substep < positions.length) {
        const [r, c] = positions[substep];
        setPoolResult(computePoolAt(vizData.conv1Maps[0], r, c));
        schedule(() => setSubphase('compute'), 400);
      } else if (subphase === 'compute') {
        schedule(() => setSubphase('result'), 600);
      } else if (subphase === 'result') {
        const [r, c] = positions[substep];
        setBuiltCells(prev => new Set([...prev, `${r},${c}`]));
        if (substep + 1 < positions.length) {
          schedule(() => { setSubstep(substep + 1); setSubphase('move'); }, 400);
        } else {
          schedule(() => setSubphase('ff'), 400);
        }
      } else if (subphase === 'ff') {
        const all = new Set<string>();
        for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) all.add(`${r},${c}`);
        setBuiltCells(all);
        schedule(() => {
          setPhase(3); setSubstep(0); setSubphase('move');
          setBuiltCells(new Set()); setPoolResult(null);
        }, 1500);
      }
    } else if (phase === 3) {
      // Conv2 + Pool2 (abbreviated)
      const positions = CONV2_POSITIONS;
      if (subphase === 'move' && substep < positions.length) {
        schedule(() => setSubphase('result'), 600);
      } else if (subphase === 'result') {
        const [r, c] = positions[substep];
        setBuiltCells(prev => new Set([...prev, `${r},${c}`]));
        if (substep + 1 < positions.length) {
          schedule(() => { setSubstep(substep + 1); setSubphase('move'); }, 400);
        } else {
          schedule(() => setSubphase('ff'), 400);
        }
      } else if (subphase === 'ff') {
        schedule(() => {
          setPhase(4); setSubstep(0); setSubphase('move'); setBuiltCells(new Set());
        }, 1200);
      }
    } else if (phase === 4) {
      // Pool2 quick
      if (subphase === 'move') {
        schedule(() => setSubphase('ff'), 1000);
      } else if (subphase === 'ff') {
        schedule(() => { setPhase(5); setSubstep(0); }, 1000);
      }
    } else if (phase === 5) {
      // Finale — auto-step through substeps
      if (substep < 4) {
        schedule(() => setSubstep(substep + 1), 1200);
      }
    }
  }, [phase, substep, subphase, playing, speed, clearTimeouts, schedule, vizData, kernel1]);

  const togglePause = () => {
    if (playing) clearTimeouts();
    setPlaying(!playing);
  };

  const skipPhase = () => {
    clearTimeouts();
    setBuiltCells(new Set());
    setConvResult(null);
    setPoolResult(null);
    if (phase < 5) {
      setPhase(phase + 1);
      setSubstep(0);
      setSubphase('move');
    }
  };

  // Current positions for overlay
  const conv1Pos = phase === 1 && substep < CONV1_POSITIONS.length ? CONV1_POSITIONS[substep] : null;
  const pool1Pos = phase === 2 && substep < POOL1_POSITIONS.length ? POOL1_POSITIONS[substep] : null;
  const conv2Pos = phase === 3 && substep < CONV2_POSITIONS.length ? CONV2_POSITIONS[substep] : null;

  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)', padding: 20, marginBottom: 24,
      position: 'relative',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--cyan)', fontWeight: 700 }}>
          MEGA ANIMAÇÃO — Fase {phase}/5: {['Intro', 'Convolução (Conv1)', 'MaxPooling', 'Conv2 + Pool2', 'Pool2', 'Flatten → Dense → Softmax'][phase]}
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {[0.5, 1, 2].map(s => (
            <button key={s} className={`porta-chip${speed === s ? ' selected' : ''}`}
              style={{ padding: '3px 8px', fontSize: 9 }}
              onClick={() => setSpeed(s)}>{s}x</button>
          ))}
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '4px 10px' }} onClick={togglePause}>
            {playing ? '⏸ PAUSAR' : '▶ PLAY'}
          </button>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '4px 10px' }} onClick={skipPhase}>
            ⏭ PULAR
          </button>
          <button className="btn btn-ghost" style={{ fontSize: 10, padding: '4px 10px', color: 'var(--pink)' }} onClick={onClose}>
            ✕ FECHAR
          </button>
        </div>
      </div>

      {/* Phase 0: Intro */}
      {phase === 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 32, justifyContent: 'center', padding: '20px 0' }}>
          <MiniGrid data={inputGrid.current} cellSize={8} label="Input 28×28" />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--on-surface)', maxWidth: 300, lineHeight: 1.8 }}>
            Vamos acompanhar como a CNN processa esta imagem, camada por camada...
            <div style={{ color: 'var(--cyan)', marginTop: 8, fontSize: 11 }}>
              Conv1 → Pool1 → Conv2 → Pool2 → Flatten → Dense → Softmax
            </div>
          </div>
        </div>
      )}

      {/* Phase 1: Conv1 Demo */}
      {phase === 1 && subphase !== 'summary' && (
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', justifyContent: 'center' }}>
          {/* Input with sliding kernel overlay */}
          <div>
            <MiniGrid
              data={inputGrid.current} cellSize={7}
              highlightR={conv1Pos?.[0]} highlightC={conv1Pos?.[1]} highlightSize={3}
              label={`Input 28×28 — kernel em (${conv1Pos?.[0] ?? '?'},${conv1Pos?.[1] ?? '?'})`}
            />
          </div>

          {/* Math panel */}
          <div style={{ minWidth: 280 }}>
            {convResult && subphase !== 'move' ? (
              <MathPanel patch={convResult.patch} kernel={kernel1} products={convResult.products}
                sum={convResult.sum} output={convResult.output} />
            ) : (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', textAlign: 'center', padding: 20 }}>
                movendo kernel...
              </div>
            )}
          </div>

          {/* Feature map building up */}
          <div>
            <MiniGrid
              data={vizData.conv1Maps[0]} cellSize={4} builtCells={subphase === 'ff' ? undefined : builtCells}
              highlightR={conv1Pos?.[0]} highlightC={conv1Pos?.[1]} highlightSize={1}
              label="Conv1 Feature Map (filtro 1/8)"
            />
          </div>
        </div>
      )}

      {/* Phase 1: Summary — show all 8 filters */}
      {phase === 1 && subphase === 'summary' && (
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--primary-glow)', marginBottom: 12 }}>
            8 filtros diferentes extraem features diferentes da imagem
          </div>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
            {vizData.conv1Maps.map((map, i) => (
              <MiniGrid key={i} data={map} cellSize={3} label={`F${i + 1}`} />
            ))}
          </div>
        </div>
      )}

      {/* Phase 2: Pool1 Demo */}
      {phase === 2 && (
        <div style={{ display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
          <div>
            <MiniGrid
              data={vizData.conv1Maps[0]} cellSize={5}
              highlightR={pool1Pos ? pool1Pos[0] * 2 : undefined}
              highlightC={pool1Pos ? pool1Pos[1] * 2 : undefined}
              highlightSize={2}
              label="Conv1 Output 26×26"
            />
          </div>

          <div style={{ minWidth: 120 }}>
            {poolResult && subphase !== 'move' ? (
              <PoolPanel values={poolResult.values} maxIdx={poolResult.maxIdx} />
            ) : (
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#555', textAlign: 'center', padding: 20 }}>
                MaxPool 2×2
              </div>
            )}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--pink)', marginTop: 8, textAlign: 'center' }}>
              Reduz pela metade, mantendo os valores mais fortes
            </div>
          </div>

          <div>
            <MiniGrid
              data={vizData.pool1Maps[0]} cellSize={7} builtCells={subphase === 'ff' ? undefined : builtCells}
              label="Pool1 13×13"
            />
          </div>
        </div>
      )}

      {/* Phase 3: Conv2 (abbreviated) */}
      {phase === 3 && (
        <div style={{ display: 'flex', gap: 32, alignItems: 'flex-start', justifyContent: 'center' }}>
          <div>
            <MiniGrid
              data={vizData.pool1Maps[0]} cellSize={7}
              highlightR={conv2Pos?.[0]} highlightC={conv2Pos?.[1]} highlightSize={3}
              label="Pool1 13×13"
            />
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', maxWidth: 200, textAlign: 'center', alignSelf: 'center' }}>
            <div style={{ color: 'var(--cyan)', fontWeight: 700, marginBottom: 8 }}>Conv2: 16 filtros 3×3</div>
            Cada filtro combina os <b style={{ color: 'var(--pink)' }}>8 canais</b> do Pool1 em features mais complexas
          </div>
          <div>
            <MiniGrid
              data={vizData.conv2Maps[0]} cellSize={6} builtCells={subphase === 'ff' ? undefined : builtCells}
              label="Conv2 11×11 (filtro 1/16)"
            />
          </div>
        </div>
      )}

      {/* Phase 4: Pool2 (quick) */}
      {phase === 4 && (
        <div style={{ display: 'flex', gap: 32, alignItems: 'center', justifyContent: 'center' }}>
          <MiniGrid data={vizData.conv2Maps[0]} cellSize={6} label="Conv2 11×11" />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--pink)' }}>→ MaxPool 2×2 →</div>
          <MiniGrid data={vizData.pool2Maps[0]} cellSize={10} label="Pool2 5×5" />
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)', maxWidth: 200 }}>
            Representação compacta: <b style={{ color: 'var(--cyan)' }}>16 mapas de 5×5</b> = 400 valores
          </div>
        </div>
      )}

      {/* Phase 5: Flatten + Dense + Softmax */}
      {phase === 5 && (
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          {/* Substep 0: show pool2 maps */}
          <div style={{ transition: 'all 0.5s', opacity: substep >= 0 ? 1 : 0 }}>
            <div style={{ display: 'flex', gap: 4, justifyContent: 'center', marginBottom: 12 }}>
              {vizData.pool2Maps.slice(0, 16).map((map, i) => (
                <MiniGrid key={i} data={map} cellSize={4} />
              ))}
            </div>
          </div>

          {/* Substep 1: Flatten bar */}
          <div style={{ transition: 'all 0.5s', opacity: substep >= 1 ? 1 : 0, height: substep >= 1 ? 'auto' : 0 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--cyan)', marginBottom: 4 }}>↓ FLATTEN ↓</div>
            <div style={{ width: 400, height: 8, background: 'linear-gradient(90deg, #00fbfb30, #00fbfb60, #00fbfb30)',
              border: '1px solid #00fbfb40', margin: '0 auto 8px' }} />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: '#888' }}>vetor de 400 valores</div>
          </div>

          {/* Substep 2: Dense layers */}
          <div style={{ transition: 'all 0.5s', opacity: substep >= 2 ? 1 : 0, marginTop: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
              <div style={{ width: 200, height: 8, background: 'linear-gradient(90deg, #00ff0020, #00ff0040)',
                border: '1px solid #00ff0040' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)' }}>Dense 400→64 (ReLU)</span>
              <span style={{ color: '#555' }}>→</span>
              <div style={{ width: 60, height: 8, background: 'linear-gradient(90deg, #00ff0020, #00ff0040)',
                border: '1px solid #00ff0040' }} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)' }}>Dense 64→26</span>
            </div>
          </div>

          {/* Substep 3: Softmax result */}
          <div style={{ transition: 'all 0.5s', opacity: substep >= 3 ? 1 : 0, marginTop: 16 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--primary-glow)', marginBottom: 4 }}>↓ SOFTMAX ↓</div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 64, fontWeight: 700, color: 'var(--primary-glow)',
              textShadow: '0 0 20px rgba(0,255,0,0.5), 0 0 40px rgba(0,255,0,0.3)',
              transition: 'all 0.5s', transform: substep >= 4 ? 'scale(1)' : 'scale(0.5)',
            }}>
              {vizData.letra}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--on-surface)' }}>
              confiança: {(vizData.top5[0].score * 100).toFixed(1)}%
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 8 }}>
              {vizData.top5.map((c, i) => (
                <span key={i} style={{ fontFamily: 'var(--font-mono)', fontSize: 10,
                  color: i === 0 ? 'var(--primary-glow)' : '#666' }}>
                  {c.letra}: {(c.score * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Progress bar */}
      <div style={{ marginTop: 16, height: 3, background: '#1a1a1a', overflow: 'hidden' }}>
        <div style={{
          height: '100%', background: 'var(--cyan)',
          width: `${(phase / 5) * 100}%`,
          transition: 'width 0.5s',
        }} />
      </div>
    </div>
  );
}
