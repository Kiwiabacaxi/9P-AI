import { useMemo, useState, useEffect, useRef } from 'react';

interface Props {
  layerSizes: number[];
  activeLayer?: number;
  hudText?: string;
  animate?: boolean; // true = idle animation cycling layers
}

export default function NetworkViz({ layerSizes, activeLayer = -1, hudText, animate = true }: Props) {
  const MAX_SHOWN = 8;
  const H = 300;
  const padX = 55, padY = 32;
  const R = 8;

  // Idle animation: cycle through layers when no activeLayer is set
  const [idleLayer, setIdleLayer] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    if (activeLayer >= 0 || !animate) {
      // Training is active or animation disabled — stop idle
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    // Idle animation: sweep from layer 0 to last and back
    const nLayers = layerSizes.length;
    if (nLayers <= 1) return;
    let step = 0;
    intervalRef.current = setInterval(() => {
      step = (step + 1) % (nLayers * 2 - 2);
      const layer = step < nLayers ? step : (nLayers * 2 - 2 - step);
      setIdleLayer(layer);
    }, 600);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [activeLayer, animate, layerSizes.length]);

  const currentActive = activeLayer >= 0 ? activeLayer : (animate ? idleLayer : -1);

  const layout = useMemo(() => {
    if (!layerSizes || layerSizes.length === 0) return null;
    const nLayers = layerSizes.length;
    const usableW = 800 - padX * 2;
    const usableH = H - padY * 2 - 24;

    const positions: { x: number; y: number }[][] = [];
    for (let l = 0; l < nLayers; l++) {
      const shown = Math.min(layerSizes[l], MAX_SHOWN);
      const xFrac = nLayers === 1 ? 0.5 : l / (nLayers - 1);
      const x = padX + xFrac * usableW;
      const layer: { x: number; y: number }[] = [];
      for (let j = 0; j < shown; j++) {
        const yFrac = shown === 1 ? 0.5 : j / (shown - 1);
        const y = padY + 12 + yFrac * usableH;
        layer.push({ x, y });
      }
      positions.push(layer);
    }
    return { positions, nLayers };
  }, [layerSizes]);

  if (!layout) return <div className="net-viz-container" style={{ height: H }} />;

  const { positions, nLayers } = layout;
  const labels = layerSizes.map((_, i) =>
    i === 0 ? 'IN' : i === nLayers - 1 ? 'OUT' : `H${i}`
  );

  return (
    <div className="net-viz-container">
      <style>{`
        @keyframes node-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
        @keyframes edge-flow {
          0% { stroke-dashoffset: 16; }
          100% { stroke-dashoffset: 0; }
        }
        .nv-node-active {
          animation: node-pulse 0.9s ease-in-out infinite;
        }
        .nv-edge-active {
          stroke-dasharray: 8 8;
          animation: edge-flow 0.8s linear infinite;
        }
      `}</style>
      <svg viewBox="0 0 800 300" style={{ width: '100%', height: H, display: 'block' }}>
        <defs>
          <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-pink" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Grid background */}
        <pattern id="net-grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#ffffff" strokeWidth="0.3" opacity="0.05" />
        </pattern>
        <rect width="800" height="300" fill="url(#net-grid)" />

        {/* Edges */}
        {positions.map((layer, l) =>
          l < nLayers - 1 && layer.map((from, i) =>
            positions[l + 1].map((to, j) => {
              const isActive = currentActive === l || currentActive === l + 1;
              return (
                <line
                  key={`e-${l}-${i}-${j}`}
                  x1={from.x} y1={from.y}
                  x2={to.x} y2={to.y}
                  stroke={isActive ? '#00ff0050' : '#ffffff10'}
                  strokeWidth={isActive ? 1.2 : 0.6}
                  className={isActive ? 'nv-edge-active' : undefined}
                />
              );
            })
          )
        )}

        {/* Nodes */}
        {positions.map((layer, l) => {
          const isInput = l === 0;
          const isOutput = l === nLayers - 1;
          const isActive = l === currentActive;
          let fill = '#1c2026';
          let stroke = '#333';
          let filter: string | undefined;
          if (isActive) {
            fill = '#002200';
            stroke = '#00ff00';
            filter = 'url(#glow-green)';
          } else if (isInput) {
            stroke = '#00fbfb';
            filter = 'url(#glow-cyan)';
          } else if (isOutput) {
            stroke = '#ff6ec7';
            filter = 'url(#glow-pink)';
          }

          return layer.map((node, j) => (
            <circle
              key={`n-${l}-${j}`}
              cx={node.x} cy={node.y} r={R}
              fill={fill} stroke={stroke} strokeWidth={1.5}
              filter={filter}
              className={isActive ? 'nv-node-active' : undefined}
            />
          ));
        })}

        {/* Ellipsis if truncated */}
        {layerSizes.map((n, l) => {
          if (n <= MAX_SHOWN) return null;
          const layer = positions[l];
          const lastY = layer[layer.length - 1].y;
          return (
            <text key={`dots-${l}`} x={layer[0].x} y={lastY + 20}
              textAnchor="middle" fill="#555" fontSize="14" fontFamily="JetBrains Mono">
              ⋮
            </text>
          );
        })}

        {/* Layer labels */}
        {positions.map((layer, l) => (
          <text key={`lbl-${l}`} x={layer[0].x} y={padY - 2}
            textAnchor="middle" fill={l === currentActive ? '#00ff00' : '#555'}
            fontSize="9" fontFamily="JetBrains Mono" letterSpacing="0.1em">
            {labels[l]} ({layerSizes[l]})
          </text>
        ))}

        {/* HUD text */}
        {hudText && (
          <text x={790} y={16} textAnchor="end" fill="#555"
            fontSize="9" fontFamily="JetBrains Mono" letterSpacing="0.05em">
            {hudText}
          </text>
        )}
      </svg>
    </div>
  );
}
