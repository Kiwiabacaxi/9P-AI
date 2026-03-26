import { useState, useRef, useEffect, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import LogChart from '../components/viz/LogChart';
import NetworkViz from '../components/viz/NetworkViz';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type { ImgregConfig, ImgregStep } from '../api/types';

// ---------------------------------------------------------------------------
// Config options
// ---------------------------------------------------------------------------

const IMAGENS = [
  { value: 'coracao', label: 'Coracao' },
  { value: 'smiley', label: 'Smiley' },
  { value: 'radial', label: 'Radial' },
  { value: 'brasil', label: 'Brasil' },
];

const LAYERS = [
  { value: '2', label: '2 camadas' },
  { value: '3', label: '3 camadas' },
  { value: '4', label: '4 camadas' },
  { value: '5', label: '5 camadas' },
];

const NEURONS = [
  { value: '16', label: '16 neuronios' },
  { value: '32', label: '32 neuronios' },
  { value: '64', label: '64 neuronios' },
  { value: '128', label: '128 neuronios' },
];

const LR_OPTIONS = [
  { value: '0.001', label: '\u03B1 0.001' },
  { value: '0.005', label: '\u03B1 0.005' },
  { value: '0.01', label: '\u03B1 0.01' },
  { value: '0.02', label: '\u03B1 0.02' },
  { value: '0.05', label: '\u03B1 0.05' },
];

const EPOCAS_OPTIONS = [
  { value: '500', label: '500' },
  { value: '1000', label: '1k' },
  { value: '2000', label: '2k' },
  { value: '5000', label: '5k' },
];

const BATCH_OPTIONS = [
  { value: '16', label: '16' },
  { value: '32', label: '32' },
  { value: '64', label: '64' },
  { value: '128', label: '128' },
];

const WORKERS_OPTIONS = [
  { value: '1', label: '1 worker' },
  { value: '2', label: '2 workers' },
  { value: '4', label: '4 workers' },
  { value: '8', label: '8 workers' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VARIANT_PREFIX: Record<string, string> = {
  standard: '/imgreg',
  goroutines: '/imgreg-goroutines',
  matrix: '/imgreg-matrix',
  minibatch: '/imgreg-minibatch',
};

const VARIANT_LABEL: Record<string, string> = {
  standard: 'Standard',
  goroutines: 'Goroutines',
  matrix: 'Matrix',
  minibatch: 'Minibatch',
};

const VARIANT_SUB: Record<string, string> = {
  standard: 'MLP pixel-a-pixel \u00b7 ReLU + Sigmoid \u00b7 He init',
  goroutines: 'MLP pixel-a-pixel \u00b7 goroutines por camada',
  matrix: 'MLP pixel-a-pixel \u00b7 operacoes matriciais',
  minibatch: 'MLP pixel-a-pixel \u00b7 mini-batch SGD + workers',
};

/** Convert [R,G,B] floats in [0,1] to a CSS hex color */
function rgbToHex(rgb: [number, number, number]): string {
  const r = Math.round(Math.min(1, Math.max(0, rgb[0])) * 255);
  const g = Math.round(Math.min(1, Math.max(0, rgb[1])) * 255);
  const b = Math.round(Math.min(1, Math.max(0, rgb[2])) * 255);
  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  variant?: 'standard' | 'goroutines' | 'matrix' | 'minibatch';
}

export default function ImgregView({ variant = 'standard' }: Props) {
  const toast = useToast();
  const prefix = VARIANT_PREFIX[variant];

  // Config
  const [imagem, setImagem] = useState('coracao');
  const [hiddenLayers, setHiddenLayers] = useState('3');
  const [neuronsPerLayer, setNeuronsPerLayer] = useState('32');
  const [learningRate, setLearningRate] = useState('0.01');
  const [maxEpocas, setMaxEpocas] = useState('2000');
  const [batchSize, setBatchSize] = useState('32');
  const [numWorkers, setNumWorkers] = useState('4');

  // Training state
  const [training, setTraining] = useState(false);
  const [epoca, setEpoca] = useState<string>('\u2014');
  const [loss, setLoss] = useState<string>('\u2014');
  const [tempo, setTempo] = useState<string>('\u2014');
  const [status, setStatus] = useState<string>('aguardando');
  const [statusColor, setStatusColor] = useState<string>('var(--on-surface)');
  const [activeLayer, setActiveLayer] = useState(-1);

  // Grids
  const [targetPixels, setTargetPixels] = useState<[number, number, number][] | null>(null);
  const [outputPixels, setOutputPixels] = useState<[number, number, number][] | null>(null);

  // Log + chart
  const [logLines, setLogLines] = useState<string[]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);

  // SSE ref
  const sseCleanup = useRef<(() => void) | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Fetch target when image changes
  useEffect(() => {
    apiGet<[number, number, number][]>(`/imgreg/target?img=${imagem}`)
      .then(setTargetPixels)
      .catch(() => toast.show('Erro ao carregar imagem-alvo'));
  }, [imagem]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => {
    return () => { sseCleanup.current?.(); };
  }, []);

  // Build layer sizes from config
  const layerSizes = (() => {
    const nLayers = parseInt(hiddenLayers);
    const nNeurons = parseInt(neuronsPerLayer);
    const sizes = [2]; // input: (x, y)
    for (let i = 0; i < nLayers; i++) sizes.push(nNeurons);
    sizes.push(3); // output: (R, G, B)
    return sizes;
  })();

  // ---------- Actions ----------

  const addLog = useCallback((msg: string) => {
    setLogLines(prev => [...prev.slice(-200), msg]);
  }, []);

  const handleTrain = useCallback(async () => {
    if (training) return;

    const cfg: ImgregConfig = {
      hiddenLayers: parseInt(hiddenLayers),
      neuronsPerLayer: parseInt(neuronsPerLayer),
      learningRate: parseFloat(learningRate),
      imagem,
      maxEpocas: parseInt(maxEpocas),
    };
    if (variant === 'minibatch') {
      cfg.batchSize = parseInt(batchSize);
      cfg.numWorkers = parseInt(numWorkers);
    }

    try {
      await apiPost(`${prefix}/config`, cfg);
    } catch (e) {
      toast.show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      return;
    }

    setTraining(true);
    setStatus('treinando...');
    setStatusColor('var(--cyan)');
    setOutputPixels(null);
    setLossHistory([]);
    setLogLines([]);
    setActiveLayer(-1);
    addLog(`[config] ${variant} | ${imagem} | ${hiddenLayers}L x ${neuronsPerLayer}N | lr=${learningRate} | epocas=${maxEpocas}`);

    sseCleanup.current = apiSSE(`${prefix}/train`, {
      onMessage(data) {
        const step = data as ImgregStep;

        // Check if this is the final "done" step
        if (step.done) {
          // Close SSE before it triggers onerror
          if (sseCleanup.current) { sseCleanup.current(); sseCleanup.current = null; }
          setEpoca(step.epoca.toLocaleString());
          setLoss(step.loss.toFixed(6));
          if (step.elapsedMs != null) {
            setTempo(`${step.elapsedMs}ms`);
          }
          if (step.outputPixels) {
            setOutputPixels(step.outputPixels);
          }
          if (step.lossHistorico) {
            setLossHistory(step.lossHistorico);
          }
          setActiveLayer(-1);
          setStatus(step.convergiu ? 'convergiu' : 'concluido');
          setStatusColor(step.convergiu ? 'var(--primary-glow)' : 'var(--on-surface)');
          setTraining(false);
          addLog(`[done] epoca=${step.epoca} loss=${step.loss.toFixed(6)} ${step.convergiu ? 'CONVERGIU' : 'limite atingido'}`);
          toast.show('Treinamento concluido');
          return;
        }

        // Progress step
        setEpoca(step.epoca.toLocaleString());
        setLoss(step.loss.toFixed(6));
        setActiveLayer(step.activeLayer);
        if (step.elapsedMs != null) {
          setTempo(`${step.elapsedMs}ms`);
        }
        if (step.outputPixels) {
          setOutputPixels(step.outputPixels);
        }
        setLossHistory(prev => [...prev, step.loss]);
        addLog(`[ep ${step.epoca}] loss=${step.loss.toFixed(6)}`);
      },
      onDone(data) {
        // Some variants might send event: done
        const step = data as ImgregStep;
        setEpoca(step.epoca.toLocaleString());
        setLoss(step.loss.toFixed(6));
        if (step.outputPixels) setOutputPixels(step.outputPixels);
        if (step.lossHistorico) setLossHistory(step.lossHistorico);
        setActiveLayer(-1);
        setStatus(step.convergiu ? 'convergiu' : 'concluido');
        setStatusColor(step.convergiu ? 'var(--primary-glow)' : 'var(--pink)');
        setTraining(false);
        sseCleanup.current = null;
        toast.show('Treinamento concluido');
      },
      onError() {
        setTraining(false);
        setActiveLayer(-1);
        setStatus('erro de conexao');
        setStatusColor('var(--pink)');
        sseCleanup.current = null;
        addLog('[erro] conexao perdida');
      },
    });
  }, [training, hiddenLayers, neuronsPerLayer, learningRate, imagem, maxEpocas, batchSize, numWorkers, variant, prefix, toast, addLog]);

  const handleReset = useCallback(async () => {
    sseCleanup.current?.();
    sseCleanup.current = null;

    try {
      await apiPost(`${prefix}/reset`);
    } catch {
      // ignore
    }

    setTraining(false);
    setEpoca('\u2014');
    setLoss('\u2014');
    setTempo('\u2014');
    setStatus('aguardando');
    setStatusColor('var(--on-surface)');
    setActiveLayer(-1);
    setOutputPixels(null);
    setLossHistory([]);
    setLogLines([]);
    toast.show('Resetado');
  }, [prefix, toast]);

  // Progress percentage
  const maxEp = parseInt(maxEpocas);
  const currentEp = epoca !== '\u2014' ? parseInt(epoca.replace(/,/g, '')) : 0;
  const progressPct = maxEp > 0 ? Math.min(100, (currentEp / maxEp) * 100) : 0;

  return (
    <div>
      {/* ===== Page Header ===== */}
      <div className="page-header">
        <div>
          <div className="page-title">Image <span>Regression ({VARIANT_LABEL[variant]})</span></div>
          <div className="page-sub">{VARIANT_SUB[variant]}</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={training}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            INICIAR
          </button>
        </div>
      </div>

      {/* ===== Config Panel ===== */}
      <div className={variant === 'minibatch' ? 'grid-2' : 'grid-3'} style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Imagem"
            options={IMAGENS}
            value={imagem}
            onChange={setImagem}
            style={{ width: '100%' }}
          />
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Camadas &middot; Neuronios</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select
              options={LAYERS}
              value={hiddenLayers}
              onChange={setHiddenLayers}
              style={{ flex: 1 }}
            />
            <Select
              options={NEURONS}
              value={neuronsPerLayer}
              onChange={setNeuronsPerLayer}
              style={{ flex: 1 }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Learning Rate &middot; Epocas</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select
              options={LR_OPTIONS}
              value={learningRate}
              onChange={setLearningRate}
              style={{ flex: 1 }}
            />
            <Select
              options={EPOCAS_OPTIONS}
              value={maxEpocas}
              onChange={setMaxEpocas}
              style={{ flex: 1 }}
            />
          </div>
        </Card>

        {variant === 'minibatch' && (
          <Card style={{ padding: '16px 20px' }}>
            <div className="imgreg-select-label">Batch Size &middot; Workers</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <Select
                options={BATCH_OPTIONS}
                value={batchSize}
                onChange={setBatchSize}
                style={{ flex: 1 }}
              />
              <Select
                options={WORKERS_OPTIONS}
                value={numWorkers}
                onChange={setNumWorkers}
                style={{ flex: 1 }}
              />
            </div>
          </Card>
        )}
      </div>

      {/* ===== Stat Bar ===== */}
      <div className="imgreg-stat-bar">
        <span>
          <span className="stat-key">epoca </span>
          <span className="stat-val">{epoca}</span>
          <span className="stat-key"> / {maxEpocas}</span>
        </span>
        <span>
          <span className="stat-key">loss </span>
          <span className="stat-val cyan">{loss}</span>
        </span>
        <span>
          <span className="stat-key">tempo </span>
          <span className="stat-val">{tempo}</span>
        </span>
        <span>
          <span className="stat-key">status </span>
          <span className="stat-val" style={{ color: statusColor }}>{status}</span>
        </span>
      </div>

      {/* ===== Progress Bar ===== */}
      {training && (
        <div className="imgreg-progress-bar" style={{ marginBottom: 16 }}>
          <div className="imgreg-progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      )}

      {/* ===== 3-Panel Layout: Target | Network | Output ===== */}
      <div className="imgreg-center-panel" style={{ marginBottom: 16, flexWrap: 'wrap' }}>
        {/* Target Image */}
        <div className="imgreg-canvas-wrap">
          <div className="imgreg-label">target</div>
          <PixelGrid16 pixels={targetPixels} />
        </div>

        {/* Network Viz */}
        <Card style={{ flex: 1, minWidth: 300 }}>
          <NetworkViz
            layerSizes={layerSizes}
            activeLayer={activeLayer}
            hudText={`${variant} | ${imagem}`}
            animate={!training}
          />
        </Card>

        {/* MLP Output */}
        <div className="imgreg-canvas-wrap">
          <div className="imgreg-label">mlp output</div>
          <PixelGrid16 pixels={outputPixels} />
        </div>
      </div>

      {/* ===== Metrics ===== */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Epoca"
          value={epoca}
          label={`de ${maxEpocas}`}
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Loss"
          value={loss}
          label="MSE"
          color="cyan"
        />
        <MetricCard
          title="Status"
          value={status}
          label={variant}
          valueStyle={{ fontSize: 18, color: statusColor }}
        />
      </div>

      {/* ===== Error Chart ===== */}
      <Card title="Curva de Erro" style={{ marginBottom: 16 }}>
        <LogChart data={lossHistory} color="#00ccff" />
      </Card>

      {/* ===== Log ===== */}
      <Card title="Log" style={{ marginBottom: 16 }}>
        <div className="log-panel" ref={logRef}>
          {logLines.map((line, i) => (
            <div key={i} className={`log-line${i < logLines.length - 3 ? ' dim' : ''}`}>{line}</div>
          ))}
          {logLines.length === 0 && (
            <div className="log-line dim">aguardando inicio do treinamento...</div>
          )}
        </div>
      </Card>

      {/* ===== Details ===== */}
      <Card title="Detalhes">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          Regressao de imagem 16x16 usando MLP. A rede aprende a mapear coordenadas (x,y) para cores RGB,
          reconstruindo a imagem pixel a pixel. Demonstra o Teorema da Aproximacao Universal.
          <br /><br />
          <b>Arquitetura:</b> 2 entradas (x,y) &middot; {hiddenLayers} camadas ocultas
          de {neuronsPerLayer} neuronios (ReLU) &middot; 3 saidas (R,G,B) (Sigmoid) &middot; He init
          {variant === 'minibatch' && (
            <>
              <br />
              <b>Mini-batch:</b> batch size = {batchSize} &middot; {numWorkers} workers
            </>
          )}
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 16x16 pixel grid sub-component
// ---------------------------------------------------------------------------

function PixelGrid16({ pixels }: { pixels: [number, number, number][] | null }) {
  if (!pixels || pixels.length === 0) {
    return (
      <div className="imgreg-grid">
        {Array.from({ length: 256 }, (_, i) => (
          <div key={i} className="imgreg-pixel" style={{ backgroundColor: '#1c2026' }} />
        ))}
      </div>
    );
  }

  return (
    <div className="imgreg-grid">
      {pixels.slice(0, 256).map((rgb, i) => (
        <div
          key={i}
          className="imgreg-pixel"
          style={{ backgroundColor: rgbToHex(rgb) }}
        />
      ))}
    </div>
  );
}
