import { useState, useRef, useCallback, useEffect } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import PixelGrid from '../components/shared/PixelGrid';
import LogChart from '../components/viz/LogChart';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import CnnMegaAnimation from '../components/viz/CnnMegaAnimation';
import type { CnnResult, CnnStep, CnnClassifyResp, CnnVisualizeResp, CnnModelMeta } from '../api/types';

// ---------------------------------------------------------------------------
// FeatureMapGrid — renderiza um feature map como heatmap cyan
// ---------------------------------------------------------------------------

function FeatureMapGrid({ data, size, label }: { data: number[][]; size: number; label?: string }) {
  // Find min/max for normalization
  let min = Infinity, max = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;

  return (
    <div style={{ display: 'inline-block', textAlign: 'center' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${data[0]?.length || 0}, ${size}px)`,
          gap: 0,
        }}
      >
        {data.flatMap((row, r) =>
          row.map((v, c) => {
            const norm = (v - min) / range;
            const brightness = Math.round(norm * 255);
            return (
              <div
                key={`${r}-${c}`}
                style={{
                  width: size,
                  height: size,
                  background: `rgb(0, ${brightness}, ${Math.round(brightness * 0.8)})`,
                }}
              />
            );
          })
        )}
      </div>
      {label && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--on-surface)', marginTop: 2 }}>
          {label}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// KernelGrid — renderiza um kernel 3x3
// ---------------------------------------------------------------------------

function KernelGrid({ data }: { data: number[][] }) {
  let min = Infinity, max = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${data[0].length}, 16px)`, gap: 1 }}>
      {data.flatMap((row, r) =>
        row.map((v, c) => {
          const norm = (v - min) / range;
          return (
            <div
              key={`${r}-${c}`}
              style={{
                width: 16,
                height: 16,
                background: v >= 0
                  ? `rgba(0, 255, 0, ${norm.toFixed(2)})`
                  : `rgba(255, 0, 127, ${(1 - norm).toFixed(2)})`,
                border: '1px solid #333',
              }}
            />
          );
        })
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pipeline Diagram — 2D with hover tooltips
// ---------------------------------------------------------------------------

const PIPELINE_BLOCKS = [
  { label: 'Input',      sub: '28×28×1',  color: '#00fbfb', bg: '#0a1a2a',
    tip: 'Imagem grayscale 28×28 pixels normalizada [0,1]. Cada pixel é um valor de intensidade.' },
  { label: 'Conv1+ReLU', sub: '26×26×8',  color: '#4488ff', bg: '#0a1a3a',
    tip: '8 filtros 3×3 deslizam sobre a imagem extraindo features (bordas, cantos). ReLU zera valores negativos.' },
  { label: 'MaxPool',    sub: '13×13×8',  color: '#ff6ec7', bg: '#2a0a1a',
    tip: 'Reduz pela metade: pega o valor máximo de cada janela 2×2. Mantém as features mais fortes.' },
  { label: 'Conv2+ReLU', sub: '11×11×16', color: '#4488ff', bg: '#0a1a3a',
    tip: '16 filtros 3×3 combinam as features do Conv1 em padrões mais complexos (partes de letras).' },
  { label: 'MaxPool',    sub: '5×5×16',   color: '#ff6ec7', bg: '#2a0a1a',
    tip: 'Outra redução 2×2. Agora temos 16 mapas de 5×5 — representação compacta da imagem.' },
  { label: 'Flatten',    sub: '400',      color: '#888',    bg: '#1a1a2a',
    tip: 'Achata os 16×5×5 = 400 valores em um vetor 1D para alimentar a rede densa.' },
  { label: 'Dense+ReLU', sub: '64',       color: '#00ff00', bg: '#0a2a0a',
    tip: 'Camada totalmente conectada: 400→64 neurônios. Aprende combinações das features para classificação.' },
  { label: 'Dense',      sub: '26',       color: '#00ff00', bg: '#0a2a0a',
    tip: 'Camada final: 64→26 neurônios (um por letra A-Z). Produz scores brutos (logits).' },
  { label: 'Softmax',    sub: 'A-Z',      color: '#00ff00', bg: '#002200',
    tip: 'Converte logits em probabilidades (somam 100%). A letra com maior probabilidade é a classificação.' },
];

function PipelineDiagram({ animStep }: { animStep: number }) {
  const [hover, setHover] = useState<number | null>(null);

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ overflowX: 'auto', padding: '12px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 0, minWidth: 860 }}>
          {PIPELINE_BLOCKS.map((b, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
              {/* Block */}
              <div
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                style={{
                  textAlign: 'center', padding: '10px 14px', cursor: 'pointer',
                  background: b.bg, border: `2px solid ${b.color}`,
                  borderRadius: 4, minWidth: 70,
                  transition: 'all 0.3s',
                  opacity: animStep < 0 || animStep === i ? 1 : 0.3,
                  boxShadow: (animStep === i || hover === i) ? `0 0 12px ${b.color}60` : 'none',
                  transform: hover === i ? 'scale(1.05)' : 'scale(1)',
                }}
              >
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: b.color, fontWeight: 700 }}>
                  {b.label}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: '#888', marginTop: 2 }}>
                  {b.sub}
                </div>
              </div>
              {/* Arrow */}
              {i < PIPELINE_BLOCKS.length - 1 && (
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 14, color: '#555', padding: '0 6px',
                  transition: 'all 0.3s', opacity: animStep < 0 || animStep <= i + 1 ? 1 : 0.3,
                }}>→</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Tooltip */}
      {hover !== null && (
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)',
          background: 'var(--surface-high)', border: `1px solid ${PIPELINE_BLOCKS[hover].color}40`,
          padding: '8px 14px', marginTop: 4, lineHeight: 1.6, maxWidth: 500,
          boxShadow: `0 0 8px ${PIPELINE_BLOCKS[hover].color}20`,
        }}>
          <span style={{ color: PIPELINE_BLOCKS[hover].color, fontWeight: 700 }}>
            {PIPELINE_BLOCKS[hover].label}:
          </span>{' '}
          {PIPELINE_BLOCKS[hover].tip}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function CnnView() {
  const { show } = useToast();

  // Config
  const [alfa, setAlfa] = useState(0.001);
  const [maxEpocas, setMaxEpocas] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [trainLimit, setTrainLimit] = useState(10000);

  // Training state
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<CnnResult | null>(null);
  const [lossHist, setLossHist] = useState<number[]>([]);
  const sseCleanup = useRef<(() => void) | null>(null);

  // Log panel
  const [logLines, setLogLines] = useState<{ msg: string; cls: string }[]>([
    { msg: '// aguardando inicio do treinamento...', cls: 'dim' },
  ]);
  const [progress, setProgress] = useState(0);

  // Interactive test
  const [testPixels, setTestPixels] = useState<number[]>(new Array(784).fill(-1));
  const [classifyResp, setClassifyResp] = useState<CnnClassifyResp | null>(null);

  // Visualization
  const [vizData, setVizData] = useState<CnnVisualizeResp | null>(null);
  const [vizFilter1, setVizFilter1] = useState(0);
  const [vizFilter2, setVizFilter2] = useState(0);
  const [animStep, setAnimStep] = useState(-1); // -1 = não animando, 0-7 = etapa ativa
  const animRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Mega animation
  const [showMegaAnim, setShowMegaAnim] = useState(false);

  // Models
  const [models, setModels] = useState<CnnModelMeta[]>([]);
  const [saveName, setSaveName] = useState('');

  const trained = result !== null;

  // Load models list on mount
  useEffect(() => {
    apiGet<CnnModelMeta[]>('/cnn/models').then(setModels).catch(() => {});
  }, []);

  const refreshModels = useCallback(() => {
    apiGet<CnnModelMeta[]>('/cnn/models').then(setModels).catch(() => {});
  }, []);

  const addLog = useCallback((msg: string, cls: string = '') => {
    setLogLines(prev => {
      const next = [{ msg, cls }, ...prev];
      return next.length > 80 ? next.slice(0, 80) : next;
    });
  }, []);

  // ---------- Train ----------

  const handleTrain = useCallback(async () => {
    if (training) return;
    setTraining(true);
    setResult(null);
    setLossHist([]);
    setLogLines([]);
    setProgress(0);
    setVizData(null);
    addLog('// iniciando treinamento CNN EMNIST Letters...', 'dim');
    addLog(`config: lr=${alfa} epocas=${maxEpocas} batch=${batchSize} limit=${trainLimit}`, 'dim');

    try {
      await apiPost('/cnn/config', { alfa, maxEpocas, batchSize, trainLimit });
    } catch (e) {
      addLog('// erro ao configurar: ' + (e instanceof Error ? e.message : String(e)), 'err');
      setTraining(false);
      return;
    }

    const cleanup = apiSSE('/cnn/train', {
      onMessage(data) {
        const step = data as CnnStep;
        const pct = ((step.epoca - 1) / maxEpocas + (step.batch / step.totalBatch) / maxEpocas) * 100;
        setProgress(Math.min(pct, 99));
        if (step.batch === step.totalBatch || step.batch % 100 === 0) {
          addLog(
            `epoca ${step.epoca}/${maxEpocas} · batch ${step.batch}/${step.totalBatch} · loss ${step.loss.toFixed(4)} · acc ${step.acuracia.toFixed(1)}%`,
            'ok'
          );
        }
      },
      onDone(data) {
        const res = data as CnnResult;
        setResult(res);
        setLossHist(res.lossHistorico || []);
        setProgress(100);
        setTraining(false);
        sseCleanup.current = null;
        addLog(
          `✓ concluido: treino ${res.acuracia.toFixed(1)}% · teste ${res.acuraciaTest.toFixed(1)}% · ${(res.tempoMs / 1000).toFixed(1)}s`,
          'ok'
        );
        show(`CNN treinada — teste ${res.acuraciaTest.toFixed(1)}%`);
      },
      onError() {
        addLog('// erro de conexao', 'err');
        setTraining(false);
        sseCleanup.current = null;
      },
    });
    sseCleanup.current = cleanup;
  }, [training, alfa, maxEpocas, batchSize, trainLimit, addLog, show]);

  // ---------- Reset ----------

  const handleReset = useCallback(async () => {
    sseCleanup.current?.();
    try {
      await apiPost('/cnn/reset');
      setResult(null);
      setLossHist([]);
      setTraining(false);
      setLogLines([{ msg: '// aguardando inicio do treinamento...', cls: 'dim' }]);
      setProgress(0);
      setClassifyResp(null);
      setTestPixels(new Array(784).fill(-1));
      setVizData(null);
      show('CNN resetada');
    } catch {
      show('Erro ao resetar');
    }
  }, [show]);

  // ---------- Classify ----------

  const classifyTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const autoClassify = useCallback((pixels: number[]) => {
    setTestPixels(pixels);
    if (classifyTimeout.current) clearTimeout(classifyTimeout.current);
    if (trained && pixels.some(v => v === 1)) {
      classifyTimeout.current = setTimeout(async () => {
        const normalized = pixels.map(v => v === 1 ? 1.0 : 0.0);
        try {
          const resp = await apiPost<CnnClassifyResp>('/cnn/classify', { pixels: normalized });
          setClassifyResp(resp);
        } catch { /* ignore */ }
      }, 150);
    } else {
      setClassifyResp(null);
    }
  }, [trained]);

  // ---------- Visualize ----------

  const handleVisualize = useCallback(async () => {
    if (!trained) return;
    const normalized = testPixels.map(v => v === 1 ? 1.0 : 0.0);
    if (!normalized.some(v => v > 0)) {
      show('Desenhe algo primeiro');
      return;
    }
    try {
      const resp = await apiPost<CnnVisualizeResp>('/cnn/visualize', { pixels: normalized });
      setVizData(resp);
    } catch {
      show('Erro ao visualizar');
    }
  }, [trained, testPixels, show]);

  // ---------- Animation ----------

  const handlePlayAnimation = useCallback(() => {
    if (animRef.current) {
      clearInterval(animRef.current);
      animRef.current = undefined;
      setAnimStep(-1);
      return;
    }
    let step = 0;
    setAnimStep(0);
    animRef.current = setInterval(() => {
      step++;
      if (step > 8) {
        clearInterval(animRef.current);
        animRef.current = undefined;
        // Keep last step visible for a moment then reset
        setTimeout(() => setAnimStep(-1), 1500);
        return;
      }
      setAnimStep(step);
    }, 800);
  }, []);

  // ---------- Save/Load ----------

  const handleSave = useCallback(async () => {
    try {
      await apiPost('/cnn/save', { nome: saveName || 'Sem nome' });
      setSaveName('');
      refreshModels();
      show('Modelo salvo');
    } catch {
      show('Erro ao salvar');
    }
  }, [saveName, refreshModels, show]);

  const handleLoad = useCallback(async (id: string) => {
    try {
      const res = await apiPost<CnnResult>('/cnn/load', { id });
      setResult(res);
      setLossHist(res.lossHistorico || []);
      setClassifyResp(null);
      setVizData(null);
      setTestPixels(new Array(784).fill(-1));
      addLog(`✓ modelo carregado: teste ${res.acuraciaTest.toFixed(1)}%`, 'ok');
      show('Modelo carregado');
    } catch {
      show('Erro ao carregar');
    }
  }, [addLog, show]);

  const handleDeleteModel = useCallback(async (id: string) => {
    try {
      await apiPost('/cnn/delete-model', { id });
      refreshModels();
      show('Modelo deletado');
    } catch {
      show('Erro ao deletar');
    }
  }, [refreshModels, show]);

  // ---------- Render ----------

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">CNN <span>EMNIST Letters</span></div>
          <div className="page-sub">
            Rede Neural Convolucional para classificação A-Z &mdash; Aula 07
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost" onClick={handleReset} disabled={training}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            TREINAR
          </button>
        </div>
      </div>

      {/* Config */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <Card title="Learning Rate">
          <div className="porta-chips">
            {[0.0005, 0.001, 0.002, 0.005].map(a => (
              <button key={a} className={`porta-chip${alfa === a ? ' selected' : ''}`}
                onClick={() => setAlfa(a)} disabled={training}>{a}</button>
            ))}
          </div>
        </Card>
        <Card title="Épocas · Batch Size">
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            {[5, 10, 20].map(e => (
              <button key={e} className={`porta-chip${maxEpocas === e ? ' selected' : ''}`}
                onClick={() => setMaxEpocas(e)} disabled={training}>{e} épocas</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {[16, 32, 64].map(b => (
              <button key={b} className={`porta-chip${batchSize === b ? ' selected' : ''}`}
                onClick={() => setBatchSize(b)} disabled={training}>batch {b}</button>
            ))}
          </div>
        </Card>
        <Card title="Amostras de Treino">
          <div className="porta-chips">
            {[5000, 10000, 30000, 0].map(n => (
              <button key={n} className={`porta-chip${trainLimit === n ? ' selected' : ''}`}
                onClick={() => setTrainLimit(n)} disabled={training}>
                {n === 0 ? 'Todas (~88k)' : `${(n / 1000).toFixed(0)}k`}
              </button>
            ))}
          </div>
        </Card>
      </div>

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <MetricCard title="Épocas" value={result ? `${result.epocas}` : '--'}
          label={result ? `${(result.tempoMs / 1000).toFixed(1)}s total` : 'aguardando'} color="cyan" pulse={training} />
        <MetricCard title="Loss" value={result ? result.lossFinal.toFixed(4) : '--'}
          label="cross-entropy" color="green" />
        <MetricCard title="Acurácia Teste" value={result ? `${result.acuraciaTest.toFixed(1)}%` : '--'}
          label={result ? `treino: ${result.acuracia.toFixed(1)}%` : 'aguardando'} color={result ? 'green' : undefined} />
      </div>

      {/* Log + Loss Chart */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <Card title="Log de Treinamento">
          <div className="progress-wrap">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="log-panel">
            {logLines.map((line, i) => (
              <div key={i} className={`log-line ${line.cls}`}>{line.msg}</div>
            ))}
          </div>
        </Card>
        <Card title="Curva de Loss — escala log">
          <LogChart data={lossHist} color="#00fbfb" />
        </Card>
      </div>

      {/* Models Panel */}
      <Card title="Modelos Salvos" style={{ marginBottom: 24 }}>
        {trained && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, alignItems: 'center' }}>
            <input
              type="text"
              placeholder="Nome do modelo (opcional)"
              value={saveName}
              onChange={e => setSaveName(e.target.value)}
              style={{
                background: 'var(--surface-low)', border: '1px solid var(--border)', color: 'var(--on-surface)',
                fontFamily: 'var(--font-mono)', fontSize: 11, padding: '6px 10px', flex: 1,
              }}
            />
            <button className="btn btn-primary" style={{ fontSize: 11, padding: '6px 14px' }} onClick={handleSave}>
              SALVAR
            </button>
          </div>
        )}
        {models.length === 0 ? (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', padding: '8px 0' }}>
            Nenhum modelo salvo
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {models.map(m => (
              <div key={m.id} style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px',
                background: 'var(--surface-low)', fontFamily: 'var(--font-mono)', fontSize: 11,
              }}>
                <span style={{ color: 'var(--cyan)', fontWeight: 700, minWidth: 120 }}>{m.id}</span>
                <span style={{ color: 'var(--on-surface)', flex: 1 }}>{m.nome || '—'}</span>
                <span style={{ color: 'var(--primary-glow)' }}>teste {m.acuraciaTest.toFixed(1)}%</span>
                <span style={{ color: 'var(--on-surface)' }}>{m.epocas}ep</span>
                <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px' }}
                  onClick={() => handleLoad(m.id)}>CARREGAR</button>
                <button className="btn btn-ghost" style={{ fontSize: 10, padding: '3px 8px', color: 'var(--pink)' }}
                  onClick={() => handleDeleteModel(m.id)}>×</button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Interactive Test + Classify */}
      {trained && (
        <Card title="Teste Interativo — Desenhe uma letra (28×28)" style={{ marginBottom: 24 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'start' }}>
            <div>
              <PixelGrid rows={28} cols={28} cellSize={12} gap={1}
                values={testPixels} onChange={autoClassify} showClear={false} />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button className="btn btn-ghost" style={{ fontSize: 10, padding: '6px 12px' }}
                  onClick={() => { setTestPixels(new Array(784).fill(-1)); setClassifyResp(null); setVizData(null); }}>
                  LIMPAR
                </button>
                <button className="btn" style={{ fontSize: 10, padding: '6px 12px' }} onClick={handleVisualize}>
                  VISUALIZAR ETAPAS
                </button>
                {vizData && (
                  <button className="btn btn-primary" style={{ fontSize: 10, padding: '6px 12px' }}
                    onClick={() => setShowMegaAnim(true)}>
                    ANIMAÇÃO PASSO A PASSO
                  </button>
                )}
              </div>
            </div>
            <div>
              {classifyResp ? (
                <>
                  <div className="result-big">{classifyResp.letra}</div>
                  <div className="result-label" style={{ marginBottom: 16 }}>
                    confiança: {(classifyResp.top5[0].score * 100).toFixed(1)}%
                  </div>
                  <div className="conf-list">
                    {classifyResp.top5.map((c, i) => (
                      <div className="conf-row" key={i}>
                        <span className="conf-letter">{c.letra}</span>
                        <div className="conf-bar-wrap">
                          <div className="conf-bar-fill" style={{ width: `${(c.score * 100).toFixed(1)}%` }} />
                        </div>
                        <span className="conf-score">{(c.score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty">
                  <div className="empty-icon">?</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)' }}>
                    Desenhe uma letra no grid
                  </div>
                </div>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* ===== Mega Animation ===== */}
      {showMegaAnim && vizData && (
        <CnnMegaAnimation vizData={vizData} onClose={() => setShowMegaAnim(false)} />
      )}

      {/* ===== CNN Pipeline Visualization ===== */}
      {vizData && (
        <Card title="Visualização das Etapas da CNN" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <button className="btn" style={{ fontSize: 10, padding: '6px 14px' }} onClick={handlePlayAnimation}>
              {animRef.current ? '⏹ PARAR' : '▶ PLAY ANIMAÇÃO'}
            </button>
            {animStep >= 0 && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--primary-glow)', alignSelf: 'center' }}>
                Etapa {animStep + 1}/9: {['Input', 'Conv1+ReLU', 'MaxPool 2×2', 'Conv2+ReLU', 'MaxPool 2×2', 'Flatten', 'Dense+ReLU (64)', 'Dense (26)', 'Softmax'][animStep]}
              </span>
            )}
          </div>

          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'flex', gap: 16, alignItems: 'center', minWidth: 1100, padding: '8px 16px' }}>

              {/* 0: INPUT */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 0 ? 1 : 0.3,
                boxShadow: animStep === 0 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginBottom: 4 }}>INPUT 28×28</div>
                <FeatureMapGrid data={
                  Array.from({ length: 28 }, (_, r) => Array.from({ length: 28 }, (_, c) => vizData.input[r * 28 + c]))
                } size={4} />
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 1 ? 1 : 0.3 }}>→</div>

              {/* 1: CONV1 + kernel */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 1 ? 1 : 0.3,
                boxShadow: animStep === 1 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginBottom: 4 }}>
                  CONV1 26×26 (filtro {vizFilter1 + 1}/8)
                </div>
                <FeatureMapGrid data={vizData.conv1Maps[vizFilter1]} size={4} />
                <div style={{ display: 'flex', gap: 2, marginTop: 4, justifyContent: 'center' }}>
                  {Array.from({ length: 8 }, (_, i) => (
                    <button key={i} className={`porta-chip${vizFilter1 === i ? ' selected' : ''}`}
                      style={{ padding: '2px 6px', fontSize: 9, minWidth: 0 }}
                      onClick={() => setVizFilter1(i)}>{i + 1}</button>
                  ))}
                </div>
                <div style={{ marginTop: 6 }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--on-surface)', marginBottom: 2 }}>kernel 3×3:</div>
                  <KernelGrid data={vizData.filters1[vizFilter1][0]} />
                </div>
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 2 ? 1 : 0.3 }}>→</div>

              {/* 2: POOL1 */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 2 ? 1 : 0.3,
                boxShadow: animStep === 2 ? '0 0 16px rgba(255,0,127,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--pink)', marginBottom: 4 }}>POOL1 13×13</div>
                <FeatureMapGrid data={vizData.pool1Maps[vizFilter1]} size={5} />
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 3 ? 1 : 0.3 }}>→</div>

              {/* 3: CONV2 + kernel */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 3 ? 1 : 0.3,
                boxShadow: animStep === 3 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginBottom: 4 }}>
                  CONV2 11×11 (filtro {vizFilter2 + 1}/16)
                </div>
                <FeatureMapGrid data={vizData.conv2Maps[vizFilter2]} size={5} />
                <div style={{ display: 'flex', gap: 2, marginTop: 4, justifyContent: 'center', flexWrap: 'wrap' }}>
                  {Array.from({ length: 16 }, (_, i) => (
                    <button key={i} className={`porta-chip${vizFilter2 === i ? ' selected' : ''}`}
                      style={{ padding: '2px 5px', fontSize: 8, minWidth: 0 }}
                      onClick={() => setVizFilter2(i)}>{i + 1}</button>
                  ))}
                </div>
                <div style={{ marginTop: 6 }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--on-surface)', marginBottom: 2 }}>kernel 3×3 (ch0):</div>
                  <KernelGrid data={vizData.filters2[vizFilter2][0]} />
                </div>
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 4 ? 1 : 0.3 }}>→</div>

              {/* 4: POOL2 */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 4 ? 1 : 0.3,
                boxShadow: animStep === 4 ? '0 0 16px rgba(255,0,127,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--pink)', marginBottom: 4 }}>POOL2 5×5</div>
                <FeatureMapGrid data={vizData.pool2Maps[vizFilter2]} size={8} />
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 5 ? 1 : 0.3 }}>→</div>

              {/* 5: FLATTEN */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 5 ? 1 : 0.3,
                boxShadow: animStep === 5 ? '0 0 16px rgba(0,251,251,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--cyan)', marginBottom: 4 }}>FLATTEN</div>
                <div style={{ width: 10, height: 60, background: 'linear-gradient(180deg, #00fbfb30, #00fbfb10)', border: '1px solid #00fbfb40', margin: '0 auto' }} />
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-surface)', marginTop: 2 }}>400</div>
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 6 ? 1 : 0.3 }}>→</div>

              {/* 6: DENSE1 (400→64) */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 6 ? 1 : 0.3,
                boxShadow: animStep === 6 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)', marginBottom: 4 }}>DENSE+ReLU</div>
                <div style={{ width: 10, height: 40, background: 'linear-gradient(180deg, #00ff0030, #00ff0010)', border: '1px solid #00ff0040', margin: '0 auto' }} />
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-surface)', marginTop: 2 }}>64</div>
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 7 ? 1 : 0.3 }}>→</div>

              {/* 7: DENSE2 (64→26) */}
              <div style={{ textAlign: 'center', transition: 'all 0.4s', opacity: animStep < 0 || animStep === 7 ? 1 : 0.3,
                boxShadow: animStep === 7 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)', marginBottom: 4 }}>DENSE</div>
                <div style={{ width: 10, height: 24, background: 'linear-gradient(180deg, #00ff0030, #00ff0010)', border: '1px solid #00ff0040', margin: '0 auto' }} />
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--on-surface)', marginTop: 2 }}>26</div>
              </div>

              <div style={{ color: 'var(--primary-glow)', fontSize: 16, transition: 'all 0.4s', opacity: animStep < 0 || animStep <= 8 ? 1 : 0.3 }}>→</div>

              {/* 8: SOFTMAX RESULT */}
              <div style={{ textAlign: 'center', minWidth: 60, transition: 'all 0.4s',
                opacity: animStep < 0 || animStep === 8 ? 1 : 0.3,
                boxShadow: animStep === 8 ? '0 0 16px rgba(0,255,0,0.4)' : 'none', padding: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--primary-glow)', marginBottom: 4 }}>SOFTMAX</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 32, fontWeight: 700, color: 'var(--primary-glow)',
                  transition: 'all 0.3s', transform: animStep === 8 ? 'scale(1.2)' : 'scale(1)' }}>
                  {vizData.letra}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--on-surface)' }}>
                  {(vizData.top5[0].score * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* ===== 2D Architecture Pipeline ===== */}
      <Card title="Arquitetura CNN — Pipeline" style={{ marginBottom: 24 }}>
        <PipelineDiagram animStep={animStep} />
        <div style={{ display: 'flex', justifyContent: 'center', gap: 24, fontFamily: 'var(--font-mono)', fontSize: 10, marginTop: 8 }}>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#0a1a3a', border: '1.5px solid #4488ff', marginRight: 6, verticalAlign: 'middle' }} />Conv+ReLU</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#2a0a1a', border: '1.5px solid #ff6ec7', marginRight: 6, verticalAlign: 'middle' }} />MaxPool</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#0a2a0a', border: '1.5px solid #00ff00', marginRight: 6, verticalAlign: 'middle' }} />Dense/Softmax</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#1a1a2a', border: '1.5px solid #888', marginRight: 6, verticalAlign: 'middle' }} />Flatten</span>
        </div>
      </Card>

      {/* Architecture Details */}
      <Card title="Detalhes">
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--on-surface)', lineHeight: 1.8 }}>
          <div><b style={{ color: 'var(--cyan)' }}>Arquitetura:</b> Conv1(8×3×3) → Pool → Conv2(16×3×3) → Pool → Dense(400→64) → Dense(64→26)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Ativação:</b> ReLU (conv + oculta), Softmax (saída)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Loss:</b> Cross-Entropy</div>
          <div><b style={{ color: 'var(--cyan)' }}>Dataset:</b> EMNIST Letters (derivado do NIST SD19) &mdash; 28×28 grayscale, 26 classes (A-Z)</div>
          <div><b style={{ color: 'var(--cyan)' }}>Inicialização:</b> He init (√(2/fan_in)) para ReLU</div>
          <div><b style={{ color: 'var(--cyan)' }}>Referência:</b> Aula 07 &mdash; RNA Convolucional, Manzan 2026</div>
        </div>
      </Card>
    </div>
  );
}
