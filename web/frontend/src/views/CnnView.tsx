import { useState, useRef, useCallback } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import PixelGrid from '../components/shared/PixelGrid';
import LogChart from '../components/viz/LogChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type { CnnResult, CnnStep, CnnClassifyResp } from '../api/types';

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

  // Interactive test (28x28 grayscale grid)
  const [testPixels, setTestPixels] = useState<number[]>(new Array(784).fill(-1));
  const [classifyResp, setClassifyResp] = useState<CnnClassifyResp | null>(null);

  const trained = result !== null;

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
        // Convert bipolar (-1/1) to grayscale (0/1) for CNN
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
              <button
                key={a}
                className={`porta-chip${alfa === a ? ' selected' : ''}`}
                onClick={() => setAlfa(a)}
                disabled={training}
              >{a}</button>
            ))}
          </div>
        </Card>
        <Card title="Épocas · Batch Size">
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            {[5, 10, 20].map(e => (
              <button
                key={e}
                className={`porta-chip${maxEpocas === e ? ' selected' : ''}`}
                onClick={() => setMaxEpocas(e)}
                disabled={training}
              >{e} épocas</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {[16, 32, 64].map(b => (
              <button
                key={b}
                className={`porta-chip${batchSize === b ? ' selected' : ''}`}
                onClick={() => setBatchSize(b)}
                disabled={training}
              >batch {b}</button>
            ))}
          </div>
        </Card>
        <Card title="Amostras de Treino">
          <div className="porta-chips">
            {[5000, 10000, 30000, 0].map(n => (
              <button
                key={n}
                className={`porta-chip${trainLimit === n ? ' selected' : ''}`}
                onClick={() => setTrainLimit(n)}
                disabled={training}
              >{n === 0 ? 'Todas (~88k)' : `${(n / 1000).toFixed(0)}k`}</button>
            ))}
          </div>
        </Card>
      </div>

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <MetricCard
          title="Épocas"
          value={result ? `${result.epocas}` : '--'}
          label={result ? `${(result.tempoMs / 1000).toFixed(1)}s total` : 'aguardando'}
          color="cyan"
          pulse={training}
        />
        <MetricCard
          title="Loss"
          value={result ? result.lossFinal.toFixed(4) : '--'}
          label="cross-entropy"
          color="green"
        />
        <MetricCard
          title="Acurácia Teste"
          value={result ? `${result.acuraciaTest.toFixed(1)}%` : '--'}
          label={result ? `treino: ${result.acuracia.toFixed(1)}%` : 'aguardando'}
          color={result ? 'green' : undefined}
        />
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

      {/* Interactive Test */}
      {trained && (
        <Card title="Teste Interativo — Desenhe uma letra (28×28)" style={{ marginBottom: 24 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 32, alignItems: 'start' }}>
            <div>
              <PixelGrid
                rows={28}
                cols={28}
                cellSize={12}
                gap={1}
                values={testPixels}
                onChange={autoClassify}
                showClear={false}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 10, padding: '6px 12px' }}
                  onClick={() => { setTestPixels(new Array(784).fill(-1)); setClassifyResp(null); }}
                >LIMPAR</button>
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
