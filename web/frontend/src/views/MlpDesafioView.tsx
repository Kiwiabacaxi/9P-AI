import { useState, useRef } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import LogChart from '../components/viz/LogChart';
import NetworkViz from '../components/viz/NetworkViz';
import { useToast } from '../components/shared/Toast';
import { apiPost } from '../api/client';

// ---------------------------------------------------------------------------
// Types matching the Go MLPResult / MLPStep / MLP structs
// ---------------------------------------------------------------------------

interface MlpForward {
  Zin: number[];
  Z: number[];
  Yin: number[];
  Y: number[];
}

interface MlpBackward {
  DeltaK: number[];
  DeltaInJ: number[];
  DeltaJ: number[];
}

interface MlpStep {
  ciclo: number;
  padrao: number;
  x: number[];
  target: number[];
  fwd: MlpForward;
  bwd: MlpBackward;
  erroTotal: number;
}

interface MlpRede {
  V: number[][];
  V0: number[];
  W: number[][];
  W0: number[];
}

interface MlpTrainResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  steps: MlpStep[];
  rede: MlpRede;
}

// Fixed training patterns and targets (same as Go code)
const PADROES: number[][] = [
  [1, 0.5, -1],
  [1, 0.5, 1],
  [1, -0.5, -1],
];
const TARGETS: number[][] = [
  [1, -1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MlpDesafioView() {
  const { show } = useToast();

  // Training state
  const [training, setTraining] = useState(false);
  const [ciclos, setCiclos] = useState('\u2014');
  const [erro, setErro] = useState('\u2014');
  const [status, setStatus] = useState('aguardando');
  const [statusColor, setStatusColor] = useState('var(--on-surface)');

  // Result data
  const [erroHistorico, setErroHistorico] = useState<number[]>([]);
  const [rede, setRede] = useState<MlpRede | null>(null);
  const [steps, setSteps] = useState<MlpStep[]>([]);
  const [stepIdx, setStepIdx] = useState(0);

  // Prevent double-click
  const lockRef = useRef(false);

  // ---------- Train ----------

  async function handleTrain() {
    if (lockRef.current) return;
    lockRef.current = true;
    setTraining(true);
    setStatus('treinando...');
    setStatusColor('var(--cyan)');

    try {
      const data = await apiPost<MlpTrainResult>('/mlp/train');
      setCiclos(data.ciclos.toLocaleString());
      setErro(data.erroFinal.toFixed(6));
      setErroHistorico(data.erroHistorico);
      setRede(data.rede);
      setSteps(data.steps || []);
      setStepIdx(0);
      setStatus(data.convergiu ? '\u2713 convergiu' : '\u26A0 nao convergiu');
      setStatusColor(data.convergiu ? 'var(--primary-glow)' : 'var(--pink)');
      show('MLP Desafio treinado');
    } catch (e) {
      setStatus('erro');
      setStatusColor('var(--pink)');
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setTraining(false);
      lockRef.current = false;
    }
  }

  // ---------- Forward pass for table ----------

  function computeOutputs(net: MlpRede, x: number[]): number[] {
    // Hidden layer
    const z: number[] = [];
    for (let j = 0; j < 2; j++) {
      let s = net.V0[j];
      for (let i = 0; i < 3; i++) s += x[i] * net.V[i][j];
      z.push(Math.tanh(s));
    }
    // Output layer
    const y: number[] = [];
    for (let k = 0; k < 3; k++) {
      let s = net.W0[k];
      for (let j = 0; j < 2; j++) s += z[j] * net.W[j][k];
      y.push(Math.tanh(s));
    }
    return y;
  }

  // ---------- Step explorer ----------

  const currentStep = steps.length > 0 ? steps[stepIdx] : null;

  function fmtVal(v: number) {
    const cls = v >= 0 ? 'pos' : 'neg';
    return <span className={`val ${cls}`}>{v.toFixed(6)}</span>;
  }

  // ---------- Render ----------

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">MLP <span>Desafio</span></div>
          <div className="page-sub">3 entradas &middot; 2 ocultos &middot; 3 saidas &middot; tanh &middot; backpropagation</div>
        </div>
        <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
          {training && <span className="spin" />}
          TREINAR REDE
        </button>
      </div>

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Ciclos"
          value={ciclos}
          label="convergencia"
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Erro Final"
          value={erro}
          label="quadratico total"
          color="cyan"
        />
        <MetricCard
          title="Status"
          value={status}
          label="estado da rede"
          valueStyle={{ fontSize: 18, color: statusColor }}
        />
      </div>

      {/* Error chart + Step explorer */}
      <div className="grid-2" style={{ marginBottom: 16 }}>
        <Card title="Curva de Erro \u2014 escala log">
          <LogChart data={erroHistorico} />
        </Card>

        <Card title="Explorador de Backprop">
          {steps.length === 0 ? (
            <div className="empty" style={{ padding: '24px 0' }}>
              <div className="empty-icon">{'\u25C8'}</div>
              <div>Treine a rede para explorar os passos</div>
            </div>
          ) : currentStep && (
            <>
              {/* Step navigation */}
              <div className="step-nav">
                <button
                  className="btn btn-ghost"
                  style={{ padding: '6px 12px' }}
                  onClick={() => setStepIdx(Math.max(0, stepIdx - 1))}
                  disabled={stepIdx === 0}
                >{'\u25C0'}</button>
                <span className="step-counter">
                  ciclo {currentStep.ciclo} &middot; padrao {currentStep.padrao} &middot; passo {stepIdx + 1}/{steps.length}
                </span>
                <button
                  className="btn btn-ghost"
                  style={{ padding: '6px 12px' }}
                  onClick={() => setStepIdx(Math.min(steps.length - 1, stepIdx + 1))}
                  disabled={stepIdx === steps.length - 1}
                >{'\u25B6'}</button>
              </div>

              {/* Step details */}
              <div className="step-info">
                <div className="step-block">
                  <div className="step-block-title">Forward Pass</div>
                  <div className="step-row">
                    <span>x</span>
                    <span>{currentStep.x.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>zin</span>
                    <span>{currentStep.fwd.Zin.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>z</span>
                    <span>{currentStep.fwd.Z.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>y</span>
                    <span>{currentStep.fwd.Y.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>target</span>
                    <span>{currentStep.target.map(v => fmtVal(v))}</span>
                  </div>
                </div>
                <div className="step-block">
                  <div className="step-block-title">Backward Pass</div>
                  <div className="step-row">
                    <span>{'\u03B4'}k</span>
                    <span>{currentStep.bwd.DeltaK.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>{'\u03B4'}inj</span>
                    <span>{currentStep.bwd.DeltaInJ.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>{'\u03B4'}j</span>
                    <span>{currentStep.bwd.DeltaJ.map(v => fmtVal(v))}</span>
                  </div>
                  <div className="step-row">
                    <span>erro acum.</span>
                    {fmtVal(currentStep.erroTotal)}
                  </div>
                </div>
              </div>
            </>
          )}
        </Card>
      </div>

      {/* Training patterns table */}
      <Card title="Padroes de Treinamento" style={{ marginBottom: 16 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Padrao</th>
              <th>x{'\u2081'}</th><th>x{'\u2082'}</th><th>x{'\u2083'}</th>
              <th>t{'\u2081'}</th><th>t{'\u2082'}</th><th>t{'\u2083'}</th>
              <th>y{'\u2081'}</th><th>y{'\u2082'}</th><th>y{'\u2083'}</th>
              <th>Classe</th>
            </tr>
          </thead>
          <tbody>
            {rede ? (
              PADROES.map((x, i) => {
                const y = computeOutputs(rede, x);
                const pred = y.indexOf(Math.max(...y));
                return (
                  <tr key={i}>
                    <td className="td-cyan">P{i + 1}</td>
                    <td>{x[0]}</td><td>{x[1]}</td><td>{x[2]}</td>
                    <td className="td-white">{TARGETS[i][0]}</td>
                    <td className="td-white">{TARGETS[i][1]}</td>
                    <td className="td-white">{TARGETS[i][2]}</td>
                    <td className={y[0] >= 0 ? 'td-green' : 'td-pink'}>{y[0].toFixed(4)}</td>
                    <td className={y[1] >= 0 ? 'td-green' : 'td-pink'}>{y[1].toFixed(4)}</td>
                    <td className={y[2] >= 0 ? 'td-green' : 'td-pink'}>{y[2].toFixed(4)}</td>
                    <td className="td-white" style={{ fontWeight: 700 }}>C{pred + 1}</td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={11} style={{ textAlign: 'center', color: 'var(--surface-top)', padding: 20 }}>
                  &mdash; treine para ver resultados &mdash;
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>

      {/* Network Visualization */}
      <Card title="Arquitetura da Rede">
        <NetworkViz layerSizes={[3, 2, 3]} hudText="tanh" />
      </Card>
    </div>
  );
}
