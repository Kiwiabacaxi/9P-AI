import { useEffect, useState } from 'react';
import type { AppStatus, ViewId } from '../../api/types';
import { apiGet } from '../../api/client';

interface Props {
  active: ViewId;
}

// Para cada view: rótulo curto + como ler treinado/treinando no /status.
// Views sem leitor de status (horário, tsp-multi, rastrigin, about…) aparecem
// só com o rótulo (telemetria ainda não exposta pelo backend pra elas).
interface ViewInfo {
  label: string;
  trained?: (s: AppStatus) => boolean;
  training?: (s: AppStatus) => boolean;
}

const VIEW_INFO: Record<ViewId, ViewInfo> = {
  hebb: { label: 'Hebb', trained: s => s.hebbCount > 0 },
  perceptron: { label: 'Perceptron', trained: s => s.percPortaCount > 0 },
  madaline: { label: 'MADALINE', trained: s => s.madTrained, training: s => s.madTraining },
  mlp: { label: 'MLP Desafio', trained: s => s.mlpTrained },
  letras: { label: 'MLP Letras', trained: s => s.letrasTrained, training: s => s.ltrTraining },
  mlpfunc: { label: 'MLP Funções', trained: s => s.mlpFuncTrained, training: s => s.mlpFuncTraining },
  mlport: { label: 'MLP Ortogonal', trained: s => s.ortTrained, training: s => s.ortTraining },
  imgreg: { label: 'IMG · Standard', trained: s => s.imgregTrained, training: s => s.imgregTraining },
  'imgreg-goroutines': { label: 'IMG · Goroutines', trained: s => s.igorTrained, training: s => s.igorTraining },
  'imgreg-matrix': { label: 'IMG · Matrix', trained: s => s.imatTrained, training: s => s.imatTraining },
  'imgreg-minibatch': { label: 'IMG · Mini-batch', trained: s => s.imbTrained, training: s => s.imbTraining },
  'imgreg-bench': { label: 'IMG · Benchmark' },
  cnn: { label: 'CNN EMNIST', trained: s => s.cnnTrained, training: s => s.cnnTraining },
  timeseries: { label: 'MLP Ações', trained: s => s.tsTrained, training: s => s.tsTraining },
  genetico: { label: 'GA · Aula 10', trained: s => s.gaTrained, training: s => s.gaTraining },
  genetico2: { label: 'GA · Aula 11', trained: s => s.ga2Trained, training: s => s.ga2Training },
  horario: { label: 'GA · Horário' },
  tsp: { label: 'GA · TSP', trained: s => s.tspTrained, training: s => s.tspTraining },
  'tsp-compare': { label: 'GA · TSP Comparativo' },
  'tsp-multi': { label: 'GA · TSP Multi-ilhas' },
  rastrigin: { label: 'GA · Rastrigin 3D' },
  about: { label: 'Arquitetura' },
};

export default function TopBar({ active }: Props) {
  const [status, setStatus] = useState<AppStatus | null>(null);

  useEffect(() => {
    const poll = () => apiGet<AppStatus>('/status').then(setStatus).catch(() => {});
    poll();
    const iv = setInterval(poll, 3000);
    return () => clearInterval(iv);
  }, []);

  const info = VIEW_INFO[active];
  const hasStatus = !!info.trained;
  const training = !!(info.training && status && info.training(status));
  const trained = !!(info.trained && status && info.trained(status));

  const dotCls = training ? 'status-dot warn' : trained ? 'status-dot active' : 'status-dot';
  const stateText = training ? 'rodando' : trained ? 'treinado' : hasStatus ? 'não treinado' : 'pronto';
  const stateColor = training ? 'var(--amber, #ffb020)' : trained ? 'var(--green, #4ade80)' : 'var(--muted)';

  return (
    <header className="topbar">
      <div className="topbar-logo">AI_MISSION_CONTROL<span style={{ color: 'var(--on-surface)' }}>_v4.0</span></div>
      <div className="topbar-sep" />
      <div className="topbar-status">
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={dotCls} />
          <span>{info.label}</span>
          <span style={{ color: 'var(--muted)' }}>·</span>
          <span style={{ color: stateColor }}>{stateText}</span>
        </span>
        <div className="pulse" />
      </div>
    </header>
  );
}
