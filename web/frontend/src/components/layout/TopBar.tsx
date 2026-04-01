import { useEffect, useState } from 'react';
import type { AppStatus } from '../../api/types';
import { apiGet } from '../../api/client';

interface DotInfo {
  id: string;
  label: string;
  trainedKey: keyof AppStatus;
  trainingKey?: keyof AppStatus;
}

const dots: DotInfo[] = [
  { id: 'hebb', label: 'HEBB', trainedKey: 'hebbCount' },
  { id: 'perc', label: 'PERCEPTRON', trainedKey: 'percPortaCount' },
  { id: 'mad', label: 'MADALINE', trainedKey: 'madTrained', trainingKey: 'madTraining' },
  { id: 'mlp', label: 'MLP', trainedKey: 'mlpTrained' },
  { id: 'ltr', label: 'LETRAS', trainedKey: 'letrasTrained', trainingKey: 'ltrTraining' },
  { id: 'imgreg', label: 'IMG_REG', trainedKey: 'imgregTrained', trainingKey: 'imgregTraining' },
  { id: 'mlpfunc', label: 'FUNC', trainedKey: 'mlpFuncTrained', trainingKey: 'mlpFuncTraining' },
  { id: 'ort', label: 'ORT', trainedKey: 'ortTrained', trainingKey: 'ortTraining' },
  { id: 'cnn', label: 'CNN', trainedKey: 'cnnTrained', trainingKey: 'cnnTraining' },
  { id: 'ts', label: 'TS', trainedKey: 'tsTrained', trainingKey: 'tsTraining' },
];

export default function TopBar() {
  const [status, setStatus] = useState<AppStatus | null>(null);

  useEffect(() => {
    const poll = () => apiGet<AppStatus>('/status').then(setStatus).catch(() => {});
    poll();
    const iv = setInterval(poll, 3000);
    return () => clearInterval(iv);
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-logo">RNA_MISSION_CONTROL<span style={{ color: 'var(--on-surface)' }}>_v3.0</span></div>
      <div className="topbar-sep" />
      <div className="topbar-status">
        {dots.map(d => {
          const trained = status ? !!(status[d.trainedKey]) : false;
          const training = d.trainingKey && status ? !!(status[d.trainingKey]) : false;
          const cls = training ? 'status-dot warn' : trained ? 'status-dot active' : 'status-dot';
          return <span key={d.id}><span className={cls} />{d.label}</span>;
        })}
        <div className="pulse" />
      </div>
    </header>
  );
}
