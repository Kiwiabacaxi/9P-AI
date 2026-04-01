import { useState } from 'react';
import './components.css';
import type { ViewId } from './api/types';
import { ToastProvider } from './components/shared/Toast';
import TopBar from './components/layout/TopBar';
import Sidebar from './components/layout/Sidebar';
import HebbView from './views/HebbView';
import PerceptronView from './views/PerceptronView';
import MadalineView from './views/MadalineView';
import MlpDesafioView from './views/MlpDesafioView';
import MlpLetrasView from './views/MlpLetrasView';
import MlpFuncView from './views/MlpFuncView';
import MlpOrtView from './views/MlpOrtView';
import ImgregView from './views/ImgregView';
import ImgregBenchView from './views/ImgregBenchView';
import CnnView from './views/CnnView';
import TimeSeriesView from './views/TimeSeriesView';
import AboutView from './views/AboutView';

const viewComponents: Record<ViewId, React.ComponentType> = {
  hebb: HebbView,
  perceptron: PerceptronView,
  madaline: MadalineView,
  mlp: MlpDesafioView,
  letras: MlpLetrasView,
  mlpfunc: MlpFuncView,
  mlport: MlpOrtView,
  imgreg: ImgregView,
  'imgreg-goroutines': () => <ImgregView variant="goroutines" />,
  'imgreg-matrix': () => <ImgregView variant="matrix" />,
  'imgreg-minibatch': () => <ImgregView variant="minibatch" />,
  'imgreg-bench': ImgregBenchView,
  cnn: CnnView,
  timeseries: TimeSeriesView,
  about: AboutView,
};

export default function App() {
  const [activeView, setActiveView] = useState<ViewId>('hebb');
  const ActiveComponent = viewComponents[activeView];

  return (
    <ToastProvider>
      <div className="shell">
        <TopBar />
        <Sidebar active={activeView} onNavigate={setActiveView} />
        <main className="main">
          <ActiveComponent />
        </main>
      </div>
    </ToastProvider>
  );
}
