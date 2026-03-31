import { useState } from 'react';
import type { ViewId } from '../../api/types';

interface Props {
  active: ViewId;
  onNavigate: (view: ViewId) => void;
}

interface NavEntry {
  view: ViewId;
  label: string;
  icon: string;
}

const algos: NavEntry[] = [
  { view: 'hebb', label: 'Hebb', icon: '\u25C7' },
  { view: 'perceptron', label: 'Perceptron', icon: '\u25C8' },
  { view: 'madaline', label: 'MADALINE', icon: '\u2B21' },
];

const desafios: NavEntry[] = [
  { view: 'mlp', label: 'MLP Desafio', icon: '\u25CE' },
  { view: 'letras', label: 'MLP Letras', icon: '\u25A4' },
  { view: 'mlpfunc', label: 'MLP Funções', icon: '\u223F' },
  { view: 'mlport', label: 'MLP Ortogonal', icon: '\u229E' },
];

const imgregItems: NavEntry[] = [
  { view: 'imgreg', label: 'Standard', icon: '' },
  { view: 'imgreg-goroutines', label: 'Goroutines', icon: '' },
  { view: 'imgreg-matrix', label: 'Matrix', icon: '' },
  { view: 'imgreg-minibatch', label: 'Mini-batch', icon: '' },
  { view: 'imgreg-bench', label: 'Benchmark', icon: '' },
];

export default function Sidebar({ active, onNavigate }: Props) {
  const [imgregOpen, setImgregOpen] = useState(
    imgregItems.some(i => i.view === active)
  );

  const navItem = (entry: NavEntry) => (
    <div
      key={entry.view}
      className={`nav-item${active === entry.view ? ' active' : ''}`}
      onClick={() => onNavigate(entry.view)}
    >
      <span className="nav-icon" style={{ textAlign: 'center', fontSize: '12px' }}>{entry.icon}</span>
      {entry.label}
    </div>
  );

  return (
    <nav className="sidebar">
      <div className="sidebar-label">Algoritmos</div>
      {algos.map(navItem)}
      <div className="sidebar-divider" />
      <div className="sidebar-label">Desafios MLP</div>
      {desafios.map(navItem)}
      <div
        className={`nav-accordion-header${imgregOpen ? ' open' : ''}`}
        onClick={() => setImgregOpen(!imgregOpen)}
      >
        <span className="nav-icon" style={{ textAlign: 'center', fontSize: '12px' }}>{'\u25A6'}</span>
        IMG_REGRESSION
        <span className="nav-accordion-arrow">{'\u25B6'}</span>
      </div>
      <div className={`nav-accordion-children${imgregOpen ? ' open' : ''}`}>
        {imgregItems.map(navItem)}
      </div>
      <div className="sidebar-divider" />
      <div className="sidebar-label">Deep Learning</div>
      <div
        className={`nav-item${active === 'cnn' ? ' active' : ''}`}
        onClick={() => onNavigate('cnn')}
      >
        <span className="nav-icon" style={{ textAlign: 'center', fontSize: '12px' }}>{'\u25A7'}</span>
        CNN EMNIST
      </div>
      <div className="sidebar-divider" />
      <div
        className={`nav-item${active === 'about' ? ' active' : ''}`}
        onClick={() => onNavigate('about')}
      >
        <span className="nav-icon" style={{ textAlign: 'center', fontSize: '12px' }}>{'\u24D8'}</span>
        Arquitetura
      </div>
    </nav>
  );
}
