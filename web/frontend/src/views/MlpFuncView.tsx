import { useState, useRef, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import NetworkViz from '../components/viz/NetworkViz';
import FuncChart from '../components/viz/FuncChart';
import LogChart from '../components/viz/LogChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type { FuncResult, FuncStep, FuncPoint } from '../api/types';

const FUNCOES = [
  { value: 'sin(x)*sin(2x)', label: 'sin(x)*sin(2x)' },
  { value: 'sin(x)', label: 'sin(x)' },
  { value: 'x^2', label: 'x\u00B2' },
  { value: 'x^3', label: 'x\u00B3' },
];

const LAYERS = [
  { value: '1', label: '1 camada' },
  { value: '2', label: '2 camadas' },
  { value: '3', label: '3 camadas' },
  { value: '4', label: '4 camadas' },
];

const NEURONS = [
  { value: '50', label: '50 neuronios' },
  { value: '100', label: '100 neuronios' },
  { value: '200', label: '200 neuronios' },
  { value: '300', label: '300 neuronios' },
];

const ATIVACOES = [
  { value: 'tanh', label: 'tanh' },
  { value: 'sigmoid', label: 'sigmoid' },
  { value: 'relu', label: 'relu' },
];

const LR_OPTIONS = [
  { value: '0.001', label: '\u03B1 0.001' },
  { value: '0.005', label: '\u03B1 0.005' },
  { value: '0.01', label: '\u03B1 0.01' },
  { value: '0.02', label: '\u03B1 0.02' },
];

const EPOCAS_OPTIONS = [
  { value: '10000', label: '10k' },
  { value: '50000', label: '50k' },
  { value: '100000', label: '100k' },
];

export default function MlpFuncView() {
  const { show } = useToast();

  // Config state
  const [funcao, setFuncao] = useState('sin(x)*sin(2x)');
  const [numLayers, setNumLayers] = useState('1');
  const [neurons, setNeurons] = useState('200');
  const [ativacao, setAtivacao] = useState('tanh');
  const [alfa, setAlfa] = useState('0.005');
  const [maxCiclo, setMaxCiclo] = useState('100000');

  // Training state
  const [training, setTraining] = useState(false);
  const [ciclos, setCiclos] = useState<string>('\u2014');
  const [erro, setErro] = useState<string>('\u2014');
  const [status, setStatus] = useState<string>('aguardando');
  const [statusColor, setStatusColor] = useState<string>('var(--on-surface)');

  // Chart data
  const [pontos, setPontos] = useState<FuncPoint[]>([]);
  const [erroHistorico, setErroHistorico] = useState<number[]>([]);

  // UI
  const [showArch, setShowArch] = useState(true);

  // Network viz
  const [activeLayer, setActiveLayer] = useState(-1);

  // SSE cleanup ref
  const closeSSE = useRef<(() => void) | null>(null);

  async function handleTrain() {
    setTraining(true);
    setStatus('treinando...');
    setStatusColor('var(--cyan)');
    setErroHistorico([]);

    const hiddenLayers = Array(parseInt(numLayers)).fill(parseInt(neurons));
    const cfg = {
      funcao,
      hiddenLayers,
      alfa: parseFloat(alfa),
      maxCiclo: parseInt(maxCiclo),
      ativacao,
    };

    try {
      await apiPost('/mlpfunc/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      setStatus('erro');
      setStatusColor('var(--pink)');
      return;
    }

    closeSSE.current = apiSSE('/mlpfunc/train', {
      onMessage(data) {
        const step = data as FuncStep;
        setCiclos(step.ciclo.toLocaleString());
        setErro(step.erroTotal.toFixed(6));
        setActiveLayer(step.activeLayer);
        if (step.pontos) {
          setPontos(step.pontos);
        }
      },
      onDone(data) {
        const result = data as FuncResult;
        setCiclos(result.ciclos.toLocaleString());
        setErro(result.erroFinal.toFixed(6));
        setPontos(result.pontos);
        setErroHistorico(result.erroHistorico);
        setStatus(result.convergiu ? 'convergiu' : 'nao convergiu');
        setStatusColor(result.convergiu ? 'var(--primary-glow)' : 'var(--pink)');
        setTraining(false);
        setActiveLayer(-1);
        closeSSE.current = null;
        show('MLP Funcoes treinado');
      },
      onError() {
        setTraining(false);
        setActiveLayer(-1);
        setStatus('erro de conexao');
        setStatusColor('var(--pink)');
        closeSSE.current = null;
      },
    });
  }

  async function handleReset() {
    // Close any active SSE connection
    if (closeSSE.current) {
      closeSSE.current();
      closeSSE.current = null;
    }

    try {
      await apiPost('/mlpfunc/reset');
    } catch {
      // ignore reset errors
    }

    setCiclos('\u2014');
    setErro('\u2014');
    setStatus('aguardando');
    setStatusColor('var(--on-surface)');
    setPontos([]);
    setErroHistorico([]);
    setTraining(false);
    setActiveLayer(-1);
    show('MLP Funcoes resetado');
  }

  const layerSizes = useMemo(() => {
    const hidden = Array(parseInt(numLayers)).fill(parseInt(neurons));
    return [1, ...hidden, 1];
  }, [numLayers, neurons]);

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">MLP <span>Funções</span></div>
          <div className="page-sub">Aproximação de funções com backpropagation &mdash; Aula 06</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" onClick={handleReset} style={{ fontSize: 11, padding: '6px 12px' }}>
            RESETAR
          </button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            TREINAR
          </button>
        </div>
      </div>

      {/* Config Panel */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Funcao"
            options={FUNCOES}
            value={funcao}
            onChange={setFuncao}
            style={{ width: '100%' }}
          />
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Camadas Ocultas &times; Neuronios</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select
              options={LAYERS}
              value={numLayers}
              onChange={setNumLayers}
              style={{ flex: 1 }}
            />
            <Select
              options={NEURONS}
              value={neurons}
              onChange={setNeurons}
              style={{ flex: 1 }}
            />
          </div>
          <div style={{ marginTop: 10 }}>
            <Select
              label="Ativacao"
              options={ATIVACOES}
              value={ativacao}
              onChange={setAtivacao}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Learning Rate &middot; Epocas</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select
              options={LR_OPTIONS}
              value={alfa}
              onChange={setAlfa}
              style={{ flex: 1 }}
            />
            <Select
              options={EPOCAS_OPTIONS}
              value={maxCiclo}
              onChange={setMaxCiclo}
              style={{ flex: 1 }}
            />
          </div>
        </Card>
      </div>

      {/* Network Visualization (collapsible) */}
      <Card title="Arquitetura da Rede" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 4 }}>
          <button
            className="btn btn-ghost"
            style={{ fontSize: 10, padding: '4px 10px' }}
            onClick={() => setShowArch(prev => !prev)}
          >
            {showArch ? 'OCULTAR' : 'MOSTRAR'}
          </button>
        </div>
        {showArch && (
          <NetworkViz
            layerSizes={layerSizes}
            activeLayer={activeLayer}
            hudText={ativacao}
            animate={!training}
          />
        )}
      </Card>

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
          title="Erro"
          value={erro}
          label="0.5 * sum((t-y)\u00B2)"
          color="cyan"
        />
        <MetricCard
          title="Status"
          value={status}
          label="estado da rede"
          valueStyle={{ fontSize: 18, color: statusColor }}
        />
      </div>

      {/* Charts (stacked vertically, full width) */}
      <Card title="Função Original vs Aproximação da Rede" style={{ marginBottom: 16 }}>
        <FuncChart pontos={pontos} height={300} />
      </Card>

      <Card title="Curva de Erro — escala log" style={{ marginBottom: 16 }}>
        <LogChart data={erroHistorico} color="#00ccff" />
      </Card>

      {/* Details */}
      <Card title="Detalhes">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          Baseado no codigo Python da Aula 06 (MLPfuncaoBase.py). A rede usa <b>{ativacao}</b> como ativacao
          para aproximar uma funcao matematica usando 50 pontos igualmente espacados no intervalo [-1, 1].
          <br /><br />
          <b>Arquitetura:</b> 1 entrada (x) &middot; {layerSizes.slice(1, -1).map(n => `${n} ocultos`).join(' · ')} ({ativacao}) &middot; 1 saida ({ativacao}) &middot; erro alvo = 0.02
        </div>
      </Card>
    </div>
  );
}
