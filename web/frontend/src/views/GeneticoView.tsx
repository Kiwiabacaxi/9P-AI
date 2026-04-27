import { useState, useRef, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import GaChart, { GaEvoChart } from '../components/viz/GaChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type { GAConfig, GAStep, GAResult, GAIndividuo } from '../api/types';

const POP_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '40', label: '40' },
  { value: '80', label: '80' },
  { value: '120', label: '120' },
];

const BITS_OPTIONS = [
  { value: '8', label: '8' },
  { value: '10', label: '10' },
  { value: '12', label: '12' },
  { value: '14', label: '14' },
  { value: '16', label: '16' },
];

const GERACOES_OPTIONS = [
  { value: '20', label: '20' },
  { value: '50', label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '500', label: '500' },
  { value: '1000', label: '1000' },
];

const PC_OPTIONS = [
  { value: '0.4', label: 'Pc 0.40' },
  { value: '0.6', label: 'Pc 0.60' },
  { value: '0.7', label: 'Pc 0.70' },
  { value: '0.8', label: 'Pc 0.80' },
  { value: '0.9', label: 'Pc 0.90' },
];

const PM_OPTIONS = [
  { value: '0.005', label: 'Pm 0.005' },
  { value: '0.01', label: 'Pm 0.01' },
  { value: '0.02', label: 'Pm 0.02' },
  { value: '0.05', label: 'Pm 0.05' },
  { value: '0.1', label: 'Pm 0.10' },
  { value: '0.2', label: 'Pm 0.20' },
];

const DOMINIO_MIN = 0;
const DOMINIO_MAX = 512;

// ótimo global teórico de f(x) = -|x sin(√x)| em [0, 512]
// (função tipo Schwefel — valores conhecidos da literatura)
const X_OPTIMO = 420.9687;
const F_OPTIMO = -418.9829;

export default function GeneticoView() {
  const { show } = useToast();

  // Config
  const [bits, setBits] = useState('10');
  const [popSize, setPopSize] = useState('20');
  const [maxGeracoes, setMaxGeracoes] = useState('100');
  const [probCruz, setProbCruz] = useState('0.8');
  const [probMut, setProbMut] = useState('0.05');

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorX, setMelhorX] = useState<string>('—');
  const [melhorFx, setMelhorFx] = useState<string>('—');
  const [mediaFx, setMediaFx] = useState<string>('—');
  const [status, setStatus] = useState<string>('aguardando');
  const [statusColor, setStatusColor] = useState<string>('var(--on-surface)');

  // Animation data
  const [populacao, setPopulacao] = useState<GAIndividuo[]>([]);
  const [melhorIndiv, setMelhorIndiv] = useState<GAIndividuo | null>(null);
  const [globalBest, setGlobalBest] = useState<GAIndividuo | null>(null);

  // History
  const [histMelhor, setHistMelhor] = useState<number[]>([]);
  const [histMedia, setHistMedia] = useState<number[]>([]);

  // Replay: histórico completo de populações (ref pra não re-renderizar a cada push)
  const histPopRef = useRef<GAIndividuo[][]>([]);
  const histMelhorRef = useRef<GAIndividuo[]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0); // 1-indexed; 0 = nada
  const [userScrub, setUserScrub] = useState(false); // true quando usuário arrastou

  const closeSSE = useRef<(() => void) | null>(null);

  async function handleTrain() {
    setTraining(true);
    setStatus('evoluindo...');
    setStatusColor('var(--cyan)');
    setHistMelhor([]);
    setHistMedia([]);
    setPopulacao([]);
    setMelhorIndiv(null);
    setGlobalBest(null);
    histPopRef.current = [];
    histMelhorRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);

    const cfg: GAConfig = {
      bits: parseInt(bits),
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGeracoes),
      probCruzamento: parseFloat(probCruz),
      probMutacao: parseFloat(probMut),
    };

    try {
      await apiPost('/genetico/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      setStatus('erro');
      setStatusColor('var(--pink)');
      return;
    }

    closeSSE.current = apiSSE('/genetico/train', {
      onMessage(data) {
        const step = data as GAStep;
        // arquiva no histórico (ref — evita N re-renders por push)
        histPopRef.current.push(step.populacao);
        histMelhorRef.current.push(step.melhorIndiv);

        setGeracao(step.geracao.toLocaleString());
        setMelhorX(step.melhorX.toFixed(2));
        setMelhorFx(step.melhorFx.toFixed(4));
        setMediaFx(step.mediaFx.toFixed(4));
        setMaxGen(step.geracao);
        // só auto-avança o slider se o usuário não estiver arrastando
        if (!userScrub) {
          setDisplayGen(step.geracao);
          setPopulacao(step.populacao);
          setMelhorIndiv(step.melhorIndiv);
        }
        // melhor global é monotônico — sempre atualiza
        setGlobalBest(prev => {
          if (!prev || step.melhorIndiv.fitness > prev.fitness) return step.melhorIndiv;
          return prev;
        });
        setHistMelhor(prev => [...prev, step.melhorFx]);
        setHistMedia(prev => [...prev, step.mediaFx]);
      },
      onDone(data) {
        const r = data as GAResult;
        setGeracao(r.geracoes.toLocaleString());
        setMelhorX(r.melhorX.toFixed(4));
        setMelhorFx(r.melhorFx.toFixed(4));
        setMelhorIndiv(r.melhorIndiv);
        setGlobalBest(r.melhorIndiv);
        setHistMelhor(r.histMelhorFx);
        setHistMedia(r.histMediaFx);
        // ao terminar, posiciona slider na última geração se o usuário não scrubou
        if (!userScrub) {
          setDisplayGen(r.geracoes);
          const last = histPopRef.current[r.geracoes - 1];
          if (last) setPopulacao(last);
        }
        setStatus('concluido');
        setStatusColor('var(--primary-glow)');
        setTraining(false);
        closeSSE.current = null;
        show('GA finalizado — arraste o slider pra ver a evolução');
      },
      onError() {
        setTraining(false);
        setStatus('erro de conexao');
        setStatusColor('var(--pink)');
        closeSSE.current = null;
      },
    });
  }

  async function handleReset() {
    if (closeSSE.current) {
      closeSSE.current();
      closeSSE.current = null;
    }
    try {
      await apiPost('/genetico/reset');
    } catch {
      // ignore
    }
    setGeracao('—');
    setMelhorX('—');
    setMelhorFx('—');
    setMediaFx('—');
    setStatus('aguardando');
    setStatusColor('var(--on-surface)');
    setPopulacao([]);
    setMelhorIndiv(null);
    setGlobalBest(null);
    setHistMelhor([]);
    setHistMedia([]);
    histPopRef.current = [];
    histMelhorRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);
    setTraining(false);
    show('GA resetado');
  }

  // Quando o usuário move o slider: mostra a população daquela geração.
  function handleSlider(gen: number) {
    setUserScrub(true);
    setDisplayGen(gen);
    const pop = histPopRef.current[gen - 1];
    const best = histMelhorRef.current[gen - 1];
    if (pop) setPopulacao(pop);
    if (best) setMelhorIndiv(best);
  }

  // Estatísticas da geração atualmente exibida (pode ser passada).
  const displayStats = useMemo(() => {
    if (displayGen <= 0 || displayGen > histMelhor.length) return null;
    return {
      melhorFx: histMelhor[displayGen - 1],
      mediaFx: histMedia[displayGen - 1],
    };
  }, [displayGen, histMelhor, histMedia]);

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Algoritmo <span>Genético</span></div>
          <div className="page-sub">
            Otimização de f(x) = -|x &middot; sin(&radic;|x|)| em [0, 512] &mdash; Aula 10
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            className="btn"
            onClick={handleReset}
            style={{ fontSize: 11, padding: '6px 12px' }}
          >
            RESETAR
          </button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            EVOLUIR
          </button>
        </div>
      </div>

      {/* Config Panel */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">População &middot; Bits</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={POP_OPTIONS} value={popSize} onChange={setPopSize} style={{ flex: 1 }} />
            <Select options={BITS_OPTIONS} value={bits} onChange={setBits} style={{ flex: 1 }} />
          </div>
          <div style={{ marginTop: 10 }}>
            <Select
              label="Gerações"
              options={GERACOES_OPTIONS}
              value={maxGeracoes}
              onChange={setMaxGeracoes}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Probabilidade de Cruzamento"
            options={PC_OPTIONS}
            value={probCruz}
            onChange={setProbCruz}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10 }}>
            <Select
              label="Probabilidade de Mutação (por bit)"
              options={PM_OPTIONS}
              value={probMut}
              onChange={setProbMut}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Melhor indivíduo</div>
          {melhorIndiv ? (
            <div style={{ fontSize: 12, lineHeight: 1.7, fontFamily: 'JetBrains Mono' }}>
              <div>bits: <code style={{ color: 'var(--cyan)' }}>{melhorIndiv.bits.join('')}</code></div>
              <div>dec: <code>{melhorIndiv.dec}</code></div>
              <div>x: <code>{melhorIndiv.x.toFixed(4)}</code></div>
              <div>f(x): <code style={{ color: 'var(--pink)' }}>{melhorIndiv.fx.toFixed(4)}</code></div>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>aguardando treino&hellip;</div>
          )}
        </Card>
      </div>

      {/* Metrics */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Geração"
          value={geracao}
          label="iteração atual"
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Melhor x*"
          value={melhorX}
          label="argmin f(x)"
          color="cyan"
        />
        <MetricCard
          title="Status"
          value={status}
          label={`f(x*) = ${melhorFx} | media = ${mediaFx}`}
          valueStyle={{ fontSize: 18, color: statusColor }}
        />
      </div>

      {/* Main chart: function curve + population scatter + replay slider */}
      <Card title="População sobre f(x)" style={{ marginBottom: 16 }}>
        <GaChart
          populacao={populacao}
          melhor={melhorIndiv}
          globalBest={globalBest}
          dominioMin={DOMINIO_MIN}
          dominioMax={DOMINIO_MAX}
          optimoX={X_OPTIMO}
          optimoFx={F_OPTIMO}
          height={340}
        />
        {maxGen > 0 && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '8px 12px',
            marginTop: 8,
            background: 'var(--surface-2)',
            borderRadius: 6,
          }}>
            <button
              className="btn btn-ghost"
              style={{ fontSize: 10, padding: '4px 10px' }}
              onClick={() => {
                setUserScrub(false);
                setDisplayGen(maxGen);
                const last = histPopRef.current[maxGen - 1];
                const lastBest = histMelhorRef.current[maxGen - 1];
                if (last) setPopulacao(last);
                if (lastBest) setMelhorIndiv(lastBest);
              }}
              disabled={training && !userScrub}
              title="voltar pra geracao mais recente"
            >
              {'⏭'} latest
            </button>
            <div style={{
              fontSize: 11,
              fontFamily: 'JetBrains Mono',
              color: 'var(--muted)',
              minWidth: 110,
            }}>
              geração {displayGen.toLocaleString()} / {maxGen.toLocaleString()}
            </div>
            <input
              type="range"
              min={1}
              max={maxGen}
              value={displayGen || 1}
              onChange={(e) => handleSlider(parseInt(e.target.value))}
              style={{ flex: 1, accentColor: 'var(--cyan)' }}
            />
            {displayStats && (
              <div style={{
                fontSize: 11,
                fontFamily: 'JetBrains Mono',
                color: 'var(--muted)',
                whiteSpace: 'nowrap',
              }}>
                f(x)<sub>melhor</sub> = <span style={{ color: 'var(--pink)' }}>{displayStats.melhorFx.toFixed(2)}</span>
                {' '}&nbsp;|&nbsp;{' '}
                <span style={{ opacity: 0.7 }}>media = {displayStats.mediaFx.toFixed(2)}</span>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Evolution chart */}
      <Card title="Evolução do f(x) — melhor e médio" style={{ marginBottom: 16 }}>
        <GaEvoChart histMelhor={histMelhor} histMedia={histMedia} />
      </Card>

      {/* Educational details */}
      <Card title="Como funciona">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          O algoritmo segue, passo a passo, o pseudocódigo do slide da Aula 10:
          <br /><br />
          <code style={{ display: 'block', whiteSpace: 'pre', color: 'var(--on-surface)', background: 'var(--surface-2)', padding: 12, borderRadius: 6, fontSize: 12 }}>
{`popBin       = gerarElementosBinarios(popDec)
imagemFuncao = gerarImagem(popDec)
probabRolet  = gerarProbabilidades(imagemFuncao)
sMelhores    = separarVinteMelhores(probabRolet, popBin)   // ${popSize} sorteios via roleta
casaisFormados = sortearCasais()                           // ${Math.floor(parseInt(popSize)/2)} casais
pontoCorte   = gerarPontoDeCorte()                         // ∈ [1, ${parseInt(bits) - 1}]
s_filhos     = cruzamento(pontoCorte, casaisFormados, sMelhores)  // Pc = ${probCruz}
s_filhos     = efetuarMutacao(s_filhos)                    // Pm = ${probMut} por bit`}
          </code>
          <br />
          <b>Codificação:</b> cada cromossomo tem <b>{bits} bits</b> ⇒ {Math.pow(2, parseInt(bits)).toLocaleString()} valores possíveis,
          mapeados linearmente em <b>x ∈ [0, 512]</b>.
          <br />
          <b>Fitness:</b> como <i>f(x) ≤ 0</i> neste domínio e queremos <i>minimizar</i>, usamos
          <code> fitness = −f(x) ≥ 0</code> — assim o melhor (mais negativo) recebe maior probabilidade na roleta.
          <br />
          <b>Sem elitismo:</b> os filhos substituem a população inteira (fiel ao slide). Mesmo assim, a UI rastreia o
          melhor global ao longo das gerações para você não perdê-lo de vista.
          <br />
          <b>Mutação só nos filhos</b> (Obs 03 do slide). <b>Cruzamento de 1 ponto</b>, com o mesmo
          <code> pontoCorte</code> sorteado uma vez por geração (interpretação literal do pseudocódigo).
          <br /><br />
          <b>Função:</b> f(x) = −|x · sin(√|x|)| é uma variante da função de Schwefel — multimodal, com
          mínimo global perto de <b>x ≈ 420.97</b> e f(x*) ≈ <b>−418.98</b>. Bom benchmark didático porque
          tem muitos mínimos locais que prendem buscas locais (gradiente, hill-climbing).
        </div>
      </Card>
    </div>
  );
}
