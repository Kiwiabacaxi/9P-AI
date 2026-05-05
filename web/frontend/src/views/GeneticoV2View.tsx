import { useState, useRef, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import GaChart, { GaEvoChart } from '../components/viz/GaChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type {
  GA2Config, GA2Step, GA2Result, GA2Selecao, GAIndividuo,
} from '../api/types';

const POP_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '40', label: '40' },
  { value: '80', label: '80' },
  { value: '120', label: '120' },
  { value: '200', label: '200' },
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

const SELECAO_OPTIONS = [
  { value: 'torneio', label: 'Torneio' },
  { value: 'roleta', label: 'Roleta' },
];

const TORNEIO_OPTIONS = [
  { value: '2', label: 'k = 2' },
  { value: '3', label: 'k = 3' },
  { value: '4', label: 'k = 4' },
  { value: '5', label: 'k = 5' },
  { value: '7', label: 'k = 7' },
  { value: '10', label: 'k = 10' },
];

const PONTOS_OPTIONS = [
  { value: '1', label: '1 ponto' },
  { value: '2', label: '2 pontos' },
];

const ELITE_OPTIONS = [
  { value: '0', label: 'sem elite' },
  { value: '1', label: 'p = 1' },
  { value: '2', label: 'p = 2' },
  { value: '4', label: 'p = 4' },
  { value: '6', label: 'p = 6' },
];

// Presets de domínio: a função f(x) = -|x sin(√|x|)| tem mínimos cada vez mais
// profundos conforme x cresce (porque o mínimo escala com |x|), então mudar
// o domínio muda também o ótimo global. O frontend calcula o ótimo numericamente
// pra cada domínio (ver useMemo de optimo abaixo).
const DOMINIO_OPTIONS = [
  { value: '256',  label: '[0, 256]' },
  { value: '512',  label: '[0, 512]  (default)' },
  { value: '768',  label: '[0, 768]' },
  { value: '1024', label: '[0, 1024]' },
  { value: '1500', label: '[0, 1500]' },
  { value: '2250', label: '[0, 2250]' }, // termina logo após (15π)² ≈ 2220, onde |x·sin(√x)|=0
];

const DOMINIO_MIN = 0;

export default function GeneticoV2View() {
  const { show } = useToast();

  // Config
  const [bits, setBits] = useState('10');
  const [popSize, setPopSize] = useState('20');
  const [maxGeracoes, setMaxGeracoes] = useState('100');
  const [probCruz, setProbCruz] = useState('0.8');
  const [probMut, setProbMut] = useState('0.05');
  const [selecao, setSelecao] = useState<string>('torneio');
  const [tamTorneio, setTamTorneio] = useState('3');
  const [pontosCorte, setPontosCorte] = useState('2');
  const [elitismo, setElitismo] = useState('2');
  const [dominioMax, setDominioMax] = useState('512');

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorX, setMelhorX] = useState<string>('—');
  const [melhorFx, setMelhorFx] = useState<string>('—');
  const [diversidade, setDiversidade] = useState<string>('—');
  const [statusColor, setStatusColor] = useState<string>('var(--on-surface)');

  // Animation data
  const [populacao, setPopulacao] = useState<GAIndividuo[]>([]);
  const [melhorIndiv, setMelhorIndiv] = useState<GAIndividuo | null>(null);
  const [globalBest, setGlobalBest] = useState<GAIndividuo | null>(null);

  // History
  const [histMelhor, setHistMelhor] = useState<number[]>([]);
  const [histMedia, setHistMedia] = useState<number[]>([]);
  const [histDiv, setHistDiv] = useState<number[]>([]);

  // Replay
  const histPopRef = useRef<GAIndividuo[][]>([]);
  const histMelhorRef = useRef<GAIndividuo[]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0);
  const [userScrub, setUserScrub] = useState(false);

  const closeSSE = useRef<(() => void) | null>(null);

  async function handleTrain() {
    setTraining(true);
    setStatusColor('var(--cyan)');
    setHistMelhor([]);
    setHistMedia([]);
    setHistDiv([]);
    setPopulacao([]);
    setMelhorIndiv(null);
    setGlobalBest(null);
    histPopRef.current = [];
    histMelhorRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);

    const cfg: GA2Config = {
      bits: parseInt(bits),
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGeracoes),
      probCruzamento: parseFloat(probCruz),
      probMutacao: parseFloat(probMut),
      selecao: selecao as GA2Selecao,
      tamanhoTorneio: parseInt(tamTorneio),
      pontosCorte: parseInt(pontosCorte) as 1 | 2,
      elitismo: parseInt(elitismo),
      dominioMin: DOMINIO_MIN,
      dominioMax: parseFloat(dominioMax),
    };

    try {
      await apiPost('/genetico2/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      setStatusColor('var(--pink)');
      return;
    }

    closeSSE.current = apiSSE('/genetico2/train', {
      onMessage(data) {
        const step = data as GA2Step;
        histPopRef.current.push(step.populacao);
        histMelhorRef.current.push(step.melhorIndiv);

        setGeracao(step.geracao.toLocaleString());
        setMelhorX(step.melhorX.toFixed(2));
        setMelhorFx(step.melhorFx.toFixed(4));
        setDiversidade(`${step.diversidade}/${cfg.popSize}`);
        setMaxGen(step.geracao);
        if (!userScrub) {
          setDisplayGen(step.geracao);
          setPopulacao(step.populacao);
          setMelhorIndiv(step.melhorIndiv);
        }
        setGlobalBest(prev => {
          if (!prev || step.melhorIndiv.fitness > prev.fitness) return step.melhorIndiv;
          return prev;
        });
        setHistMelhor(prev => [...prev, step.melhorFx]);
        setHistMedia(prev => [...prev, step.mediaFx]);
        setHistDiv(prev => [...prev, step.diversidade]);
      },
      onDone(data) {
        const r = data as GA2Result;
        setGeracao(r.geracoes.toLocaleString());
        setMelhorX(r.melhorX.toFixed(4));
        setMelhorFx(r.melhorFx.toFixed(4));
        setMelhorIndiv(r.melhorIndiv);
        setGlobalBest(r.melhorIndiv);
        setHistMelhor(r.histMelhorFx);
        setHistMedia(r.histMediaFx);
        setHistDiv(r.histDiversidade);
        if (!userScrub) {
          setDisplayGen(r.geracoes);
          const last = histPopRef.current[r.geracoes - 1];
          if (last) setPopulacao(last);
        }
        setStatusColor('var(--primary-glow)');
        setTraining(false);
        closeSSE.current = null;
        show('GA v2 finalizado — arraste o slider pra ver a evolução');
      },
      onError() {
        setTraining(false);
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
      await apiPost('/genetico2/reset');
    } catch {
      // ignore
    }
    setGeracao('—');
    setMelhorX('—');
    setMelhorFx('—');
    setDiversidade('—');
    setStatusColor('var(--on-surface)');
    setPopulacao([]);
    setMelhorIndiv(null);
    setGlobalBest(null);
    setHistMelhor([]);
    setHistMedia([]);
    setHistDiv([]);
    histPopRef.current = [];
    histMelhorRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);
    setTraining(false);
    show('GA v2 resetado');
  }

  function handleSlider(gen: number) {
    setUserScrub(true);
    setDisplayGen(gen);
    const pop = histPopRef.current[gen - 1];
    const best = histMelhorRef.current[gen - 1];
    if (pop) setPopulacao(pop);
    if (best) setMelhorIndiv(best);
  }

  const displayStats = useMemo(() => {
    if (displayGen <= 0 || displayGen > histMelhor.length) return null;
    return {
      melhorFx: histMelhor[displayGen - 1],
      mediaFx: histMedia[displayGen - 1],
      diversidade: histDiv[displayGen - 1],
    };
  }, [displayGen, histMelhor, histMedia, histDiv]);

  // Diversidade como série pra reusar GaEvoChart (passamos como "histMedia"
  // num segundo gráfico — mas pra facilitar leitura, usamos um chart próprio
  // simples na frente). Calculamos um percentual aqui pra leitura humana.
  const diversidadePerc = useMemo(() => {
    if (!histDiv.length) return [];
    const N = parseInt(popSize) || 1;
    return histDiv.map(d => 100 * d / N);
  }, [histDiv, popSize]);

  const isTorneio = selecao === 'torneio';

  // Calcula o ótimo (mínimo) numericamente pro domínio atual amostrando 4000 pontos
  // — preciso o suficiente pra plotar uma referência visual mesmo em domínios largos.
  // Em [0, 512] isso acerta x* ≈ 420.97 / f* ≈ -418.98 (valor clássico do Schwefel 1D).
  const dominioMaxNum = parseFloat(dominioMax) || 512;
  const optimo = useMemo(() => {
    const N = 4000;
    const min = DOMINIO_MIN;
    const max = dominioMaxNum;
    const step = (max - min) / (N - 1);
    let bestX = min;
    let bestFx = Infinity;
    for (let i = 0; i < N; i++) {
      const x = min + i * step;
      const fx = -Math.abs(x * Math.sin(Math.sqrt(Math.abs(x))));
      if (fx < bestFx) {
        bestFx = fx;
        bestX = x;
      }
    }
    return { x: bestX, fx: bestFx };
  }, [dominioMaxNum]);

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Algoritmo Genético <span>v2</span></div>
          <div className="page-sub">
            Torneio, elitismo e cruzamento em 2 pontos &mdash; Aula 11
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

      {/* Config — linha 1: parâmetros básicos */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
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

      {/* Config — linha 2: parâmetros novos da aula 11 */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Método de Seleção"
            options={SELECAO_OPTIONS}
            value={selecao}
            onChange={setSelecao}
            style={{ width: '100%' }}
          />
          <div style={{
            marginTop: 10,
            opacity: isTorneio ? 1 : 0.4,
            pointerEvents: isTorneio ? 'auto' : 'none',
            transition: 'opacity 0.15s',
          }}>
            <Select
              label={`Tamanho do Torneio${isTorneio ? '' : ' (só pra torneio)'}`}
              options={TORNEIO_OPTIONS}
              value={tamTorneio}
              onChange={setTamTorneio}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Pontos de Corte (cruzamento)"
            options={PONTOS_OPTIONS}
            value={pontosCorte}
            onChange={setPontosCorte}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10 }}>
            <Select
              label="Elitismo (preserva p melhores)"
              options={ELITE_OPTIONS}
              value={elitismo}
              onChange={setElitismo}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Domínio de busca x"
            options={DOMINIO_OPTIONS}
            value={dominioMax}
            onChange={setDominioMax}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
            <div>resolução: <span style={{ color: 'var(--cyan)' }}>{(dominioMaxNum / ((1 << parseInt(bits)) - 1)).toFixed(3)}</span> por bit-step</div>
            <div>ótimo (numérico): x* ≈ <span style={{ color: 'var(--cyan)' }}>{optimo.x.toFixed(2)}</span>, f* ≈ <span style={{ color: 'var(--pink)' }}>{optimo.fx.toFixed(2)}</span></div>
            <div>casais/geração: <span style={{ color: 'var(--cyan)' }}>{Math.ceil((parseInt(popSize) - parseInt(elitismo)) / 2)}</span></div>
          </div>
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
          label={`f(x*) = ${melhorFx}`}
          color="cyan"
        />
        <MetricCard
          title="Diversidade"
          value={diversidade}
          label="x's únicos / população"
          valueStyle={{ color: statusColor }}
        />
      </div>

      {/* Main chart with replay slider */}
      <Card title="População sobre f(x)" style={{ marginBottom: 16 }}>
        <GaChart
          populacao={populacao}
          melhor={melhorIndiv}
          globalBest={globalBest}
          dominioMin={DOMINIO_MIN}
          dominioMax={dominioMaxNum}
          optimoX={optimo.x}
          optimoFx={optimo.fx}
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
                {' '}&nbsp;|&nbsp;{' '}
                <span style={{ color: 'var(--cyan)' }}>div = {displayStats.diversidade}</span>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Evolution */}
      <Card title="Evolução do f(x) — melhor e médio" style={{ marginBottom: 16 }}>
        <GaEvoChart histMelhor={histMelhor} histMedia={histMedia} />
      </Card>

      {/* Diversidade chart — usa GaEvoChart com percent (compartilha estilo) */}
      <Card title="Diversidade genética (% de x's únicos na população)" style={{ marginBottom: 16 }}>
        <GaEvoChart histMelhor={diversidadePerc} histMedia={[]} />
        <div style={{ padding: '4px 12px 8px', fontSize: 11, color: 'var(--muted)', fontFamily: 'JetBrains Mono' }}>
          A linha amarela acompanha o "melhor acumulado" da diversidade (máxima vista).
          Diversidade caindo rápido = "perda de diversidade genética" (slide aula 11): a população
          inteira convergindo num indivíduo. Roleta sem elitismo é o pior caso; torneio mantém
          diversidade mais alta.
        </div>
      </Card>

      {/* Educational details */}
      <Card title="Como funciona — adições da Aula 11">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Comparado ao v1 (aula 10), três upgrades pedidos no slide:</b>
          <ul style={{ marginLeft: 18 }}>
            <li><b>Seleção por torneio:</b> sorteia <code>k={tamTorneio}</code> indivíduos aleatórios e
              os 2 mais aptos formam o casal. Repete a cada casal. Diferente da roleta, dá
              chance pra indivíduos médios serem pais — combate o problema do "super-indivíduo"
              que monopoliza a roleta.</li>
            <li><b>Elitismo:</b> os <code>p={elitismo}</code> melhores da geração atual são
              copiados intactos pra próxima (não sofrem mutação nem cruzamento). Garante que o
              melhor encontrado nunca se perca — a estrela amarela do gráfico fica monotônica.</li>
            <li><b>Cruzamento de {pontosCorte} ponto{pontosCorte === '2' ? 's' : ''}:</b>{' '}
              {pontosCorte === '2'
                ? 'sorteia dois pontos p1 < p2 e troca os bits do segmento [p1, p2). Tende a embaralhar mais e pode acelerar a convergência.'
                : 'sorteia um ponto p e troca os bits depois dele.'}
            </li>
          </ul>
          <br />
          <b>Experimente trocar o método:</b> roleta + sem elitismo + 1 ponto = aula 10 quase puro
          (só sem o "tudo via roleta sempre"). Torneio + p≥1 + 2 pontos = configuração mais robusta.
          Compare a curva de diversidade nos dois casos pra ver o efeito visualmente.
          <br /><br />
          <b>Função objetivo (mesma da aula 10):</b> f(x) = -|x · sin(√|x|)| em <b>[{DOMINIO_MIN}, {dominioMaxNum}]</b>.
          Mínimo numérico do domínio: <b>x* ≈ {optimo.x.toFixed(2)}</b> com <b>f(x*) ≈ {optimo.fx.toFixed(2)}</b>.
          {dominioMaxNum !== 512 && (
            <>
              {' '}<b>Atenção:</b> mudar o domínio muda também o mínimo global (porque |x · sin(√|x|)| escala com |x|).
              Aumentar o domínio sem aumentar os bits piora a resolução do passo do x — bom pra discutir o
              trade-off resolução vs alcance da codificação binária.
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
