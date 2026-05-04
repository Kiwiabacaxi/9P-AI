import { useState, useRef, useEffect, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import TspMap, { TspEvoChart } from '../components/viz/TspMap';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  TspCidade, TspConfig, TspStep, TspResult,
  TspSelecao, TspCrossover, TspMutacao, TspDistMode,
} from '../api/types';

const DIST_OPTIONS = [
  { value: 'haversine',  label: 'Haversine (km reais)' },
  { value: 'euclidiana', label: 'Euclidiana (graus)' },
];

const POP_OPTIONS = [
  { value: '20',  label: '20' },
  { value: '40',  label: '40' },
  { value: '80',  label: '80' },
  { value: '160', label: '160' },
];

const GERACOES_OPTIONS = [
  { value: '50',   label: '50' },
  { value: '100',  label: '100' },
  { value: '300',  label: '300' },
  { value: '500',  label: '500' },
  { value: '1000', label: '1000' },
];

const PC_OPTIONS = [
  { value: '0.6', label: 'Pc 0.60' },
  { value: '0.7', label: 'Pc 0.70' },
  { value: '0.85', label: 'Pc 0.85' },
  { value: '0.95', label: 'Pc 0.95' },
];

const PM_OPTIONS = [
  { value: '0.05', label: 'Pm 0.05' },
  { value: '0.1', label: 'Pm 0.10' },
  { value: '0.15', label: 'Pm 0.15' },
  { value: '0.25', label: 'Pm 0.25' },
];

const SELECAO_OPTIONS = [
  { value: 'torneio', label: 'Torneio' },
  { value: 'roleta', label: 'Roleta' },
];

const TORNEIO_OPTIONS = [
  { value: '2', label: 'k = 2' },
  { value: '3', label: 'k = 3' },
  { value: '4', label: 'k = 4' },
  { value: '6', label: 'k = 6' },
];

const CRUZAMENTO_OPTIONS = [
  { value: 'ox',  label: 'OX (Order Crossover)' },
  { value: 'pmx', label: 'PMX (Partially Mapped)' },
];

const MUTACAO_OPTIONS = [
  { value: 'inversao', label: 'Inversão (2-opt)' },
  { value: 'swap',     label: 'Swap (troca)' },
];

const ELITE_OPTIONS = [
  { value: '0', label: 'sem elite' },
  { value: '1', label: 'p = 1' },
  { value: '2', label: 'p = 2' },
  { value: '4', label: 'p = 4' },
];

export default function TspView() {
  const { show } = useToast();

  // Cidades
  const [cidades, setCidades] = useState<TspCidade[]>([]);
  const [distMode, setDistMode] = useState<TspDistMode>('haversine');
  const [matrizPronta, setMatrizPronta] = useState(false);

  // Config GA
  const [popSize, setPopSize] = useState('80');
  const [maxGeracoes, setMaxGeracoes] = useState('300');
  const [probCruz, setProbCruz] = useState('0.85');
  const [probMut, setProbMut] = useState('0.15');
  const [selecao, setSelecao] = useState<TspSelecao>('torneio');
  const [tamTorneio, setTamTorneio] = useState('4');
  const [cruzamento, setCruzamento] = useState<TspCrossover>('ox');
  const [mutacao, setMutacao] = useState<TspMutacao>('inversao');
  const [elitismo, setElitismo] = useState('2');

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorDist, setMelhorDist] = useState<string>('—');
  const [diversidade, setDiversidade] = useState<string>('—');

  // Animation
  const [tourAtual, setTourAtual] = useState<number[]>([]);
  const [tourGlobal, setTourGlobal] = useState<number[]>([]);
  const [globalDist, setGlobalDist] = useState<number | null>(null);

  // History
  const [histMelhor, setHistMelhor] = useState<number[]>([]);
  const [histMedia, setHistMedia] = useState<number[]>([]);

  // Replay
  const histTourRef = useRef<number[][]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0);
  const [userScrub, setUserScrub] = useState(false);

  const closeSSE = useRef<(() => void) | null>(null);

  // Carrega o preset capitais BR no mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const preset = await apiGet<TspCidade[]>('/tsp/preset?name=capitais-br');
        if (cancelled) return;
        await apiPost('/tsp/cities', preset);
        await apiPost('/tsp/distancias', { modo: 'haversine' });
        if (cancelled) return;
        setCidades(preset);
        setMatrizPronta(true);
      } catch (e) {
        show('Erro ao carregar cidades: ' + (e instanceof Error ? e.message : String(e)));
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Quando muda o modo de distância, recalcula a matriz.
  async function handleDistModeChange(novo: string) {
    const modo = novo as TspDistMode;
    setDistMode(modo);
    setMatrizPronta(false);
    try {
      await apiPost('/tsp/distancias', { modo });
      setMatrizPronta(true);
      show(`Matriz recalculada (${modo})`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  const unidade = distMode === 'haversine' ? 'km' : 'graus';

  async function handleTrain() {
    if (!matrizPronta) {
      show('Matriz não calculada — recarregue a página');
      return;
    }
    setTraining(true);
    setHistMelhor([]);
    setHistMedia([]);
    setTourAtual([]);
    setTourGlobal([]);
    setGlobalDist(null);
    histTourRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);

    const cfg: TspConfig = {
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGeracoes),
      probCruzamento: parseFloat(probCruz),
      probMutacao: parseFloat(probMut),
      selecao,
      tamanhoTorneio: parseInt(tamTorneio),
      cruzamento,
      mutacao,
      elitismo: parseInt(elitismo),
    };

    try {
      await apiPost('/tsp/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      return;
    }

    closeSSE.current = apiSSE('/tsp/train', {
      onMessage(data) {
        const step = data as TspStep;
        histTourRef.current.push(step.melhorTour);

        setGeracao(step.geracao.toLocaleString());
        setMelhorDist(`${step.melhorDist.toFixed(1)} ${unidade}`);
        setDiversidade(`${step.diversidade}/${cfg.popSize}`);
        setMaxGen(step.geracao);
        if (!userScrub) {
          setDisplayGen(step.geracao);
          setTourAtual(step.melhorTour);
        }
        setTourGlobal(step.melhorGlobal);
        setGlobalDist(step.melhorGlobalDist);

        setHistMelhor(prev => [...prev, step.melhorDist]);
        setHistMedia(prev => [...prev, step.mediaDist]);
      },
      onDone(data) {
        const r = data as TspResult;
        setGeracao(r.geracoes.toLocaleString());
        setMelhorDist(`${r.melhorDist.toFixed(1)} ${unidade}`);
        setTourGlobal(r.melhorTour);
        setGlobalDist(r.melhorDist);
        setHistMelhor(r.histMelhor);
        setHistMedia(r.histMedia);
        if (!userScrub) {
          setDisplayGen(r.geracoes);
          const last = histTourRef.current[r.geracoes - 1];
          if (last) setTourAtual(last);
        }
        setTraining(false);
        closeSSE.current = null;
        show(`Tour final: ${r.melhorDist.toFixed(1)} ${unidade}`);
      },
      onError() {
        setTraining(false);
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
      await apiPost('/tsp/reset');
    } catch {
      // ignore
    }
    setGeracao('—');
    setMelhorDist('—');
    setDiversidade('—');
    setTourAtual([]);
    setTourGlobal([]);
    setGlobalDist(null);
    setHistMelhor([]);
    setHistMedia([]);
    histTourRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);
    setTraining(false);
    show('TSP resetado');
  }

  function handleSlider(gen: number) {
    setUserScrub(true);
    setDisplayGen(gen);
    const t = histTourRef.current[gen - 1];
    if (t) setTourAtual(t);
  }

  const isTorneio = selecao === 'torneio';

  // Cidades em ordem do tour atual — pra exibir lista lateral
  const cidadesNaOrdem = useMemo(() => {
    if (tourAtual.length === 0) return [];
    const byId = new Map(cidades.map(c => [c.id, c]));
    return tourAtual.map(id => byId.get(id)).filter(Boolean) as TspCidade[];
  }, [tourAtual, cidades]);

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Caixeiro <span>Viajante</span></div>
          <div className="page-sub">
            Encontrando o tour mais curto pelas 27 capitais brasileiras &mdash; Aula 12
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
          <button className="btn btn-primary" onClick={handleTrain} disabled={training || !matrizPronta}>
            {training && <span className="spin" />}
            OTIMIZAR
          </button>
        </div>
      </div>

      {/* Config — linha 1: GA params */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Modo de distância"
            options={DIST_OPTIONS}
            value={distMode}
            onChange={handleDistModeChange}
            style={{ width: '100%' }}
          />
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
            label="População"
            options={POP_OPTIONS}
            value={popSize}
            onChange={setPopSize}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
            <Select options={PC_OPTIONS} value={probCruz} onChange={setProbCruz} style={{ flex: 1 }} />
            <Select options={PM_OPTIONS} value={probMut} onChange={setProbMut} style={{ flex: 1 }} />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Seleção"
            options={SELECAO_OPTIONS}
            value={selecao}
            onChange={(v) => setSelecao(v as TspSelecao)}
            style={{ width: '100%' }}
          />
          <div style={{
            marginTop: 10,
            opacity: isTorneio ? 1 : 0.4,
            pointerEvents: isTorneio ? 'auto' : 'none',
          }}>
            <Select
              label="Tamanho do torneio"
              options={TORNEIO_OPTIONS}
              value={tamTorneio}
              onChange={setTamTorneio}
              style={{ width: '100%' }}
            />
          </div>
        </Card>
      </div>

      {/* Config — linha 2: operadores específicos de permutação */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Cruzamento"
            options={CRUZAMENTO_OPTIONS}
            value={cruzamento}
            onChange={(v) => setCruzamento(v as TspCrossover)}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10 }}>
            <Select
              label="Mutação"
              options={MUTACAO_OPTIONS}
              value={mutacao}
              onChange={(v) => setMutacao(v as TspMutacao)}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Elitismo"
            options={ELITE_OPTIONS}
            value={elitismo}
            onChange={setElitismo}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)', fontFamily: 'JetBrains Mono' }}>
            cidades: <span style={{ color: 'var(--cyan)' }}>{cidades.length}</span>
            {' '}&middot;{' '} tours possíveis: <span style={{ color: 'var(--cyan)' }}>
              {cidades.length > 0 ? `${cidades.length}!` : '—'}
            </span>
            <br />
            (pra 27, isso é {'≈'} 1.09e28 — busca exaustiva impossível)
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Melhor global encontrado</div>
          {globalDist !== null ? (
            <div style={{ fontSize: 12, lineHeight: 1.7, fontFamily: 'JetBrains Mono' }}>
              <div>distância: <code style={{ color: 'var(--pink)' }}>
                {globalDist.toFixed(1)} {unidade}
              </code></div>
              <div>tour de <code>{tourGlobal.length}</code> cidades</div>
              <div style={{ color: 'var(--muted)' }}>
                início: {tourGlobal.length > 0 ? cidades.find(c => c.id === tourGlobal[0])?.nome ?? '—' : '—'}
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>aguardando otimização&hellip;</div>
          )}
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Geração"
          value={geracao}
          label="iteração atual"
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Distância"
          value={melhorDist}
          label="tour mais curto da geração"
          color="cyan"
        />
        <MetricCard
          title="Diversidade"
          value={diversidade}
          label="tours únicos / população"
        />
      </div>

      {/* Mapa principal + slider de replay */}
      <Card title="Tour atual no mapa" style={{ marginBottom: 16 }}>
        <TspMap
          cidades={cidades}
          tour={tourAtual}
          globalTour={tourGlobal}
          height={500}
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
                const last = histTourRef.current[maxGen - 1];
                if (last) setTourAtual(last);
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
              minWidth: 120,
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
            {displayGen > 0 && histMelhor[displayGen - 1] !== undefined && (
              <div style={{
                fontSize: 11,
                fontFamily: 'JetBrains Mono',
                color: 'var(--muted)',
                whiteSpace: 'nowrap',
              }}>
                dist = <span style={{ color: 'var(--pink)' }}>
                  {histMelhor[displayGen - 1].toFixed(1)} {unidade}
                </span>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Chart de evolução */}
      <Card title={`Evolução da distância (${unidade})`} style={{ marginBottom: 16 }}>
        <TspEvoChart histMelhor={histMelhor} histMedia={histMedia} unidade={unidade} />
      </Card>

      {/* Lista do tour atual (texto) */}
      {cidadesNaOrdem.length > 0 && (
        <Card title="Sequência do tour exibido" style={{ marginBottom: 16 }}>
          <div style={{
            padding: 12, fontSize: 12, fontFamily: 'JetBrains Mono',
            color: 'var(--muted)', lineHeight: 1.7,
          }}>
            {cidadesNaOrdem.map((c, i) => (
              <span key={c.id}>
                <span style={{ color: i === 0 ? 'var(--pink)' : 'var(--cyan)' }}>
                  {i + 1}. {c.nome}{c.uf ? `/${c.uf}` : ''}
                </span>
                {i < cidadesNaOrdem.length - 1 && <span style={{ color: '#444' }}> → </span>}
              </span>
            ))}
            <span style={{ color: '#444' }}> → </span>
            <span style={{ color: 'var(--pink)' }}>{cidadesNaOrdem[0].nome} (volta)</span>
          </div>
        </Card>
      )}

      {/* Educational */}
      <Card title="Por que TSP precisa de operadores especiais">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Encoding por permutação:</b> cada indivíduo é uma sequência de N índices de cidades, cada um aparecendo
          exatamente uma vez. É o tipo de codificação que o slide aula 12 chama atenção — diferente da bit-string
          das aulas 10-11.
          <br /><br />
          <b>Por que cruzamento de 1/2 pontos quebra:</b> recortar e trocar segmentos entre dois pais gera filhos com
          cidades repetidas e outras faltando — permutação inválida. Por isso usamos:
          <ul style={{ marginLeft: 18 }}>
            <li><b>OX (Order Crossover):</b> copia um segmento de p1 e completa o resto na ordem em que aparecem em p2 (pulando duplicatas). Preserva validade.</li>
            <li><b>PMX (Partially Mapped):</b> copia o segmento de p1 e mapeia as cidades de p2 que ainda faltam, seguindo a "cadeia" do segmento.</li>
          </ul>
          <b>Mutação:</b> <i>swap</i> troca duas cidades; <i>inversão</i> reverte um segmento — esta é equivalente a um movimento <b>2-opt</b> e tipicamente bem mais efetiva pra TSP, porque conserta "cruzamentos" no tour de uma vez.
          <br /><br />
          <b>Distância:</b> com <i>Haversine</i>, calculamos a distância de great-circle (km reais entre as capitais). A
          fitness = -distância (quanto menor, melhor). Roleta normaliza via (max - dist) pra trabalhar com valores positivos.
          <br /><br />
          <b>Por que AG aqui:</b> 27 capitais geram 27! ≈ 10²⁸ tours possíveis. Busca exaustiva é absurda; o AG acha
          tours quase-ótimos em segundos. É exatamente o cenário que o slide aula 12 destaca.
        </div>
      </Card>
    </div>
  );
}
