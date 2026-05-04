import { useState, useRef, useEffect, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import TspMap, { TspEvoChart } from '../components/viz/TspMap';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  TspCidade, TspConfig, TspStep, TspResult, TspRouteGeometry,
  TspSelecao, TspCrossover, TspMutacao, TspDistMode,
} from '../api/types';

const PRESET_OPTIONS = [
  { value: 'triangulo',   label: 'Triângulo Mineiro · 20 cidades' },
  { value: 'capitais-br', label: '27 Capitais BR' },
];

const PRESET_INFO: Record<string, { titulo: string; sub: string }> = {
  'triangulo': {
    titulo: 'Logística regional',
    sub: 'Roteirização de entregas no Triângulo Mineiro / Alto Paranaíba (MG)',
  },
  'capitais-br': {
    titulo: 'Tour pelas capitais',
    sub: 'Encontrando o tour mais curto pelas 27 capitais brasileiras',
  },
};

const DIST_OPTIONS = [
  { value: 'haversine',  label: 'Haversine (linha reta)' },
  { value: 'osrm',       label: 'OSRM (estrada real)' },
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

const LAMBDA_OPTIONS = [
  { value: '0',   label: 'λ = 0  (TSP puro)' },
  { value: '0.5', label: 'λ = 0.5' },
  { value: '1',   label: 'λ = 1' },
  { value: '2',   label: 'λ = 2' },
  { value: '5',   label: 'λ = 5  (penaliza forte)' },
];

function fatorial(n: number): number {
  if (n <= 1) return 1;
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function formatFatorialAprox(n: number): string {
  if (n <= 1) return '1';
  return fatorial(n).toExponential(2).replace('e+', ' · 10^');
}

export default function TspView() {
  const { show } = useToast();

  // Cidades
  const [preset, setPreset] = useState<string>('triangulo');
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
  const [lambdaMaxLeg, setLambdaMaxLeg] = useState('0');

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorDist, setMelhorDist] = useState<string>('—');
  const [melhorMaxLeg, setMelhorMaxLeg] = useState<string>('—');
  const [melhorCusto, setMelhorCusto] = useState<string>('—');
  const [diversidade, setDiversidade] = useState<string>('—');

  // Animation
  const [tourAtual, setTourAtual] = useState<number[]>([]);
  const [tourGlobal, setTourGlobal] = useState<number[]>([]);
  const [globalDist, setGlobalDist] = useState<number | null>(null);

  // Geometry curvada (OSRM) — só preenchida quando dist mode == 'osrm' e há tour
  const [routeGeometry, setRouteGeometry] = useState<TspRouteGeometry | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);

  // History
  const [histMelhor, setHistMelhor] = useState<number[]>([]);
  const [histMedia, setHistMedia] = useState<number[]>([]);

  // Replay
  const histTourRef = useRef<number[][]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0);
  const [userScrub, setUserScrub] = useState(false);

  const closeSSE = useRef<(() => void) | null>(null);

  // Carrega preset + recalcula matriz. Reusa para mount e troca de preset.
  async function carregarPreset(name: string, modo: TspDistMode) {
    setMatrizPronta(false);
    const cs = await apiGet<TspCidade[]>(`/tsp/preset?name=${encodeURIComponent(name)}`);
    await apiPost('/tsp/cities', cs);
    await apiPost('/tsp/distancias', { modo });
    setCidades(cs);
    setMatrizPronta(true);
    setRouteGeometry(null); // cidades trocaram → geometria salva ficou inválida
    return cs;
  }

  // Busca a geometria curvada da rota (OSRM) — chamada após otimização finalizar.
  async function fetchRouteGeometry(tour: number[]) {
    if (distMode !== 'osrm' || tour.length < 3) return;
    setGeometryLoading(true);
    try {
      const geo = await apiPost<TspRouteGeometry>('/tsp/geometry', { tour });
      setRouteGeometry(geo);
      show(`Geometria real: ${geo.distancia.toFixed(1)} km via estradas (${(geo.duracao / 3600).toFixed(1)} h dirigindo)`);
    } catch (e) {
      show('Falha ao obter geometria OSRM: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setGeometryLoading(false);
    }
  }

  // Mount: carrega o preset default.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await carregarPreset('triangulo', 'haversine');
        if (cancelled) return;
      } catch (e) {
        show('Erro ao carregar cidades: ' + (e instanceof Error ? e.message : String(e)));
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handlePresetChange(novo: string) {
    setPreset(novo);
    // limpa qualquer treino anterior e zera o mapa
    handleResetSilent();
    try {
      await carregarPreset(novo, distMode);
      show(`Preset carregado: ${PRESET_INFO[novo]?.titulo ?? novo}`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  // Quando muda o modo de distância, recalcula a matriz.
  async function handleDistModeChange(novo: string) {
    const modo = novo as TspDistMode;
    setDistMode(modo);
    setMatrizPronta(false);
    setRouteGeometry(null); // muda distância → geometria salva pode não corresponder
    try {
      await apiPost('/tsp/distancias', { modo });
      setMatrizPronta(true);
      show(modo === 'osrm'
        ? 'Matriz OSRM cacheada (estradas reais). Clique OTIMIZAR.'
        : `Matriz recalculada (${modo})`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  // versão silenciosa do reset (sem chamar API/toast — só limpa estado local).
  function handleResetSilent() {
    if (closeSSE.current) {
      closeSSE.current();
      closeSSE.current = null;
    }
    setRouteGeometry(null);
    setGeracao('—');
    setMelhorDist('—');
    setMelhorMaxLeg('—');
    setMelhorCusto('—');
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
  }

  const unidade = distMode === 'euclidiana' ? 'graus' : 'km';

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
      lambdaMaxLeg: parseFloat(lambdaMaxLeg),
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
        setMelhorMaxLeg(`${step.melhorMaxLeg.toFixed(1)} ${unidade}`);
        setMelhorCusto(`${step.melhorCusto.toFixed(1)} ${unidade}`);
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
        setMelhorMaxLeg(`${r.melhorMaxLeg.toFixed(1)} ${unidade}`);
        setMelhorCusto(`${r.melhorCusto.toFixed(1)} ${unidade}`);
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
        if (distMode === 'osrm') {
          void fetchRouteGeometry(r.melhorTour);
        }
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
    setMelhorMaxLeg('—');
    setMelhorCusto('—');
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
            {PRESET_INFO[preset]?.sub ?? 'Roteirização com AG'} &mdash; Aula 12
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
            label="Cenário (dataset)"
            options={PRESET_OPTIONS}
            value={preset}
            onChange={handlePresetChange}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10 }}>
            <Select
              label="Modo de distância"
              options={DIST_OPTIONS}
              value={distMode}
              onChange={handleDistModeChange}
              style={{ width: '100%' }}
            />
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
            ({'≈'} <span style={{ color: 'var(--cyan)' }}>{formatFatorialAprox(cidades.length)}</span> — busca exaustiva inviável)
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

      {/* Função de Fitness — explícita + tempero */}
      <Card title="Função de fitness" style={{ marginBottom: 16 }}>
        <div style={{ padding: '8px 16px 4px' }}>
          <div style={{
            background: 'var(--surface-2)', borderRadius: 6,
            padding: '12px 16px', marginBottom: 10,
            fontFamily: 'JetBrains Mono', fontSize: 13, lineHeight: 1.7,
            color: 'var(--on-surface)',
          }}>
            <div>fitness(tour) = <span style={{ color: 'var(--cyan)' }}>−custo(tour)</span></div>
            <div>custo(tour) = <span style={{ color: 'var(--pink)' }}>Σ d(c<sub>i</sub>, c<sub>i+1</sub>)</span> + <span style={{ color: 'var(--pink)' }}>λ · max d(c<sub>i</sub>, c<sub>i+1</sub>)</span></div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
              soma fechada (volta a c<sub>0</sub>); d = haversine ou euclidiana, conforme o modo escolhido
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ flex: '0 0 auto', minWidth: 200 }}>
              <Select
                label="λ (tempero — penaliza max-leg)"
                options={LAMBDA_OPTIONS}
                value={lambdaMaxLeg}
                onChange={setLambdaMaxLeg}
                style={{ width: '100%' }}
              />
            </div>
            <div style={{
              flex: 1, fontSize: 12, color: 'var(--muted)',
              fontFamily: 'JetBrains Mono', lineHeight: 1.6, minWidth: 280,
            }}>
              <b>λ = 0:</b> TSP clássico — só minimiza distância total.<br />
              <b>λ &gt; 0:</b> também penaliza tours com algum trecho muito longo.
              Útil pra cenários reais onde o motorista tem autonomia limitada,
              precisa parar pra descanso ou prefere balancear paradas.
              {parseFloat(lambdaMaxLeg) > 0 && (
                <>
                  <br />
                  <span style={{ color: 'var(--cyan)' }}>
                    Cada {unidade === 'km' ? 'km' : 'grau'} a mais no maior trecho custa <b>{lambdaMaxLeg} {unidade}</b> equivalentes na fitness.
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </Card>

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
          title={parseFloat(lambdaMaxLeg) > 0 ? 'Custo' : 'Distância'}
          value={parseFloat(lambdaMaxLeg) > 0 ? melhorCusto : melhorDist}
          label={parseFloat(lambdaMaxLeg) > 0
            ? `dist + λ·max-leg (real: ${melhorDist})`
            : 'tour mais curto da geração'}
          color="cyan"
        />
        <MetricCard
          title="Maior trecho"
          value={melhorMaxLeg}
          label="leg mais longo do tour"
        />
      </div>

      {/* Diversidade fica numa linha solo (compacta) */}
      <div style={{
        display: 'flex', justifyContent: 'flex-end',
        marginBottom: 16, fontSize: 11,
        fontFamily: 'JetBrains Mono', color: 'var(--muted)',
      }}>
        diversidade da população: <span style={{ color: 'var(--cyan)', marginLeft: 6 }}>{diversidade}</span> tours únicos
      </div>

      {/* Mapa principal + slider de replay */}
      <Card
        title={routeGeometry
          ? `Tour atual no mapa (estradas reais — ${routeGeometry.distancia.toFixed(0)} km / ${(routeGeometry.duracao / 3600).toFixed(1)} h)`
          : 'Tour atual no mapa'}
        style={{ marginBottom: 16 }}
      >
        {geometryLoading && (
          <div style={{
            padding: '6px 12px', marginBottom: 8,
            background: 'var(--surface-2)', borderRadius: 6,
            fontSize: 11, color: 'var(--cyan)', fontFamily: 'JetBrains Mono',
          }}>
            <span className="spin" /> consultando OSRM…
          </div>
        )}
        <TspMap
          cidades={cidades}
          tour={tourAtual}
          globalTour={tourGlobal}
          routeGeometry={routeGeometry ?? undefined}
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

      {/* Logistics context — por que esse cenário foi escolhido */}
      {preset === 'triangulo' && (
        <Card title="Por que Triângulo Mineiro?" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
            Em vez do tour acadêmico ligando capitais por linhas reta, este preset modela um cenário que
            <b> empresas brasileiras resolvem todo dia</b>: uma frota saindo de um centro de distribuição
            (Uberlândia) atendendo lojas/clientes em outras cidades da região.
            <br /><br />
            <b>Por que essa região:</b> o Triângulo Mineiro / Alto Paranaíba é um dos hubs logísticos
            mais densos do interior do Brasil. Concentra:
            <ul style={{ marginLeft: 18 }}>
              <li><b>Frigoríficos</b> — JBS e Marfrig em Uberlândia/Uberaba coletam gado de fazendas espalhadas pelo interior e distribuem carne;</li>
              <li><b>Fertilizantes</b> — <i>Mosaic Fertilizantes</i> em Araxá distribui pra fazendas em todo o Triângulo;</li>
              <li><b>Cooperativas leiteiras</b> — caminhões da CCPR/Itambé fazem rota diária por dezenas de fazendas pra coletar leite;</li>
              <li><b>Varejo regional</b> — redes como <i>Bretas</i>, <i>Mais Mart</i> e <i>Bahamas</i> têm CDs em Uberlândia abastecendo lojas em cidades vizinhas;</li>
              <li><b>Bebidas e combustíveis</b> — Coca-Cola FEMSA e BR Distribuidora operam corredores BR-050 / BR-262 / BR-365.</li>
            </ul>
            <br />
            Distâncias entre as cidades aqui ficam na faixa de <b>50–300 km</b> — escala onde diferenças de
            ordem mudam <b>centenas de reais</b> em diesel por dia. É o cenário em que TSP/VRP
            <i> realmente se paga</i>, e por isso é caso clássico em pesquisa em engenharia de produção
            no Brasil (Linden 2008; literatura de roteirização).
            <br /><br />
            <b>Note no mapa:</b> o tour ótimo costuma formar um caminho que evita zigue-zagues entre o
            sul (Uberaba/Sacramento) e o leste (Araxá/Patos de Minas) — exatamente o que um motorista
            humano experiente faria intuitivamente. O AG redescobre essa intuição a partir do zero.
          </div>
        </Card>
      )}

      {preset === 'capitais-br' && (
        <Card title="Sobre o cenário das capitais" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
            Este preset é mais <b>didático</b> que real — distâncias inter-capitais são da ordem de
            milhares de km e nenhuma frota terrestre faz exatamente esse percurso. Mas serve como
            referência: 27 cidades é o limite onde busca exaustiva ainda <i>"quase"</i> faz sentido na
            cabeça (e o AG já mostra ganho dramático).
            <br /><br />
            Pra ver o uso real do TSP em logística brasileira, troque pra <b>Triângulo Mineiro</b> no seletor
            de cenário lá em cima.
          </div>
        </Card>
      )}

      {/* Clarification: "passar por" vs "visitar" */}
      <Card title='Esclarecimento: "passar por" outras cidades' style={{ marginBottom: 16 }}>
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          Pergunta natural: "uma rota A→C que naturalmente passa por B (porque B
          fica no caminho) — o algoritmo aproveita isso?"
          <br /><br />
          <b>Em parte:</b> o tour ótimo <i>geralmente já visita B entre A e C na sequência</i>,
          porque é o que minimiza distância total. Você vê isso visualmente quando o tour
          "abraça" o mapa em ordem geográfica e não dá ziguezagues.
          <br /><br />
          <b>Mas o algoritmo NÃO atravessa fisicamente B sem visitá-lo.</b> No TSP cada cidade
          é visitada exatamente uma vez, na ordem do tour. A distância A→C é calculada
          ponta-a-ponta pela função de distância (Haversine no modo atual).
          <br /><br />
          <b>Limitação atual (Haversine):</b> calcula a linha reta entre lat/lng. Não conhece
          rodovias, relevo, obras na BR-050. Pode <i>subestimar</i> a distância real entre
          duas cidades onde o caminho de carro contorna serra, atravessa rio, etc.
          <br /><br />
          <b>Fase 2 (próxima — OSRM):</b> a matriz vai ser preenchida com <b>distâncias reais
          por estrada</b> calculadas pelo OpenStreetMap Routing Machine. Aí o tour mais curto
          em km pode parecer não-óbvio no mapa em linha reta — mas reflete o que sai do
          tanque de combustível na vida real. O <b>desenho da rota</b> também vira a
          geometria curvada das ruas/rodovias, não mais segmentos diretos.
        </div>
      </Card>

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
          <b>Distância:</b> com <i>Haversine</i>, calculamos a distância de great-circle (km reais entre as cidades).
          A fitness = -distância (quanto menor, melhor). Roleta normaliza via (max - dist) pra trabalhar com valores positivos.
          <br /><br />
          <b>Por que AG aqui:</b> {cidades.length} cidades geram {cidades.length}! ≈ {formatFatorialAprox(cidades.length)} tours possíveis.
          Busca exaustiva é absurda mesmo pra esse N pequeno; o AG acha tours quase-ótimos em segundos.
          É exatamente o cenário que o slide aula 12 destaca: <i>"a grande utilidade dos AGs está em resolver
          problemas onde os métodos exaustivos não conseguem chegar à solução em um tempo razoável."</i>
        </div>
      </Card>
    </div>
  );
}
