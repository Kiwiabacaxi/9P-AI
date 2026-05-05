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
  TspPreset, TspPresetMeta,
} from '../api/types';

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

const GAMMA_OPTIONS = [
  { value: '0',   label: 'γ = 0  (sem peso)' },
  { value: '15',  label: 'γ = 15' },
  { value: '20',  label: 'γ = 20' },
  { value: '30',  label: 'γ = 30' },
  { value: '50',  label: 'γ = 50  (cold-chain)' },
  { value: '60',  label: 'γ = 60' },
  { value: '100', label: 'γ = 100  (tempo manda)' },
];

const MU_OPTIONS = [
  { value: '10',  label: 'μ = 10  (leve)' },
  { value: '30',  label: 'μ = 30' },
  { value: '50',  label: 'μ = 50  (médio)' },
  { value: '80',  label: 'μ = 80  (forte)' },
  { value: '150', label: 'μ = 150  (proibitivo)' },
];

const LAMBDA_OPTIONS = [
  { value: '0',   label: 'λ = 0  (TSP puro)' },
  { value: '0.5', label: 'λ = 0.5' },
  { value: '1',   label: 'λ = 1' },
  { value: '1.5', label: 'λ = 1.5' },
  { value: '2',   label: 'λ = 2' },
  { value: '3',   label: 'λ = 3' },
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
  const [preset, setPreset] = useState<string>('itambe-leite');
  const [presets, setPresets] = useState<TspPresetMeta[]>([]);
  const [presetMeta, setPresetMeta] = useState<TspPreset | null>(null);
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
  // Restrição lógica do cenário: cidade que deve ser visitada por último
  // (ex.: porto de descarga). -1 = sem restrição. Vem do preset.
  const [lastVisit, setLastVisit] = useState<number>(-1);
  // γ — peso do tempo na fitness (km equiv por hora). Vem do preset.
  const [gamma, setGamma] = useState<string>('0');
  // μ — coef. da penalidade de overtime (jornada > 10h). Vem do preset.
  const [muOvertime, setMuOvertime] = useState<string>('0');
  // Toggle do limite de jornada (10h). Quando off, μ=0 efetivo.
  const [jornadaAtiva, setJornadaAtiva] = useState<boolean>(false);

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorDist, setMelhorDist] = useState<string>('—');
  const [melhorMaxLeg, setMelhorMaxLeg] = useState<string>('—');
  const [melhorTempo, setMelhorTempo] = useState<string>('—');
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
  // Snapshot do histórico de tours pra passar como prop ao TspMap. Só populado
  // ao final do treino — durante o treino o ref muta mas a prop não precisa
  // re-renderizar (all-gens animation só faz sentido depois de done).
  const [histToursDone, setHistToursDone] = useState<number[][]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0);
  const [userScrub, setUserScrub] = useState(false);

  // Evolution playback — auto-advance do slider, animando o melhor de cada
  // geração (a rota se "estabiliza" enquanto o GA evolui).
  const [playingEvo, setPlayingEvo] = useState(false);
  const [evoSpeed, setEvoSpeed] = useState(20); // gerações por segundo

  useEffect(() => {
    if (!playingEvo || maxGen === 0) return;
    let last = performance.now();
    // Se já estamos no fim, recomeça do 1; senão continua de onde está.
    let acc = displayGen >= maxGen ? 1 : Math.max(1, displayGen);
    let raf = 0;
    // Variável local (não ref compartilhado) — evita race em strict mode
    // onde dois efeitos podem coexistir momentaneamente.

    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      acc += dt * evoSpeed;
      const gen = Math.max(1, Math.min(Math.floor(acc), maxGen));
      setDisplayGen(gen);
      const t = histTourRef.current[gen - 1];
      if (t) setTourAtual(t);
      if (acc >= maxGen) {
        setPlayingEvo(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      if (raf) cancelAnimationFrame(raf);
    };
    // displayGen lido só no setup (não disparar re-effect a cada frame).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playingEvo, evoSpeed, maxGen]);

  const closeSSE = useRef<(() => void) | null>(null);

  // Carrega preset + aplica settings sugeridos + recalcula matriz.
  // O preset agora é um objeto completo (com cidades + narrativa + sugestões).
  async function carregarPreset(name: string, opts?: { aplicarSugestoes?: boolean }) {
    setMatrizPronta(false);
    const p = await apiGet<TspPreset>(`/tsp/preset?name=${encodeURIComponent(name)}`);
    await apiPost('/tsp/cities', p.cidades);
    // Modo de distância: aplica o sugerido (na primeira carga ou troca de preset),
    // ou mantém o que o usuário tinha selecionado (em recarregamento manual).
    const modoEfetivo = opts?.aplicarSugestoes !== false ? p.modoSugerido : distMode;
    await apiPost('/tsp/distancias', { modo: modoEfetivo });
    setCidades(p.cidades);
    setPresetMeta(p);
    setMatrizPronta(true);
    setRouteGeometry(null);
    if (opts?.aplicarSugestoes !== false) {
      setDistMode(modoEfetivo);
      setLambdaMaxLeg(String(p.lambdaSugerido));
      setGamma(String(p.gammaSugerido ?? 0));
      setMuOvertime(String(p.muOvertimeSugerido ?? 0));
      setJornadaAtiva((p.muOvertimeSugerido ?? 0) > 0);
    }
    // A restrição "última visita" é parte da definição do cenário, não uma
    // sugestão — sempre aplica.
    setLastVisit(typeof p.lastVisit === 'number' ? p.lastVisit : -1);
    return p;
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

  // Mount: busca a lista de presets + carrega o default.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await apiGet<TspPresetMeta[]>('/tsp/presets');
        if (cancelled) return;
        setPresets(list);
        await carregarPreset('itambe-leite', { aplicarSugestoes: true });
        if (cancelled) return;
      } catch (e) {
        show('Erro ao carregar cenários: ' + (e instanceof Error ? e.message : String(e)));
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handlePresetChange(novo: string) {
    setPreset(novo);
    handleResetSilent();
    try {
      const p = await carregarPreset(novo, { aplicarSugestoes: true });
      show(`Cenário: ${p.nome} — ${p.cidades.length} pontos`);
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
    setMelhorTempo('—');
    setMelhorCusto('—');
    setDiversidade('—');
    setTourAtual([]);
    setTourGlobal([]);
    setGlobalDist(null);
    setHistMelhor([]);
    setHistMedia([]);
    histTourRef.current = [];
    setHistToursDone([]);
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
    setHistToursDone([]);
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
      lastVisit,
      gamma: parseFloat(gamma),
      jornadaMaxSec: 36000, // 10h fixo (regulamentação ANTT)
      muOvertime: jornadaAtiva ? parseFloat(muOvertime) : 0,
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
        setMelhorTempo(step.melhorTempoSec > 0
          ? `${(step.melhorTempoSec / 3600).toFixed(1)} h`
          : '—');
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
        setMelhorTempo(r.melhorTempoSec > 0
          ? `${(r.melhorTempoSec / 3600).toFixed(1)} h`
          : '—');
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
        // Snapshot do histórico de tours pra liberar a animação "todas gerações"
        // no TspMap (durante treino o ref muta sem disparar re-render).
        setHistToursDone(histTourRef.current.slice());
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
    setMelhorTempo('—');
    setMelhorCusto('—');
    setDiversidade('—');
    setTourAtual([]);
    setTourGlobal([]);
    setGlobalDist(null);
    setHistMelhor([]);
    setHistMedia([]);
    histTourRef.current = [];
    setHistToursDone([]);
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
            {presetMeta?.descricao ?? 'Roteirização com AG'}
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
            label="Cenário"
            options={presets.map(p => ({ value: p.id, label: p.nome }))}
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
          <div style={{ marginTop: 10 }}>
            <div className="imgreg-select-label">
              Pc · Pm <span style={{ color: 'var(--muted)', fontWeight: 400 }}>(prob. cruzamento · mutação)</span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <Select options={PC_OPTIONS} value={probCruz} onChange={setProbCruz} style={{ flex: 1 }} />
              <Select options={PM_OPTIONS} value={probMut} onChange={setProbMut} style={{ flex: 1 }} />
            </div>
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
            <div>custo(tour) = <span style={{ color: 'var(--pink)' }}>Σ d(c<sub>i</sub>, c<sub>i+1</sub>)</span>
              {' '}+ <span style={{ color: 'var(--pink)' }}>λ · max d(c<sub>i</sub>, c<sub>i+1</sub>)</span>
              {lastVisit >= 0 && (
                <> {' '}+ <span style={{ color: '#ffaa00' }}>ω · desvio<sub>lastVisit</sub></span></>
              )}
              {parseFloat(gamma) > 0 && (
                <> {' '}+ <span style={{ color: 'var(--primary-glow)' }}>γ · T<sub>h</sub></span></>
              )}
              {jornadaAtiva && parseFloat(muOvertime) > 0 && (
                <> {' '}+ <span style={{ color: '#ff8800' }}>μ · max(0, T − T<sub>max</sub>)²</span></>
              )}
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
              soma fechada (volta a c<sub>0</sub>); d em km (haversine/OSRM) ou graus (euclidiana);
              T = tempo em horas (real OSRM ou sintetizado a 70 km/h)
            </div>
          </div>

          {/* Linha 1: λ + γ */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <div style={{ flex: '0 0 auto', minWidth: 200 }}>
              <Select
                label="λ (penaliza max-leg)"
                options={LAMBDA_OPTIONS}
                value={lambdaMaxLeg}
                onChange={setLambdaMaxLeg}
                style={{ width: '100%' }}
              />
            </div>
            <div style={{ flex: '0 0 auto', minWidth: 200 }}>
              <Select
                label="γ (peso do tempo, km/h equiv)"
                options={GAMMA_OPTIONS}
                value={gamma}
                onChange={setGamma}
                style={{ width: '100%' }}
              />
            </div>
            <div style={{
              flex: 1, fontSize: 11, color: 'var(--muted)',
              fontFamily: 'JetBrains Mono', lineHeight: 1.6, minWidth: 240,
            }}>
              <b>γ = 0:</b> só conta distância — TSP clássico.<br />
              <b>γ &gt; 0:</b> cada hora de tour custa γ km na fitness. Útil
              em cold-chain (leite/carne) onde tempo importa mais que km.
              {parseFloat(gamma) > 0 && melhorTempo !== '—' && (
                <>
                  <br />
                  <span style={{ color: 'var(--primary-glow)' }}>
                    Tempo atual: <b>{melhorTempo}</b> · custo do tempo: <b>≈ {(parseFloat(gamma) * (parseFloat(melhorTempo) || 0)).toFixed(0)} km equiv</b>
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Linha 2: jornada (overtime) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <label style={{
              display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer',
              fontSize: 12, fontFamily: 'JetBrains Mono',
              color: jornadaAtiva ? 'var(--cyan)' : 'var(--muted)',
            }}>
              <input
                type="checkbox"
                checked={jornadaAtiva}
                onChange={e => setJornadaAtiva(e.target.checked)}
                style={{ accentColor: 'var(--cyan)' }}
              />
              limitar jornada motorista (10h ANTT)
            </label>
            {jornadaAtiva && (
              <div style={{ minWidth: 180 }}>
                <Select
                  label="μ (overtime, km/h² equiv)"
                  options={MU_OPTIONS}
                  value={muOvertime}
                  onChange={setMuOvertime}
                  style={{ width: '100%' }}
                />
              </div>
            )}
            <div style={{
              flex: 1, fontSize: 11, color: 'var(--muted)',
              fontFamily: 'JetBrains Mono', lineHeight: 1.6, minWidth: 240,
            }}>
              {jornadaAtiva ? (
                <>
                  Cada hora além de 10h custa μ · h² na fitness (quadrático). Usar pra
                  forçar tours que cabem num único turno do motorista.
                </>
              ) : (
                <>
                  Sem limite de jornada — tours podem chegar a 24h+ sem penalidade.
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
          title={(parseFloat(lambdaMaxLeg) > 0 || parseFloat(gamma) > 0 || (jornadaAtiva && parseFloat(muOvertime) > 0))
            ? 'Custo' : 'Distância'}
          value={(parseFloat(lambdaMaxLeg) > 0 || parseFloat(gamma) > 0 || (jornadaAtiva && parseFloat(muOvertime) > 0))
            ? melhorCusto : melhorDist}
          label={(parseFloat(lambdaMaxLeg) > 0 || parseFloat(gamma) > 0 || (jornadaAtiva && parseFloat(muOvertime) > 0))
            ? `fitness com penalidades (real: ${melhorDist})`
            : 'tour mais curto da geração'}
          color="cyan"
        />
        <MetricCard
          title="Tempo · Maior trecho"
          value={`${melhorTempo} · ${melhorMaxLeg}`}
          label="duração total · leg mais longo"
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
          histTours={histToursDone}
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
                setPlayingEvo(false);
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
            <button
              className="btn btn-ghost"
              style={{
                fontSize: 10, padding: '4px 10px',
                color: playingEvo ? 'var(--cyan)' : undefined,
              }}
              onClick={() => {
                setUserScrub(true);
                setPlayingEvo(p => !p);
              }}
              title="anima o melhor de cada geração (auto-scrub do slider)"
            >
              {playingEvo ? '⏸ evo' : '▶ evo'}
            </button>
            <div style={{ display: 'flex', gap: 2, fontSize: 10, fontFamily: 'JetBrains Mono' }}>
              {[5, 20, 60].map(s => (
                <button
                  key={s}
                  className="btn btn-ghost"
                  style={{
                    fontSize: 10, padding: '3px 6px',
                    color: s === evoSpeed ? 'var(--cyan)' : 'var(--muted)',
                    fontWeight: s === evoSpeed ? 700 : 400,
                  }}
                  onClick={() => setEvoSpeed(s)}
                  title={`${s} gerações por segundo`}
                >
                  {s}/s
                </button>
              ))}
            </div>
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
              onChange={(e) => {
                setPlayingEvo(false); // arrastar slider pausa evolução
                handleSlider(parseInt(e.target.value));
              }}
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

      {/* Cenário ativo — narrativa logística específica + sugestões de fitness */}
      {presetMeta && (
        <Card title={`Cenário: ${presetMeta.nome}`} style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 12,
              color: 'var(--cyan)', marginBottom: 8,
            }}>
              Origem (depot): <b>{presetMeta.origem}</b>
              {' '}&middot;{' '}
              {presetMeta.cidades.length} pontos no total
            </div>
            {presetMeta.narrativa.split('\n\n').map((paragrafo, i) => (
              <p key={i} style={{ marginBottom: 8 }}>{paragrafo}</p>
            ))}
            {presetMeta.lastVisit >= 0 && presetMeta.lastVisitNome && (
              <div style={{
                marginTop: 12, padding: '10px 12px',
                background: 'rgba(255, 170, 0, 0.08)',
                border: '1px solid rgba(255, 170, 0, 0.3)',
                borderRadius: 6,
                fontSize: 12, lineHeight: 1.6,
              }}>
                <b style={{ color: '#ffaa00' }}>⚠ Restrição lógica:</b>{' '}
                <b>{presetMeta.lastVisitNome}</b> deve ser visitada{' '}
                <i>por último</i> (imediatamente antes do retorno ao depot).
                <br />
                <span style={{ color: 'var(--muted)' }}>
                  Sem essa restrição, o TSP puro pode achar tours absurdos
                  tipo "depot → porto vazio → silos → depot cheio". A penalidade
                  no fitness empurra a cidade-última pra posição certa.
                </span>
              </div>
            )}
            <div style={{
              marginTop: 12, padding: '10px 12px',
              background: 'var(--surface-2)', borderRadius: 6,
              fontSize: 12, lineHeight: 1.6,
            }}>
              <b style={{ color: 'var(--pink)' }}>Sugestão de fitness pra esse cenário:</b>{' '}
              <code>λ = {presetMeta.lambdaSugerido}</code>
              {' '}&middot;{' '}
              modo <code>{presetMeta.modoSugerido}</code>
              <br />
              {presetMeta.fitnessNota}
              <br />
              <span style={{ color: 'var(--muted)', fontStyle: 'italic' }}>
                (já aplicados automaticamente — você pode ajustar acima)
              </span>
            </div>
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
