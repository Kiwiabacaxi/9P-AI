import { useState, useEffect, useRef, useMemo } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip, ReferenceLine,
} from 'recharts';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import TspMap from '../components/viz/TspMap';
import { MiniRoute, RingDiagram, GeneStrip, ChromosomeFollower, OperatorLab, islandColor } from '../components/viz/IslandViz';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  TspCidade, TspPreset, TspDistMode, TspRouteGeometry,
  TspSelecao, TspCrossover, TspMutacao,
  TspMultiConfig, TspMultiStep, TspMultiResult, TspMigracao,
} from '../api/types';

const PRESET_OPTIONS = [
  { value: 'triangulo50', label: 'Triângulo 50 cidades (complexo)' },
  { value: 'triangulo20', label: 'Triângulo 20 cidades (enunciado)' },
];
const LAMBDA_OPTIONS = [
  { value: '0',   label: 'λ = 0 (TSP puro)' },
  { value: '0.5', label: 'λ = 0.5' },
  { value: '1',   label: 'λ = 1' },
  { value: '1.5', label: 'λ = 1.5 (penaliza trecho longo)' },
];

const DIST_OPTIONS = [
  { value: 'haversine',  label: 'Haversine (linha reta)' },
  { value: 'osrm',       label: 'OSRM (estrada real)' },
  { value: 'euclidiana', label: 'Euclidiana (graus)' },
];
const ILHAS_OPTIONS = [
  { value: '2', label: '2 ilhas' },
  { value: '3', label: '3 ilhas' },
  { value: '4', label: '4 ilhas' },
  { value: '6', label: '6 ilhas' },
];
const TAMILHA_OPTIONS = [
  { value: '10', label: '10 / ilha' },
  { value: '20', label: '20 / ilha' },
  { value: '30', label: '30 / ilha' },
  { value: '50', label: '50 / ilha' },
];
const GERACOES_OPTIONS = [
  { value: '50',  label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '300', label: '300' },
];
const INTERVALO_OPTIONS = [
  { value: '5',  label: 'a cada 5 ger' },
  { value: '10', label: 'a cada 10 ger' },
  { value: '20', label: 'a cada 20 ger' },
];
const MIGRANTES_OPTIONS = [
  { value: '1', label: '1 migrante' },
  { value: '2', label: '2 migrantes' },
  { value: '3', label: '3 migrantes' },
];
const SELECAO_OPTIONS = [
  { value: 'torneio', label: 'Torneio' },
  { value: 'roleta', label: 'Roleta' },
];
const CRUZAMENTO_OPTIONS = [
  { value: 'ox',  label: 'OX (Order Crossover)' },
  { value: 'pmx', label: 'PMX (Partially Mapped)' },
];
const MUTACAO_OPTIONS = [
  { value: 'inversao', label: 'Inversão (2-opt)' },
  { value: 'swap',     label: 'Swap (troca)' },
];
const PC_OPTIONS = [
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

interface ChartRow {
  gen: number;
  global: number;
  div: number;
  refUnica?: number;
  refdiv?: number;
  [ilha: string]: number | undefined;
}

export default function TspMultiView() {
  const { show } = useToast();

  // Cidades / matriz (reusa o pipeline do TSP)
  const [preset, setPreset] = useState('triangulo50');
  const [cidades, setCidades] = useState<TspCidade[]>([]);
  const [distMode, setDistMode] = useState<TspDistMode>('haversine');
  const [matrizPronta, setMatrizPronta] = useState(false);

  // Config multi
  const [numIlhas, setNumIlhas] = useState('3');
  const [tamIlha, setTamIlha] = useState('30');
  const [maxGeracoes, setMaxGeracoes] = useState('200');
  const [intervalo, setIntervalo] = useState('10');
  const [numMigrantes, setNumMigrantes] = useState('1');
  const [comparar, setComparar] = useState(true);

  // Config GA
  const [selecao, setSelecao] = useState<TspSelecao>('torneio');
  const [cruzamento, setCruzamento] = useState<TspCrossover>('ox');
  const [mutacao, setMutacao] = useState<TspMutacao>('inversao');
  const [probCruz, setProbCruz] = useState('0.85');
  const [probMut, setProbMut] = useState('0.15');
  const [lambda, setLambda] = useState('0');

  // Estado de treino
  const [training, setTraining] = useState(false);
  const [step, setStep] = useState<TspMultiStep | null>(null);
  const [chartData, setChartData] = useState<ChartRow[]>([]);
  const [migracaoGens, setMigracaoGens] = useState<number[]>([]);
  const [globalTour, setGlobalTour] = useState<number[]>([]);
  const [result, setResult] = useState<TspMultiResult | null>(null);
  const [routeGeometry, setRouteGeometry] = useState<TspRouteGeometry | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);

  // Highlight de migração (segurado por ~1.2s pra ficar visível mesmo se o
  // streaming for rápido).
  const [migracaoFlash, setMigracaoFlash] = useState<{ gen: number; migracoes: TspMigracao[] } | null>(null);
  const flashTimer = useRef<number | null>(null);

  const closeSSE = useRef<(() => void) | null>(null);
  const unidade = distMode === 'euclidiana' ? 'graus' : 'km';
  const nIlhas = parseInt(numIlhas);

  async function carregarPreset(name: string, modo: TspDistMode) {
    setMatrizPronta(false);
    const p = await apiGet<TspPreset>(`/tsp/preset?name=${name}`);
    await apiPost('/tsp/cities', p.cidades);
    await apiPost('/tsp/distancias', { modo });
    setCidades(p.cidades);
    setMatrizPronta(true);
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await carregarPreset(preset, 'haversine');
        if (cancelled) return;
      } catch (e) {
        show('Erro ao carregar cenário: ' + (e instanceof Error ? e.message : String(e)));
      }
    })();
    return () => {
      cancelled = true;
      if (closeSSE.current) closeSSE.current();
      if (flashTimer.current) window.clearTimeout(flashTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handlePresetChange(novo: string) {
    setPreset(novo);
    limparEstado();
    try {
      await carregarPreset(novo, distMode);
      const n = novo === 'triangulo50' ? 50 : 20;
      show(`Cenário: ${n} cidades carregado`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  async function handleDistModeChange(novo: string) {
    const modo = novo as TspDistMode;
    setDistMode(modo);
    setMatrizPronta(false);
    try {
      await apiPost('/tsp/distancias', { modo });
      setMatrizPronta(true);
      show(modo === 'osrm' ? 'Matriz OSRM cacheada (estradas reais).' : `Matriz recalculada (${modo})`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  function limparEstado() {
    setStep(null);
    setChartData([]);
    setMigracaoGens([]);
    setGlobalTour([]);
    setResult(null);
    setMigracaoFlash(null);
    setRouteGeometry(null);
  }

  // Busca a geometria curvada (estradas reais via OSRM) da melhor rota global.
  // Independe do modo de distância — é só pra desenhar a rota no mapa seguindo
  // as estradas, em vez de linhas retas entre cidades.
  async function fetchRouteGeometry(tour: number[]) {
    if (tour.length < 3) return;
    setGeometryLoading(true);
    try {
      const geo = await apiPost<TspRouteGeometry>('/tsp/geometry', { tour });
      setRouteGeometry(geo);
      show(`Rota real: ${geo.distancia.toFixed(0)} km por estrada (${(geo.duracao / 3600).toFixed(1)} h dirigindo)`);
    } catch (e) {
      show('OSRM indisponível — mapa fica em linha reta. ' + (e instanceof Error ? e.message : ''));
    } finally {
      setGeometryLoading(false);
    }
  }

  function triggerFlash(gen: number, migracoes: TspMigracao[]) {
    setMigracaoFlash({ gen, migracoes });
    if (flashTimer.current) window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setMigracaoFlash(null), 1200);
  }

  async function handleTrain() {
    if (!matrizPronta) {
      show('Matriz não calculada — aguarde o cenário carregar');
      return;
    }
    setTraining(true);
    limparEstado();

    const cfg: TspMultiConfig = {
      numIlhas: nIlhas,
      tamIlha: parseInt(tamIlha),
      maxGeracoes: parseInt(maxGeracoes),
      intervaloMigracao: parseInt(intervalo),
      numMigrantes: parseInt(numMigrantes),
      topologia: 'anel',
      compararPopUnica: comparar,
      ga: {
        popSize: parseInt(tamIlha),
        maxGeracoes: parseInt(maxGeracoes),
        probCruzamento: parseFloat(probCruz),
        probMutacao: parseFloat(probMut),
        selecao,
        tamanhoTorneio: 4,
        cruzamento,
        mutacao,
        elitismo: 2,
        lambdaMaxLeg: parseFloat(lambda),
        lastVisit: -1,
        gamma: 0,
        jornadaMaxSec: 36000,
        muOvertime: 0,
      },
    };

    try {
      await apiPost('/tspmulti/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      return;
    }

    const localMigr: number[] = [];
    closeSSE.current = apiSSE('/tspmulti/train', {
      onMessage(data) {
        const s = data as TspMultiStep;
        setStep(s);
        setGlobalTour(s.melhorGlobalTour);

        setChartData(prev => {
          const row: ChartRow = { gen: s.geracao, global: s.melhorGlobalDist, div: s.diversidadeGlobal };
          s.ilhas.forEach(il => { row[`ilha${il.ilha}`] = il.melhorDist; });
          if (s.refUnicaDist) row.refUnica = s.refUnicaDist;
          if (s.refUnicaDiv !== undefined) row.refdiv = s.refUnicaDiv;
          return [...prev, row];
        });

        if (s.migrou) {
          localMigr.push(s.geracao);
          setMigracaoGens([...localMigr]);
          triggerFlash(s.geracao, s.migracoes ?? []);
        }
      },
      onDone(data) {
        const r = data as TspMultiResult;
        setResult(r);
        setGlobalTour(r.melhorGlobalTour);
        // reconstrói o chart a partir do resultado (autoritativo)
        const rows: ChartRow[] = r.histGlobal.map((g, i) => {
          const row: ChartRow = { gen: i + 1, global: g, div: r.histDiversidade[i] };
          r.histIlhas.forEach((h, isl) => { row[`ilha${isl}`] = h[i]; });
          if (r.histRefUnica) row.refUnica = r.histRefUnica[i];
          if (r.histRefUnicaDiv) row.refdiv = r.histRefUnicaDiv[i];
          return row;
        });
        setChartData(rows);
        setMigracaoGens(r.geracoesMigracao);
        setTraining(false);
        closeSSE.current = null;
        void fetchRouteGeometry(r.melhorGlobalTour);
        const cmp = r.melhorRefUnicaDist
          ? ` · pop única: ${r.melhorRefUnicaDist.toFixed(1)} ${unidade}`
          : '';
        show(`Melhor global (ilhas): ${r.melhorGlobalDist.toFixed(1)} ${unidade}${cmp}`);
      },
      onError() {
        setTraining(false);
        closeSSE.current = null;
        show('Erro no treino (stream)');
      },
    });
  }

  async function handleReset() {
    if (closeSSE.current) { closeSSE.current(); closeSSE.current = null; }
    try { await apiPost('/tspmulti/reset'); } catch { /* ignore */ }
    limparEstado();
    setTraining(false);
    show('Multi-ilhas resetado');
  }

  // melhor das ilhas pra tabela (do step atual ou do resultado)
  const ilhasAtuais = step?.ilhas ?? [];
  const flashSet = useMemo(() => {
    const m = new Map<number, { de: number; tour?: number[] }>();
    (migracaoFlash?.migracoes ?? []).forEach(mg => m.set(mg.para, { de: mg.de, tour: mg.migranteTour }));
    return m;
  }, [migracaoFlash]);

  const melhorGlobalDist = step?.melhorGlobalDist ?? result?.melhorGlobalDist ?? null;
  const ilhaVencedora = step?.ilhaVencedora ?? result?.ilhaVencedora ?? 0;
  const estagnacao = step?.geracoesSemMelhora ?? 0;
  const refDist = result?.melhorRefUnicaDist ?? (comparar ? step?.refUnicaDist : undefined);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">TSP <span>Multi-ilhas</span></div>
          <div className="page-sub">
            AG multipopulacional (modelo de ilhas) — Trabalho 12 · Aula 14 · {cidades.length} cidades do Triângulo Mineiro
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" onClick={handleReset} style={{ fontSize: 11, padding: '6px 12px' }}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training || !matrizPronta}>
            {training && <span className="spin" />}
            EVOLUIR ILHAS
          </button>
        </div>
      </div>

      {/* Config linha 1 — multipopulacional */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Nº de ilhas" options={ILHAS_OPTIONS} value={numIlhas} onChange={setNumIlhas} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Tamanho da ilha" options={TAMILHA_OPTIONS} value={tamIlha} onChange={setTamIlha} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Migração" options={INTERVALO_OPTIONS} value={intervalo} onChange={setIntervalo} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Migrantes por ilha" options={MIGRANTES_OPTIONS} value={numMigrantes} onChange={setNumMigrantes} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Gerações" options={GERACOES_OPTIONS} value={maxGeracoes} onChange={setMaxGeracoes} style={{ width: '100%' }} />
          <label style={{
            display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', marginTop: 14,
            fontSize: 12, fontFamily: 'JetBrains Mono', color: comparar ? 'var(--cyan)' : 'var(--muted)',
          }}>
            <input type="checkbox" checked={comparar} onChange={e => setComparar(e.target.checked)} style={{ accentColor: 'var(--cyan)' }} />
            comparar com população única
          </label>
        </Card>
      </div>

      {/* Config linha 2 — cenário + GA */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Cenário (nº de cidades)" options={PRESET_OPTIONS} value={preset} onChange={handlePresetChange} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Modo de distância" options={DIST_OPTIONS} value={distMode} onChange={handleDistModeChange} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Seleção" options={SELECAO_OPTIONS} value={selecao} onChange={(v) => setSelecao(v as TspSelecao)} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Mutação" options={MUTACAO_OPTIONS} value={mutacao} onChange={(v) => setMutacao(v as TspMutacao)} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Cruzamento" options={CRUZAMENTO_OPTIONS} value={cruzamento} onChange={(v) => setCruzamento(v as TspCrossover)} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <div className="imgreg-select-label">Pc · Pm <span style={{ color: 'var(--muted)', fontWeight: 400 }}>· λ</span></div>
            <div style={{ display: 'flex', gap: 6 }}>
              <Select options={PC_OPTIONS} value={probCruz} onChange={setProbCruz} style={{ flex: 1 }} />
              <Select options={PM_OPTIONS} value={probMut} onChange={setProbMut} style={{ flex: 1 }} />
              <Select options={LAMBDA_OPTIONS} value={lambda} onChange={setLambda} style={{ flex: 1 }} />
            </div>
          </div>
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard
          title="Geração"
          value={step ? step.geracao.toLocaleString() : '—'}
          label={`de ${parseInt(maxGeracoes).toLocaleString()}`}
          color="green"
          pulse={training}
        />
        <MetricCard
          title="Melhor global"
          value={melhorGlobalDist !== null ? `${melhorGlobalDist.toFixed(1)} ${unidade}` : '—'}
          label={refDist ? `pop única: ${refDist.toFixed(1)} ${unidade}` : 'menor tour entre todas as ilhas'}
          color="cyan"
        />
        <MetricCard
          title="Estagnação"
          value={`${estagnacao} ger`}
          label="gerações sem melhorar o global"
          color={estagnacao > parseInt(intervalo) ? 'pink' : undefined}
        />
      </div>

      {/* Painel "melhor de todas" + dança de cadeiras */}
      <div className="grid-2" style={{ marginBottom: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Card title="Dança de cadeiras (migração em anel)">
          <div style={{ padding: 8 }}>
            <RingDiagram
              numIlhas={nIlhas}
              ilhaVencedora={ilhaVencedora}
              migracoes={migracaoFlash?.migracoes}
              size={240}
            />
            <div style={{ textAlign: 'center', fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginTop: 4 }}>
              {migracaoFlash
                ? <span style={{ color: 'var(--cyan)' }}>⇄ migração na geração {migracaoFlash.gen} — melhores viajam ilha → ilha</span>
                : <>cada ilha manda seu melhor pra próxima a cada <b>{intervalo.replace(/\D/g, '')}</b> gerações</>}
            </div>
          </div>
        </Card>

        <Card title={routeGeometry
          ? `Melhor rota global (estradas reais — ${routeGeometry.distancia.toFixed(0)} km / ${(routeGeometry.duracao / 3600).toFixed(1)} h)`
          : 'Melhor rota global no mapa'}>
          {geometryLoading && (
            <div style={{
              padding: '6px 12px', marginBottom: 8, background: 'var(--surface-2)', borderRadius: 6,
              fontSize: 11, color: 'var(--cyan)', fontFamily: 'JetBrains Mono',
            }}>
              <span className="spin" /> consultando OSRM (estradas reais)…
            </div>
          )}
          <TspMap
            cidades={cidades}
            tour={globalTour}
            globalTour={globalTour}
            routeGeometry={routeGeometry ?? undefined}
            height={300}
          />
        </Card>
      </div>

      {/* Small multiples — uma rota por ilha */}
      <Card title="Ilhas evoluindo em paralelo (cada uma se especializa numa rota)" style={{ marginBottom: 16 }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: `repeat(auto-fill, minmax(170px, 1fr))`,
          gap: 12, padding: 12,
        }}>
          {Array.from({ length: nIlhas }).map((_, i) => {
            const il = ilhasAtuais[i];
            const cor = islandColor(i);
            const flash = flashSet.get(i);
            const venceu = i === ilhaVencedora;
            return (
              <div key={i} style={{
                border: `1px solid ${flash ? islandColor(flash.de) : (venceu ? cor : '#222')}`,
                borderRadius: 6, padding: 8,
                boxShadow: flash ? `0 0 12px ${islandColor(flash.de)}` : (venceu ? `0 0 8px ${cor}66` : 'none'),
                transition: 'box-shadow 0.3s, border-color 0.3s',
                background: 'var(--surface-2)',
              }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  fontSize: 11, fontFamily: 'JetBrains Mono', marginBottom: 6,
                }}>
                  <span style={{ color: cor, fontWeight: 700 }}>
                    Ilha {i + 1}{venceu ? ' 👑' : ''}
                  </span>
                  {il && <span style={{ color: 'var(--muted)' }}>{il.melhorDist.toFixed(0)} {unidade}</span>}
                </div>
                <MiniRoute
                  cidades={cidades}
                  tour={il?.melhorTour ?? []}
                  color={cor}
                  size={154}
                  highlightTour={flash?.tour}
                  highlightColor={flash ? islandColor(flash.de) : undefined}
                />
                {il && (
                  <div style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginTop: 6 }}>
                    diversidade: <span style={{ color: cor }}>{il.diversidade}</span>/{tamIlha}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {/* Convergência multi-linha + comparativo + marcadores de migração */}
      {chartData.length > 0 && (
        <Card title={`Convergência — distância do melhor tour (${unidade})`} style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="gen" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  labelFormatter={(v) => `geração ${v}`}
                  formatter={(v, name) => [`${Number(v).toFixed(1)} ${unidade}`, String(name)]}
                />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                {/* marcadores verticais de migração */}
                {migracaoGens.map((g, i) => (
                  <ReferenceLine
                    key={`mig-${g}`}
                    x={g}
                    stroke="#ffd400"
                    strokeDasharray="2 3"
                    strokeOpacity={0.5}
                    label={i === 0 ? { value: 'migração', fill: '#ffd400', fontSize: 9, position: 'top' } : undefined}
                  />
                ))}
                {/* linha por ilha */}
                {Array.from({ length: nIlhas }).map((_, i) => (
                  <Line
                    key={`l-ilha-${i}`}
                    name={`ilha ${i + 1}`}
                    type="monotone"
                    dataKey={`ilha${i}`}
                    stroke={islandColor(i)}
                    strokeWidth={1}
                    strokeOpacity={0.55}
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                ))}
                {/* pop única (comparativo) */}
                {comparar && (
                  <Line
                    name="pop única"
                    type="monotone"
                    dataKey="refUnica"
                    stroke="#888"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                )}
                {/* melhor global das ilhas */}
                <Line
                  name="melhor global (ilhas)"
                  type="monotone"
                  dataKey="global"
                  stroke="#ffff00"
                  strokeWidth={2.6}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
              </ComposedChart>
            </ResponsiveContainer>
            {comparar && refDist && melhorGlobalDist !== null && (() => {
              const diff = ((refDist - melhorGlobalDist) / refDist) * 100;
              const cor = diff > 0.3 ? 'var(--cyan)' : diff < -0.3 ? '#ff8a3d' : 'var(--muted)';
              return (
                <div style={{ textAlign: 'center', fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginTop: 4, lineHeight: 1.6 }}>
                  multi-ilhas: <b style={{ color: 'var(--cyan)' }}>{melhorGlobalDist.toFixed(0)}</b> {unidade}
                  {'  ·  '}população única (mesmo total de indivíduos): <b>{refDist.toFixed(0)}</b> {unidade}
                  {'  →  '}
                  <b style={{ color: cor }}>
                    {Math.abs(diff) < 0.3 ? 'praticamente empate' : diff > 0 ? `multi ${diff.toFixed(1)}% melhor` : `multi ${(-diff).toFixed(1)}% pior`}
                  </b>
                  <br />
                  <span style={{ fontStyle: 'italic' }}>
                    {cidades.length <= 25
                      ? <>Com poucas cidades o problema é fácil e ambas costumam achar (quase) o mesmo ótimo. Troque pro cenário de 50 cidades pra ver a vantagem do modelo de ilhas aparecer.</>
                      : <>Com {cidades.length} cidades a paisagem tem muitos mínimos locais: a população única tende a convergir cedo e travar, enquanto as ilhas mantêm diversidade e escapam via migração.</>}
                  </span>
                </div>
              );
            })()}
          </div>
        </Card>
      )}

      {/* Medidor de diversidade — ilhas vs pop única */}
      {chartData.length > 0 && (
        <Card title="Diversidade (tours únicos) — ilhas vs população única" style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <ResponsiveContainer width="100%" height={180}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="gen" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  labelFormatter={(v) => `geração ${v}`}
                  formatter={(v, name) => [`${Number(v)} tours únicos`, String(name)]}
                />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                {migracaoGens.map((g) => (
                  <ReferenceLine key={`md-${g}`} x={g} stroke="#ffd400" strokeDasharray="2 3" strokeOpacity={0.5} />
                ))}
                <Area name="ilhas (soma)" type="monotone" dataKey="div" stroke="#7cf67c" fill="#7cf67c" fillOpacity={0.15} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                {comparar && (
                  <Line name="população única" type="monotone" dataKey="refdiv" stroke="#888" strokeWidth={1.5} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls />
                )}
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)', marginTop: 2 }}>
              repare nos <span style={{ color: '#ffd400' }}>saltos</span> de diversidade das ilhas logo após cada linha de migração — é a reinjeção de variedade que o modelo de ilhas promove
            </div>
          </div>
        </Card>
      )}

      {/* Tabela por ilha */}
      {ilhasAtuais.length > 0 && (
        <Card title="Placar das ilhas" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'JetBrains Mono' }}>
              <thead>
                <tr style={{ color: 'var(--muted)', textAlign: 'left' }}>
                  <th style={{ padding: '4px 8px' }}>Ilha</th>
                  <th style={{ padding: '4px 8px' }}>Melhor dist.</th>
                  <th style={{ padding: '4px 8px' }}>Diversidade</th>
                  <th style={{ padding: '4px 8px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {ilhasAtuais.map((il) => (
                  <tr key={il.ilha} style={{ borderTop: '1px solid #222' }}>
                    <td style={{ padding: '4px 8px', color: islandColor(il.ilha), fontWeight: 700 }}>
                      ● Ilha {il.ilha + 1}
                    </td>
                    <td style={{ padding: '4px 8px', color: 'var(--on-surface)' }}>{il.melhorDist.toFixed(1)} {unidade}</td>
                    <td style={{ padding: '4px 8px', color: 'var(--muted)' }}>{il.diversidade}/{tamIlha}</td>
                    <td style={{ padding: '4px 8px' }}>
                      {il.ilha === ilhaVencedora
                        ? <span style={{ color: islandColor(il.ilha) }}>👑 detém o melhor global</span>
                        : <span style={{ color: 'var(--muted)' }}>—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Desenho do cromossomo — animação seguindo o melhor + comparação entre ilhas */}
      {globalTour.length > 0 && (
        <Card title="Cromossomo — como a rota é codificada (animação seguindo o melhor)" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12 }}>
            <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 12 }}>
              Cada indivíduo é um <b>cromossomo = permutação</b> das {cidades.length} cidades — a ordem em que o
              caminhão visita. Diferente das aulas 10–11 (cromossomo binário), aqui cada <b>gene é uma cidade</b>
              {' '}(o número é o id; o depot Uberlândia = <span style={{ color: '#ff00aa' }}>0</span>, em rosa). O
              destaque amarelo percorre os genes na ordem do tour — é o caminhão "lendo" o cromossomo.
            </div>
            <div style={{ marginBottom: 6, fontSize: 11, fontFamily: 'JetBrains Mono', color: '#ffff00' }}>
              ★ melhor cromossomo global
            </div>
            <ChromosomeFollower cidades={cidades} tour={globalTour} color="#ffff00" />

            <div style={{ marginTop: 18, fontSize: 13, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10 }}>
              E abaixo, o melhor cromossomo de <b>cada ilha</b> comparado com o global: genes <b style={{ color: 'var(--cyan)' }}>iguais ao global desbotam</b>;
              só os <b style={{ color: 'var(--cyan)' }}>diferentes ficam fortes</b>. Assim você vê em 1 olhada quanto e onde cada ilha divergiu —
              perto das migrações elas vão "alinhando" e diminuindo o número de diferenças.
            </div>
            {ilhasAtuais.map(il => {
              const cor = islandColor(il.ilha);
              const ndif = il.melhorTour.reduce((acc, g, i) => acc + (g !== globalTour[i] ? 1 : 0), 0);
              const igual = ndif === 0;
              return (
                <div key={il.ilha} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: cor, marginBottom: 4 }}>
                    ● Ilha {il.ilha + 1} — {il.melhorDist.toFixed(0)} {unidade}
                    <span style={{ color: 'var(--muted)', marginLeft: 8 }}>
                      {igual
                        ? '(idêntica ao global ✓)'
                        : `(${ndif}/${cidades.length} genes diferentes do global)`}
                    </span>
                  </div>
                  <GeneStrip
                    cidades={cidades}
                    tour={il.melhorTour}
                    color={cor}
                    compareTo={globalTour}
                    dimMatching
                  />
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Laboratório de operadores — seleção → cruzamento → mutação passo a passo */}
      {ilhasAtuais.length >= 2 && (
        <Card title="Laboratório de operadores — veja seleção, cruzamento e mutação passo a passo" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12 }}>
            <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 12 }}>
              Pega dois pais reais (os melhores de duas ilhas) e anima como nasce um filho, etapa por etapa:
              <b style={{ color: 'var(--cyan)' }}> seleção</b> →
              cruzamento <b>{cruzamento.toUpperCase()}</b> → mutação <b>{mutacao}</b>.
              No filho, <b>cada gene fica colorido pela sua origem</b> — assim você vê na hora qual parte veio do
              Pai 1, qual veio do Pai 2 e quais foram mutadas. Use ◀ ▶ pra navegar as etapas, ▶ pra tocar,
              e "gerar novo filho" pra sortear novos cortes. Troque os operadores nos controles do topo (OX↔PMX,
              swap↔inversão) pra ver o padrão mudar.
            </div>
            <OperatorLab
              cidades={cidades}
              pais={[...ilhasAtuais]
                .sort((a, b) => a.melhorDist - b.melhorDist)
                .map(il => ({ tour: il.melhorTour, label: `Ilha ${il.ilha + 1} (${il.melhorDist.toFixed(0)} ${unidade})`, color: islandColor(il.ilha) }))}
              cruzamento={cruzamento}
              mutacao={mutacao}
              probMut={parseFloat(probMut)}
            />
          </div>
        </Card>
      )}

      {/* Educacional */}
      <Card title="Como funciona o AG multi-populacional (modelo de ilhas)">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Ilhas em paralelo:</b> em vez de uma única população, rodamos várias subpopulações
          ("ilhas") simultaneamente — cada uma em sua própria goroutine. Como comunidades em
          continentes separados, cada ilha evolui por um caminho diferente e se especializa
          numa rota.
          <br /><br />
          <b>Migração (dança de cadeiras):</b> a cada <b>{intervalo.replace(/\D/g, '')}</b> gerações,
          o(s) <b>{numMigrantes}</b> melhor(es) indivíduo(s) de cada ilha são copiados para a ilha
          vizinha (topologia em anel), substituindo os piores de lá. Isso reinjeta diversidade e
          espalha bons "genes" entre as ilhas.
          <br /><br />
          <b>Por que ajuda:</b> uma população única tende a convergir cedo e travar num mínimo
          local (todo mundo parecido). Com ilhas + migração, quando uma ilha empaca, recebe sangue
          novo de outra que explorou região diferente do espaço de busca. O comparativo com a
          <b> população única</b> (mesmo total de indivíduos) deixa esse ganho explícito na curva.
          <br /><br />
          <b>Determinismo:</b> cada ilha tem seu próprio gerador aleatório semeado a partir da seed,
          então o resultado é reprodutível independente de como as goroutines são escalonadas.
        </div>
      </Card>
    </div>
  );
}
