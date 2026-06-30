import { useState, useRef, useEffect, useMemo } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip,
} from 'recharts';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import { useToast } from '../components/shared/Toast';
import TspMap, { TspEvoChart } from '../components/viz/TspMap';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  TspRankConfig, TspRankStep, TspRankResult, TspRankMapa,
  TspRankSelecao, TspRankCruzamento, TspRankMutacao,
} from '../api/types';

// =============================================================================
// Trabalho 14 — AG com seleção por RANKING aplicado ao TSP (Aulas 13 + 16).
// =============================================================================

const POP_OPTIONS = [20, 40, 60, 80, 120, 200].map(n => ({ value: String(n), label: String(n) }));
const GER_OPTIONS = [50, 100, 250, 500].map(n => ({ value: String(n), label: String(n) }));
const PC_OPTIONS = ['0.6', '0.75', '0.85', '0.9'].map(v => ({ value: v, label: `Pc ${v}` }));
const PM_OPTIONS = ['0.05', '0.1', '0.2', '0.3'].map(v => ({ value: v, label: `Pm ${v}` }));
const SELECAO_OPTIONS = [
  { value: 'rankingLinear', label: 'Ranking linear' },
  { value: 'rankingExp', label: 'Ranking exponencial' },
  { value: 'torneio', label: 'Torneio (clássico)' },
  { value: 'roleta', label: 'Roleta (clássico)' },
];
const TORNEIO_OPTIONS = [2, 4, 6].map(k => ({ value: String(k), label: `k = ${k}` }));
const CRUZ_OPTIONS = [
  { value: 'ox', label: 'OX (Order Crossover)' },
  { value: 'pmx', label: 'PMX (Partially Mapped)' },
];
const MUT_OPTIONS = [
  { value: 'swap', label: 'Swap (troca 2 cidades)' },
  { value: 'inversao', label: 'Inversão (segmento)' },
];
const ELITE_OPTIONS = [0, 1, 2, 4].map(p => ({ value: String(p), label: p === 0 ? 'sem elite' : `p = ${p}` }));

// Fórmulas de ranking (espelham o backend Go) — pra o "Laboratório do Ranking".
function probsRankingLinear(n: number, etaMax: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [1];
  const etaMin = 2 - etaMax;
  const out: number[] = [];
  for (let i = 1; i <= n; i++) {
    out.push((1 / n) * (etaMax - (etaMax - etaMin) * (i - 1) / (n - 1)));
  }
  return out;
}
function probsRankingExp(n: number, c: number): number[] {
  if (n <= 0) return [];
  if (n === 1) return [1];
  if (c <= 1) c = 1 + 1e-6;
  const pesos: number[] = [];
  let soma = 0;
  for (let i = 1; i <= n; i++) {
    const w = Math.pow(c, n - i);
    pesos.push(w);
    soma += w;
  }
  return pesos.map(w => w / soma);
}

function abreviar(nome: string): string {
  const mapa: Record<string, string> = {
    'Uberaba': 'Ubra', 'Uberlândia': 'Ubln', 'Araguari': 'Argr', 'Ituiutaba': 'Itui',
    'Patos de Minas': 'Patos', 'Frutal': 'Frut', 'Araxá': 'Arax', 'Monte Carmelo': 'MtCar',
    'Tupaciguara': 'Tupa', 'Campina Verde': 'CpVrd',
  };
  return mapa[nome] ?? nome.slice(0, 4);
}

export default function TspRankingView() {
  const { show } = useToast();

  // Config
  const [popSize, setPopSize] = useState('80');
  const [maxGer, setMaxGer] = useState('250');
  const [pc, setPc] = useState('0.9');
  const [pm, setPm] = useState('0.2');
  const [selecao, setSelecao] = useState<TspRankSelecao>('rankingLinear');
  const [tamTorneio, setTamTorneio] = useState('4');
  const [etaMax, setEtaMax] = useState(1.5);
  const [cExp, setCExp] = useState(1.07);
  const [cruzamento, setCruzamento] = useState<TspRankCruzamento>('ox');
  const [mutacao, setMutacao] = useState<TspRankMutacao>('swap');
  const [elitismo, setElitismo] = useState('2');

  // Cenário + treino
  const [mapa, setMapa] = useState<TspRankMapa | null>(null);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<TspRankResult | null>(null);
  const [frames, setFrames] = useState<TspRankStep[]>([]);
  const framesRef = useRef<TspRankStep[]>([]);
  const closeSSE = useRef<(() => void) | null>(null);

  // Laboratório do Ranking — nasce com os parâmetros do exemplo da Aula 16
  // (N=5, η_max=1.5, c=2) pra reproduzir os slides exatamente; depois de evoluir,
  // os sliders são sincronizados com a pressão realmente usada na execução.
  const [labEtaMax, setLabEtaMax] = useState(1.5);
  const [labCExp, setLabCExp] = useState(2.0);
  const [labN, setLabN] = useState(5);
  const [labMostrarPop, setLabMostrarPop] = useState(true);

  useEffect(() => {
    apiGet<TspRankMapa>('/tspranking/cidades').then(setMapa).catch(() => show('Erro ao carregar cidades'));
    return () => { if (closeSSE.current) closeSSE.current(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleTrain() {
    setTraining(true);
    framesRef.current = [];
    setFrames([]);
    setResult(null);
    // sincroniza os sliders do laboratório com a pressão configurada (pra inspecionar o que rodou)
    setLabEtaMax(etaMax);
    setLabCExp(cExp);

    const cfg: TspRankConfig = {
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGer),
      probCruzamento: parseFloat(pc),
      probMutacao: parseFloat(pm),
      selecao,
      tamanhoTorneio: parseInt(tamTorneio),
      etaMax,
      cExp,
      cruzamento,
      mutacao,
      elitismo: parseInt(elitismo),
    };
    try {
      await apiPost('/tspranking/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      return;
    }
    closeSSE.current = apiSSE('/tspranking/train', {
      onMessage(data) {
        const s = data as TspRankStep;
        framesRef.current.push(s);
        setFrames(framesRef.current.slice());
      },
      onDone(data) {
        const r = data as TspRankResult;
        setResult(r);
        setTraining(false);
        closeSSE.current = null;
        show(`Melhor rota: ${r.melhorDist.toFixed(0)} km em ${r.geracoes} gerações`);
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
    try { await apiPost('/tspranking/reset'); } catch { /* ignore */ }
    framesRef.current = [];
    setFrames([]);
    setResult(null);
    setTraining(false);
    show('TSP Ranking resetado');
  }

  // Derivados
  const cidades = mapa?.cidades ?? [];
  const lastFrame = frames.length ? frames[frames.length - 1] : null;
  const bestTour = result?.melhorTour ?? lastFrame?.melhorGlobalTour ?? [];
  const histTours = useMemo(() => frames.map(f => f.melhorGlobalTour), [frames]);
  const histMelhor = useMemo(() => frames.map(f => f.melhorDist), [frames]);
  const histMedia = useMemo(() => frames.map(f => f.mediaDist), [frames]);
  const popDistAtual = lastFrame?.popDist ?? [];
  const melhorDist = result?.melhorDist ?? lastFrame?.melhorGlobalDist ?? null;
  const geracaoAtual = lastFrame?.geracao ?? 0;
  const distInicial = frames.length ? frames[0].melhorDist : null;
  const melhoria = distInicial && melhorDist ? ((distInicial - melhorDist) / distInicial) * 100 : null;
  const diversidade = lastFrame?.diversidade ?? null;

  const isTorneio = selecao === 'torneio';
  const isLinear = selecao === 'rankingLinear';
  const isExp = selecao === 'rankingExp';

  // Dados do Laboratório do Ranking
  const lab = useMemo(() => {
    const n = Math.max(2, labN);
    const pl = probsRankingLinear(n, labEtaMax);
    const pe = probsRankingExp(n, labCExp);
    const alinhaPop = labMostrarPop && popDistAtual.length === n;
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push({
        rank: i + 1,
        linear: pl[i] * 100,
        exp: pe[i] * 100,
        popDist: alinhaPop ? popDistAtual[i] : null,
      });
    }
    return { rows, alinhaPop };
  }, [labN, labEtaMax, labCExp, labMostrarPop, popDistAtual]);

  const labTabela = lab.rows.slice(0, Math.min(lab.rows.length, 5));

  // Nome da cidade pela ordem no melhor tour (pra listar a rota)
  const rotaNomes = bestTour.map(id => cidades.find(c => c.id === id)?.nome ?? `#${id}`);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">AG <span>TSP Ranking</span></div>
          <div className="page-sub">
            AG com seleção por <b>Ranking</b> — Trabalho 14 · Aulas 13 + 16 · roteia 10 cidades do
            Triângulo Mineiro partindo de Uberaba, minimizando a distância
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" onClick={handleReset} style={{ fontSize: 11, padding: '6px 12px' }}>RESETAR</button>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training && <span className="spin" />}
            EVOLUIR
          </button>
        </div>
      </div>

      {/* Config linha 1 */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="População (N)" options={POP_OPTIONS} value={popSize} onChange={setPopSize} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            <Select label="Gerações" options={GER_OPTIONS} value={maxGer} onChange={setMaxGer} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">
            Pc · Pm <span style={{ color: 'var(--muted)', fontWeight: 400 }}>(cruzamento · mutação)</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={PC_OPTIONS} value={pc} onChange={setPc} style={{ flex: 1 }} />
            <Select options={PM_OPTIONS} value={pm} onChange={setPm} style={{ flex: 1 }} />
          </div>
          <div style={{ marginTop: 10 }}>
            <Select label="Elitismo" options={ELITE_OPTIONS} value={elitismo} onChange={setElitismo} style={{ width: '100%' }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Seleção" options={SELECAO_OPTIONS} value={selecao} onChange={(v) => setSelecao(v as TspRankSelecao)} style={{ width: '100%' }} />
          <div style={{ marginTop: 10 }}>
            {isLinear && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
                η_max
                <input type="range" min={1} max={2} step={0.05} value={etaMax} onChange={e => setEtaMax(parseFloat(e.target.value))} style={{ flex: 1 }} />
                <span style={{ color: 'var(--cyan)', minWidth: 34, textAlign: 'right' }}>{etaMax.toFixed(2)}</span>
              </label>
            )}
            {isExp && (
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
                c
                <input type="range" min={1.01} max={2} step={0.01} value={cExp} onChange={e => setCExp(parseFloat(e.target.value))} style={{ flex: 1 }} />
                <span style={{ color: 'var(--cyan)', minWidth: 34, textAlign: 'right' }}>{cExp.toFixed(2)}</span>
              </label>
            )}
            {isTorneio && (
              <Select label="Tamanho do torneio" options={TORNEIO_OPTIONS} value={tamTorneio} onChange={setTamTorneio} style={{ width: '100%' }} />
            )}
            {selecao === 'roleta' && (
              <div style={{ fontSize: 11, color: '#777', fontFamily: 'JetBrains Mono' }}>
                Roleta proporcional clássica — privilegia demais quem tem fitness alto (a "armadilha" da Aula 16).
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Config linha 2 */}
      <div className="grid-2" style={{ marginBottom: 16 }}>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Cruzamento (permutação)" options={CRUZ_OPTIONS} value={cruzamento} onChange={(v) => setCruzamento(v as TspRankCruzamento)} style={{ width: '100%' }} />
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <Select label="Mutação (permutação)" options={MUT_OPTIONS} value={mutacao} onChange={(v) => setMutacao(v as TspRankMutacao)} style={{ width: '100%' }} />
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard title="Geração" value={geracaoAtual ? geracaoAtual.toLocaleString() : '—'} label={`de ${parseInt(maxGer).toLocaleString()}`} color="green" pulse={training} />
        <MetricCard title="Melhor rota" value={melhorDist !== null ? `${melhorDist.toFixed(0)} km` : '—'} label="distância total do ciclo" color="cyan" />
        <MetricCard title="Melhoria" value={melhoria !== null ? `${melhoria.toFixed(1)}%` : '—'} label={diversidade !== null ? `diversidade: ${diversidade}` : 'vs. melhor inicial'} />
      </div>

      {/* Mapa */}
      <Card title={`Mapa — melhor rota${training ? ' (ao vivo)' : ''} · 10 cidades do Triângulo Mineiro`} style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: 'JetBrains Mono',
          }}>
            Partida em <b style={{ color: '#ff00aa' }}>Uberaba</b> (marcador rosa). O número em cada cidade é a
            <b> ordem de visita</b> na melhor rota. Use <b>▶ play</b> pra o caminhão percorrer o tour; marque
            <b> "todas gerações"</b> pra ver o AG convergindo geração a geração.
          </div>
          {cidades.length > 0 ? (
            <TspMap cidades={cidades} tour={bestTour} globalTour={bestTour} histTours={histTours} height={480} />
          ) : (
            <div style={{ height: 480, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontFamily: 'JetBrains Mono' }}>
              carregando cidades…
            </div>
          )}
          {rotaNomes.length > 0 && (
            <div style={{ marginTop: 10, fontSize: 12, color: 'var(--muted)', fontFamily: 'JetBrains Mono', lineHeight: 1.7 }}>
              <b style={{ color: 'var(--cyan)' }}>Rota:</b> {rotaNomes.join(' → ')} → {rotaNomes[0]}
            </div>
          )}
        </div>
      </Card>

      {/* Convergência */}
      {frames.length > 0 && (
        <Card title="Convergência — distância por geração (km)" style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <TspEvoChart histMelhor={histMelhor} histMedia={histMedia} unidade="km" height={240} />
          </div>
        </Card>
      )}

      {/* === Laboratório do Ranking (o destaque do Trabalho 14) === */}
      <Card title="🏆 Laboratório do Ranking (Aula 16) — probabilidade por POSIÇÃO" style={{ marginBottom: 16 }}>
        <div style={{ padding: 8 }}>
          <div style={{
            display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
            marginBottom: 10, padding: '10px 12px', background: 'var(--surface-2)', borderRadius: 6,
          }}>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#888', marginRight: 4 }}>N:</span>
            {[5, 10, 20].map(n => (
              <button key={n} className="btn" onClick={() => setLabN(n)} aria-pressed={labN === n}
                style={{ fontSize: 11, padding: '5px 10px', borderColor: labN === n ? 'var(--cyan)' : '#333', opacity: labN === n ? 1 : 0.5 }}>
                {n}{n === 5 ? ' (slide)' : ''}
              </button>
            ))}
            {popDistAtual.length > 0 && (
              <button className="btn" onClick={() => setLabN(popDistAtual.length)} aria-pressed={labN === popDistAtual.length}
                style={{ fontSize: 11, padding: '5px 10px', borderColor: labN === popDistAtual.length ? 'var(--cyan)' : '#333', opacity: labN === popDistAtual.length ? 1 : 0.5 }}>
                população ({popDistAtual.length})
              </button>
            )}
            <div style={{ width: 1, height: 22, background: '#333', margin: '0 4px' }} />
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
              η_max (linear)
              <input type="range" min={1} max={2} step={0.05} value={labEtaMax} onChange={e => setLabEtaMax(parseFloat(e.target.value))} style={{ width: 110 }} />
              <span style={{ color: '#ff00aa' }}>{labEtaMax.toFixed(2)}</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
              c (exponencial)
              <input type="range" min={1.01} max={2} step={0.01} value={labCExp} onChange={e => setLabCExp(parseFloat(e.target.value))} style={{ width: 110 }} />
              <span style={{ color: '#00e5ff' }}>{labCExp.toFixed(2)}</span>
            </label>
            {popDistAtual.length > 0 && (
              <button className="btn" onClick={() => setLabMostrarPop(v => !v)} aria-pressed={labMostrarPop}
                style={{ fontSize: 11, padding: '5px 10px', opacity: labMostrarPop ? 1 : 0.4 }}>
                sobrepor distância real
              </button>
            )}
          </div>

          <div style={{
            fontSize: 12, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10,
            padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6, fontFamily: 'JetBrains Mono',
          }}>
            A seleção por ranking <b>ignora o valor absoluto</b> da distância e usa só a <b>posição</b> (rank 1 = melhor).
            A <b style={{ color: '#ff00aa' }}>linear (Baker)</b> decai em reta; a <b style={{ color: '#00e5ff' }}>exponencial</b> concentra
            mais nos primeiros.
            {lab.alinhaPop
              ? <> A linha cinza é a <b>distância real</b> de cada rank na população atual — note como ela é íngreme,
                  mas o ranking distribui as chances de forma suave.</>
              : <> Escolha N = "população" depois de evoluir pra sobrepor a distância real por rank.</>}
          </div>

          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={lab.rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="rank" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false}
                  label={{ value: 'posição no ranking (1 = melhor)', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 11 }} />
                <YAxis yAxisId="p" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false}
                  label={{ value: 'P seleção (%)', angle: -90, position: 'insideLeft', fill: '#666', fontSize: 11 }} />
                {lab.alinhaPop && (
                  <YAxis yAxisId="km" orientation="right" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false}
                    label={{ value: 'distância (km)', angle: 90, position: 'insideRight', fill: '#666', fontSize: 11 }} />
                )}
                <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  labelFormatter={(v) => `rank ${v}`}
                  formatter={(val, name) => {
                    if (name === 'distância real') return [`${Number(val).toFixed(0)} km`, name];
                    return [`${Number(val).toFixed(2)} %`, name];
                  }} />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Line yAxisId="p" name="linear (Baker)" type="monotone" dataKey="linear" stroke="#ff00aa" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                <Line yAxisId="p" name="exponencial" type="monotone" dataKey="exp" stroke="#00e5ff" strokeWidth={2.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
                {lab.alinhaPop && (
                  <Line yAxisId="km" name="distância real" type="monotone" dataKey="popDist" stroke="#888" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Tabela reproduzindo os slides "Resultado Final / Comparação" */}
          <div style={{ marginTop: 12, overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontFamily: 'JetBrains Mono', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#888' }}>
                  <th style={{ padding: '4px 14px', textAlign: 'left', borderBottom: '1px solid #333' }}>Rank</th>
                  <th style={{ padding: '4px 14px', textAlign: 'right', borderBottom: '1px solid #333', color: '#ff00aa' }}>P linear</th>
                  <th style={{ padding: '4px 14px', textAlign: 'right', borderBottom: '1px solid #333', color: '#00e5ff' }}>P exponencial</th>
                </tr>
              </thead>
              <tbody>
                {labTabela.map(r => (
                  <tr key={r.rank}>
                    <td style={{ padding: '4px 14px', color: 'var(--muted)' }}>{r.rank}º</td>
                    <td style={{ padding: '4px 14px', textAlign: 'right', color: '#ff7ac8' }}>{r.linear.toFixed(1)}%</td>
                    <td style={{ padding: '4px 14px', textAlign: 'right', color: '#7fe8ff' }}>{r.exp.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {labN === 5 && (
              <div style={{ fontSize: 11, color: '#777', marginTop: 6, fontFamily: 'JetBrains Mono' }}>
                Com N=5, η_max=1.5 → 30/25/20/15/10% e c=2 → 51.6/25.8/12.9/6.4/3.2% — exatamente os slides da Aula 16.
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Matriz de distâncias (Aula 13) */}
      {mapa && (
        <Card title="Matriz de distâncias (km) — tabela da Aula 13" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, overflowX: 'auto' }}>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10, fontFamily: 'JetBrains Mono', lineHeight: 1.6 }}>
              Células <b style={{ color: 'var(--cyan)' }}>em ciano</b> vêm da tabela do slide; as
              <b style={{ color: '#c98aff' }}> em roxo (itálico)</b> eram "—" e foram preenchidas por
              Haversine × {mapa.fator.toFixed(2)} (fator de calibração estrada/reta).
            </div>
            <table style={{ borderCollapse: 'collapse', fontFamily: 'JetBrains Mono', fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px', borderBottom: '1px solid #333' }}></th>
                  {mapa.cidades.map(c => (
                    <th key={c.id} style={{ padding: '4px 8px', borderBottom: '1px solid #333', color: '#888', textAlign: 'right' }}>{abreviar(c.nome)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {mapa.cidades.map((c, i) => (
                  <tr key={c.id}>
                    <td style={{ padding: '4px 8px', color: '#888', whiteSpace: 'nowrap', borderRight: '1px solid #333' }}>{abreviar(c.nome)}</td>
                    {mapa.cidades.map((_, j) => {
                      const v = mapa.matriz[i][j];
                      const daTabela = mapa.fonte[i][j];
                      return (
                        <td key={j} style={{
                          padding: '4px 8px', textAlign: 'right',
                          color: i === j ? '#444' : daTabela ? 'var(--cyan)' : '#c98aff',
                          fontStyle: !daTabela && i !== j ? 'italic' : 'normal',
                        }}>
                          {i === j ? '·' : v.toFixed(0)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Educacional */}
      <Card title="Como funciona — TSP por AG com seleção por Ranking">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Cromossomo (Aula 13):</b> uma <b>permutação</b> das cidades = ordem de visita do ciclo fechado
          (parte de Uberaba, visita todas uma vez, volta). O fitness é a <b>distância total</b> — quanto menor, melhor.
          <br /><br />
          <b>Cruzamento sem repetir cidade:</b> crossover de bit-string quebra aqui (gera cidade repetida/faltando),
          então usamos operadores de permutação:
          <ul style={{ marginLeft: 18 }}>
            <li><b>OX</b> (Order Crossover): copia um trecho de um pai e completa na ordem do outro.</li>
            <li><b>PMX</b> (Partially Mapped): copia um trecho e resolve conflitos pela cadeia de mapeamento.</li>
          </ul>
          <b>Mutação:</b> <b>swap</b> (troca duas cidades — exemplo do slide) ou <b>inversão</b> (reverte um segmento).
          <br /><br />
          <b>Seleção por RANKING (Aula 16) — o foco do Trabalho 14:</b> a chance de ser pai <b>não</b> depende do valor
          absoluto do fitness, só da <b>posição</b> no ranking. Isso combate convergência prematura, domínio dos
          extremamente aptos e perda de diversidade (problemas da roleta proporcional).
          <ul style={{ marginLeft: 18 }}>
            <li><b>Linear</b> (Baker, 1985): <code>P_i = (1/N)·[η_max − (η_max − η_min)·(i−1)/(N−1)]</code>, com <code>η_min = 2 − η_max</code>.</li>
            <li><b>Exponencial</b>: <code>P_i = c^(N−i) / Σ c^(N−j)</code>, com <code>c &gt; 1</code> controlando a pressão.</li>
          </ul>
          Compare no laboratório acima: aumentar η_max (linear) ou c (exponencial) aumenta a <b>pressão seletiva</b>
          (favorece mais os primeiros), enquanto valores baixos preservam diversidade. Troque a seleção para
          <b> Roleta</b> e veja a convergência ficar mais instável — é a "armadilha" que o ranking resolve.
        </div>
      </Card>
    </div>
  );
}
