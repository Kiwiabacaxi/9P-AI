import { useState, useRef, useEffect, useMemo } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Legend, Tooltip,
} from 'recharts';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  RnaGaConfig, RnaGaStep, RnaGaResult, RnaGaIndividuo,
  RnaGaBenchModo, RnaGaBenchResult, RnaGaBenchSaved,
} from '../api/types';

function fmtMs(ms: number): string {
  if (ms >= 60000) return `${(ms / 60000).toFixed(1)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${ms.toFixed(0)} ms`;
}
// cores dos 5 modos (ingênuo → atual)
const BAR_COR = ['#ff3b3b', '#ff9d2e', '#e0c000', '#00ccff', '#22c55e'];
const BENCH_PRESETS = [
  { value: 'amostra', label: 'amostra (16×8 · ~25 s)' },
  { value: 'media', label: 'média (24×25 · ~5 min)' },
  { value: 'cheio', label: 'cheio 40×100 · ~70 min!' },
];

// =============================================================================
// Trabalho 15 — AG que descobre a melhor arquitetura de uma MLP (RNA + AG).
// =============================================================================

const POP_OPTIONS = [20, 40, 60].map(n => ({ value: String(n), label: String(n) }));
const GER_OPTIONS = [20, 50, 100].map(n => ({ value: String(n), label: String(n) }));
const TETO_OPTIONS = [50, 100, 200, 300, 500, 1000].map(n => ({ value: String(n), label: `${n} épocas` }));
const PM_OPTIONS = [
  { value: '0.05', label: '5% (enunciado)' },
  { value: '0.1', label: '10%' },
  { value: '0.2', label: '20%' },
];

const NEURO_MIN = 2, NEURO_MAX = 15, CAM_MIN = 2, CAM_MAX = 5;

// Cor da célula do heatmap por MSE (log): verde (melhor) → amarelo → vermelho.
function corMSE(v: number, lmin: number, lmax: number): string {
  if (v < 0) return '#141414';
  const t = lmax > lmin ? (Math.log(v) - lmin) / (lmax - lmin) : 0.5;
  const tt = Math.max(0, Math.min(1, t));
  // verde(120) → vermelho(0)
  const hue = 120 * (1 - tt);
  return `hsl(${hue}, 70%, ${28 + 14 * (1 - tt)}%)`;
}

export default function RnaGaView() {
  const { show } = useToast();

  const [popSize, setPopSize] = useState('40');
  const [maxGer, setMaxGer] = useState('100');
  const [teto, setTeto] = useState('300');
  const [pm, setPm] = useState('0.05');

  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<RnaGaResult | null>(null);
  const [frames, setFrames] = useState<RnaGaStep[]>([]);
  const framesRef = useRef<RnaGaStep[]>([]);
  const closeSSE = useRef<(() => void) | null>(null);

  // Benchmark
  const [benchPreset, setBenchPreset] = useState('amostra');
  const [benchModos, setBenchModos] = useState<RnaGaBenchModo[]>([]);
  const [benchResult, setBenchResult] = useState<RnaGaBenchResult | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchSalvos, setBenchSalvos] = useState<RnaGaBenchSaved[]>([]);
  const benchRef = useRef<RnaGaBenchModo[]>([]);
  const closeBench = useRef<(() => void) | null>(null);

  function carregarLista() {
    apiGet<RnaGaBenchSaved[]>('/rnaga/benchmarks')
      .then(l => setBenchSalvos(l.sort((a, b) => b.timestampUnix - a.timestampUnix)))
      .catch(() => {});
  }

  useEffect(() => {
    carregarLista();
    return () => {
      if (closeSSE.current) closeSSE.current();
      if (closeBench.current) closeBench.current();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleBenchmark() {
    if (benchPreset === 'cheio' && !window.confirm(
      'O preset "cheio" (40×100) roda os 5 modos no tamanho real — os dois modos de 1 core levam ~29 min cada e o total passa de ~70 min com o navegador aberto. Recomendado rodar pela CLI (cmd/rnabench). Continuar mesmo assim?'
    )) return;
    setBenchRunning(true);
    setBenchResult(null);
    benchRef.current = [];
    setBenchModos([]);
    closeBench.current = apiSSE(`/rnaga/benchmark?preset=${benchPreset}`, {
      onMessage(data) {
        benchRef.current = [...benchRef.current, data as RnaGaBenchModo].sort((a, b) => a.ordem - b.ordem);
        setBenchModos(benchRef.current.slice());
      },
      onDone(data) {
        setBenchResult(data as RnaGaBenchResult);
        setBenchRunning(false);
        closeBench.current = null;
        carregarLista();
      },
      onError() {
        setBenchRunning(false);
        closeBench.current = null;
        show('Erro no benchmark (stream)');
      },
    });
  }

  async function handleCarregarSalvo(nome: string) {
    if (!nome) return;
    try {
      const r = await apiGet<RnaGaBenchResult>(`/rnaga/benchmark/load?nome=${encodeURIComponent(nome)}`);
      setBenchResult(r);
      benchRef.current = [...r.modos].sort((a, b) => a.ordem - b.ordem);
      setBenchModos(benchRef.current.slice());
      setBenchPreset(r.preset || 'amostra');
    } catch {
      show('Erro ao carregar benchmark salvo');
    }
  }

  function handleBaixarJSON() {
    if (!benchResult) return;
    const blob = new Blob([JSON.stringify(benchResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rnaga-bench-${benchResult.preset || 'amostra'}-${benchResult.timestampUnix}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleTrain() {
    setTraining(true);
    framesRef.current = [];
    setFrames([]);
    setResult(null);
    const cfg: RnaGaConfig = {
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGer),
      probMutacao: parseFloat(pm),
      tetoEpocas: parseInt(teto),
    };
    try {
      await apiPost('/rnaga/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      return;
    }
    closeSSE.current = apiSSE('/rnaga/train', {
      onMessage(data) {
        framesRef.current.push(data as RnaGaStep);
        setFrames(framesRef.current.slice());
      },
      onDone(data) {
        const r = data as RnaGaResult;
        setResult(r);
        setTraining(false);
        closeSSE.current = null;
        show(`Melhor arquitetura: ${r.melhorView.string} · MSE ${r.melhorMse.toFixed(2)}`);
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
    try { await apiPost('/rnaga/reset'); } catch { /* ignore */ }
    framesRef.current = [];
    setFrames([]);
    setResult(null);
    setTraining(false);
    show('RNA+AG resetado');
  }

  const last = frames.length ? frames[frames.length - 1] : null;
  const melhor: RnaGaIndividuo | null = result?.melhorView ?? last?.melhorCromossomo ?? null;
  const geracao = last?.geracao ?? 0;
  const melhorMSE = result?.melhorMse ?? last?.melhorGlobalMse ?? null;
  const mediaMSE = last?.mediaMse ?? null;

  const chartData = useMemo(
    () => frames.map(f => ({ gen: f.geracao, melhor: f.melhorMse, melhorAcum: f.melhorGlobalMse, media: f.mediaMse })),
    [frames],
  );

  // ----- Diagrama SVG da melhor rede (15 → camadas×neurônios → 13) -----
  const diagrama = useMemo(() => {
    if (!melhor) return null;
    const sizes = [15, ...Array(melhor.camadas).fill(melhor.neuronios), 13];
    const cols = sizes.length;
    const W = 920, H = 380, padX = 80, padY = 28;
    const colX = (i: number) => padX + (i * (W - 2 * padX)) / (cols - 1);
    const nodeY = (count: number, idx: number) => {
      if (count === 1) return H / 2;
      const gap = (H - 2 * padY) / (count - 1);
      return padY + idx * gap;
    };
    const corCol = (i: number) => (i === 0 ? '#00ccff' : i === cols - 1 ? '#22c55e' : '#ff00aa');

    // Arestas na cor da coluna de origem, com traço em px de tela
    // (vectorEffect) — senão o reescalonamento do SVG some com as linhas.
    const edges: React.ReactNode[] = [];
    for (let i = 0; i < cols - 1; i++) {
      const cor = i === 0 ? '#00ccff' : i === cols - 2 ? '#22c55e' : '#ff00aa';
      for (let a = 0; a < sizes[i]; a++) {
        for (let b = 0; b < sizes[i + 1]; b++) {
          edges.push(
            <line key={`e-${i}-${a}-${b}`} x1={colX(i)} y1={nodeY(sizes[i], a)}
              x2={colX(i + 1)} y2={nodeY(sizes[i + 1], b)} stroke={cor} strokeWidth={0.7}
              strokeOpacity={0.16} vectorEffect="non-scaling-stroke" />,
          );
        }
      }
    }
    const nodes: React.ReactNode[] = [];
    const labels: React.ReactNode[] = [];
    for (let i = 0; i < cols; i++) {
      const lbl = i === 0 ? '15 entradas' : i === cols - 1 ? '13 saídas' : `oculta ${i} (${sizes[i]})`;
      labels.push(
        <text key={`l-${i}`} x={colX(i)} y={H - 6} fill="#888" fontSize={11}
          fontFamily="JetBrains Mono" textAnchor="middle">{lbl}</text>,
      );
      for (let n = 0; n < sizes[i]; n++) {
        nodes.push(
          <circle key={`n-${i}-${n}`} cx={colX(i)} cy={nodeY(sizes[i], n)} r={7}
            fill="#0a0a0a" stroke={corCol(i)} strokeWidth={2}
            style={{ transition: 'cx .45s ease, cy .45s ease' }} />,
        );
      }
    }
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxHeight: 420 }}>
        {edges}{nodes}{labels}
      </svg>
    );
  }, [melhor]);

  // ----- Heatmap (neurônios × camadas) -----
  const heat = useMemo(() => {
    const grade = last?.gradeMse;
    if (!grade) return null;
    let lmin = Infinity, lmax = -Infinity;
    for (const row of grade) for (const v of row) {
      if (v >= 0) { const l = Math.log(v); if (l < lmin) lmin = l; if (l > lmax) lmax = l; }
    }
    return { grade, lmin, lmax };
  }, [last]);

  const popOrdenada = useMemo(
    () => (last ? [...last.populacao].sort((a, b) => a.mse - b.mse) : []),
    [last],
  );

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">AG <span>Arquitetura RNA</span></div>
          <div className="page-sub">
            AG descobre a melhor arquitetura de MLP — Trabalho 15 · Aula 20 · 15 entradas → 13 saídas ·
            fitness = MSE da rede treinada (minimizar)
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

      {/* Config */}
      <div className="grid-2" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select label="População" options={POP_OPTIONS} value={popSize} onChange={setPopSize} style={{ flex: 1 }} />
            <Select label="Gerações" options={GER_OPTIONS} value={maxGer} onChange={setMaxGer} style={{ flex: 1 }} />
          </div>
        </Card>
        <Card style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select label="Teto de épocas (treino de cada rede)" options={TETO_OPTIONS} value={teto} onChange={setTeto} style={{ flex: 1 }} />
            <Select label="Mutação" options={PM_OPTIONS} value={pm} onChange={setPm} style={{ width: 150 }} />
          </div>
          <div style={{ fontSize: 11, color: '#777', marginTop: 6, fontFamily: 'JetBrains Mono' }}>
            teto menor = demo mais rápida · maior = busca mais completa (treina cada arquitetura por mais épocas)
          </div>
        </Card>
      </div>

      {/* Métricas */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
        <MetricCard title="Geração" value={geracao ? geracao.toLocaleString() : '—'} label={`de ${parseInt(maxGer)}`} color="green" pulse={training} />
        <MetricCard title="Melhor MSE" value={melhorMSE !== null ? melhorMSE.toFixed(2) : '—'} label="erro da melhor arquitetura (menor = melhor)" color="cyan" />
        <MetricCard title="MSE médio da pop" value={mediaMSE !== null ? mediaMSE.toFixed(0) : '—'} label="inclui arquiteturas ruins (sem normalização)" />
      </div>

      {/* Diagrama da melhor rede */}
      <Card title={`Melhor arquitetura${training ? ' (evoluindo ao vivo)' : ''}`} style={{ marginBottom: 16 }}>
        <div style={{ padding: 12 }}>
          {melhor ? (
            <>
              <div style={{
                display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 10, alignItems: 'center',
                fontFamily: 'JetBrains Mono', fontSize: 12, color: 'var(--muted)',
                padding: '8px 12px', background: 'var(--surface-2)', borderRadius: 6,
              }}>
                <span>cromossomo: <b style={{ color: 'var(--cyan)' }}>{melhor.string}</b></span>
                <span style={{ color: '#555' }}>|</span>
                <span>MSE: <b style={{ color: '#ffff00' }}>{melhor.mse.toFixed(2)}</b></span>
                <span style={{ color: '#555' }}>|</span>
                <span><b>{melhor.camadas}</b> camadas × <b>{melhor.neuronios}</b> neurônios</span>
                <span style={{ color: melhor.online ? 'var(--cyan)' : '#ff9d2e' }}>{melhor.online ? 'online' : 'offline'}</span>
                <span style={{ color: melhor.normaliza ? '#22c55e' : '#ff3b3b' }}>{melhor.normaliza ? 'normaliza' : 'sem normalização'}</span>
              </div>
              {diagrama}
            </>
          ) : (
            <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontFamily: 'JetBrains Mono' }}>
              clique EVOLUIR — o AG vai treinar várias arquiteturas e desenhar a melhor aqui
            </div>
          )}
        </div>
      </Card>

      {/* Heatmap do espaço de busca */}
      {heat && (
        <Card title="Espaço de busca — melhor MSE por (neurônios × nº de camadas)" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 10, fontFamily: 'JetBrains Mono', lineHeight: 1.6 }}>
              Cada célula = menor MSE já encontrado para aquela combinação de arquitetura.
              <b style={{ color: 'hsl(120,70%,40%)' }}> verde</b> = bom, <b style={{ color: 'hsl(0,70%,45%)' }}>vermelho</b> = ruim,
              <b style={{ color: '#444' }}> escuro</b> = ainda não testada. O AG concentra a busca onde está mais verde.
            </div>
            <div style={{ display: 'flex', gap: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }}>
              {/* eixo Y (neurônios) */}
              <div style={{ display: 'flex', flexDirection: 'column-reverse', justifyContent: 'space-between', color: '#888', paddingBottom: 18 }}>
                {Array.from({ length: NEURO_MAX - NEURO_MIN + 1 }, (_, i) => (
                  <div key={i} style={{ height: 22, display: 'flex', alignItems: 'center' }}>{NEURO_MIN + i}</div>
                ))}
              </div>
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${CAM_MAX - CAM_MIN + 1}, 64px)`, gridAutoRows: 22, gap: 2 }}>
                  {/* linhas de cima (15) para baixo (2) */}
                  {Array.from({ length: NEURO_MAX - NEURO_MIN + 1 }, (_, ri) => {
                    const neuro = NEURO_MAX - ri; // topo = 15
                    return Array.from({ length: CAM_MAX - CAM_MIN + 1 }, (_, ci) => {
                      const cam = CAM_MIN + ci;
                      const v = heat.grade[neuro - 2][cam - 2];
                      const best = melhor && melhor.neuronios === neuro && melhor.camadas === cam;
                      return (
                        <div key={`${ri}-${ci}`} title={v >= 0 ? `${neuro} neurônios × ${cam} camadas → MSE ${v.toFixed(1)}` : 'não testada'}
                          style={{
                            background: corMSE(v, heat.lmin, heat.lmax),
                            border: best ? '2px solid #ffff00' : '1px solid #222',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            color: '#eee', fontSize: 9,
                          }}>
                          {v >= 0 ? (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)) : ''}
                        </div>
                      );
                    });
                  })}
                </div>
                {/* eixo X (camadas) */}
                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${CAM_MAX - CAM_MIN + 1}, 64px)`, gap: 2, color: '#888', marginTop: 4 }}>
                  {Array.from({ length: CAM_MAX - CAM_MIN + 1 }, (_, ci) => (
                    <div key={ci} style={{ textAlign: 'center' }}>{CAM_MIN + ci} cam</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Convergência */}
      {chartData.length > 0 && (
        <Card title="Convergência — MSE por geração (escala log)" style={{ marginBottom: 16 }}>
          <div style={{ padding: '8px 4px' }}>
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="gen" stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} />
                <YAxis stroke="#555" tick={{ fill: '#555', fontSize: 10, fontFamily: 'JetBrains Mono' }} tickLine={false} scale="log" domain={['auto', 'auto']} allowDataOverflow />
                <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11, fontFamily: 'JetBrains Mono' }} labelFormatter={(v) => `geração ${v}`} formatter={(v) => Number(v).toFixed(2)} />
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Line name="MSE médio da pop" type="monotone" dataKey="media" stroke="#00ccff" strokeWidth={1} strokeOpacity={0.6} strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                <Line name="melhor da geração" type="monotone" dataKey="melhor" stroke="#ff00aa" strokeWidth={1} strokeOpacity={0.5} dot={false} isAnimationActive={false} />
                <Line name="melhor acumulado" type="monotone" dataKey="melhorAcum" stroke="#ffff00" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* População */}
      {popOrdenada.length > 0 && (
        <Card title="População — arquiteturas testadas nesta geração (ordenadas por MSE)" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontFamily: 'JetBrains Mono', fontSize: 12, width: '100%' }}>
              <thead>
                <tr style={{ color: '#888' }}>
                  <th style={{ padding: '4px 10px', textAlign: 'left', borderBottom: '1px solid #333' }}>#</th>
                  <th style={{ padding: '4px 10px', textAlign: 'left', borderBottom: '1px solid #333' }}>cromossomo (neur | cam | taxa | épocas | modo | norm)</th>
                  <th style={{ padding: '4px 10px', textAlign: 'right', borderBottom: '1px solid #333' }}>MSE</th>
                </tr>
              </thead>
              <tbody>
                {popOrdenada.slice(0, 12).map((ind, i) => (
                  <tr key={i} style={{ background: i === 0 ? 'rgba(255,255,0,0.06)' : undefined }}>
                    <td style={{ padding: '3px 10px', color: '#666' }}>{i + 1}</td>
                    <td style={{ padding: '3px 10px', color: ind.normaliza ? 'var(--cyan)' : '#ff7a7a' }}>{ind.string}</td>
                    <td style={{ padding: '3px 10px', textAlign: 'right', color: i === 0 ? '#ffff00' : 'var(--muted)' }}>{ind.mse.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: '#777', marginTop: 8, fontFamily: 'JetBrains Mono' }}>
              Em <b style={{ color: 'var(--cyan)' }}>ciano</b>: arquiteturas que normalizam (dominam o topo). Em
              <b style={{ color: '#ff7a7a' }}> vermelho</b>: sem normalização (tanh satura → MSE altíssimo).
            </div>
          </div>
        </Card>
      )}

      {/* Vencedor (entregável) */}
      {result && (
        <Card title="🏆 Melhor arquitetura encontrada pelo AG" style={{ marginBottom: 16 }}>
          <div style={{ padding: 16, fontFamily: 'JetBrains Mono', fontSize: 14, color: 'var(--muted)', lineHeight: 1.9 }}>
            <div>cromossomo vencedor: <b style={{ color: 'var(--cyan)' }}>{result.melhorView.string}</b></div>
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #222', fontSize: 13 }}>
              <div>neurônios/camada: <b style={{ color: '#fff' }}>{result.melhorView.neuronios}</b></div>
              <div>nº de camadas ocultas: <b style={{ color: '#fff' }}>{result.melhorView.camadas}</b></div>
              <div>taxa de aprendizagem: <b style={{ color: '#fff' }}>{result.melhorView.genes[2].toFixed(5)}</b></div>
              <div>máx. épocas: <b style={{ color: '#fff' }}>{Math.round(result.melhorView.genes[3])}</b></div>
              <div>treino: <b style={{ color: '#fff' }}>{result.melhorView.online ? 'on-line' : 'off-line'}</b> · dados: <b style={{ color: '#fff' }}>{result.melhorView.normaliza ? 'normalizados' : 'não normalizados'}</b></div>
            </div>
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid #222' }}>
              MSE final = <b style={{ color: '#ffff00' }}>{result.melhorMse.toFixed(4)}</b>
              <span style={{ color: '#888', marginLeft: 8 }}>(em {result.geracoes} gerações)</span>
            </div>
          </div>
        </Card>
      )}

      {/* Benchmark — por que é rápido de verdade */}
      <Card title="⚡ Benchmark — por que treina rápido (e a versão ingênua não treinaria)" style={{ marginBottom: 16 }}>
        <div style={{ padding: 12 }}>
          <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7, marginBottom: 12 }}>
            O professor avisou que treinar tudo isso <b>demoraria muito</b> — e está certo: uma implementação
            ingênua leva minutos. Este benchmark roda <b>o mesmo AG</b> (mesma seed, mesmas arquiteturas) sob 5
            níveis cumulativos de otimização e mede o tempo de cada. Como nenhuma técnica muda <i>o que</i> é
            computado, só <i>quão rápido</i>, o <b>MSE final sai idêntico</b> — a prova de que é o mesmo trabalho.
            Escolha o tamanho, rode, e <b>salve em JSON</b> pra mostrar depois (ou rode pela CLI <code>cmd/rnabench</code>).
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-end', marginBottom: 14 }}>
            <div style={{ minWidth: 230 }}>
              <Select label="Tamanho" options={BENCH_PRESETS} value={benchPreset} onChange={setBenchPreset} style={{ width: '100%' }} />
            </div>
            <button className="btn btn-primary" onClick={handleBenchmark} disabled={benchRunning || training}>
              {benchRunning && <span className="spin" />}
              {benchRunning ? 'MEDINDO… (o modo ingênuo é lento de propósito)' : 'RODAR BENCHMARK'}
            </button>
            {benchSalvos.length > 0 && (
              <div style={{ minWidth: 250 }}>
                <Select label="Carregar resultado salvo" value=""
                  options={[{ value: '', label: '— escolher —' }, ...benchSalvos.map(s => ({
                    value: s.nome, label: `${s.preset || '?'} ${s.popSize}×${s.maxGeracoes} · ${s.speedupTotal.toFixed(1)}×`,
                  }))]}
                  onChange={handleCarregarSalvo} style={{ width: '100%' }} />
              </div>
            )}
            {benchResult && <button className="btn" onClick={handleBaixarJSON} style={{ height: 38 }}>baixar JSON</button>}
          </div>

          {benchModos.length > 0 && (() => {
            const maxMs = Math.max(...benchModos.map(m => m.ms), 1);
            const ingenuo = benchModos.find(m => m.ordem === 0)?.ms ?? 0;
            return (
              <div style={{ marginBottom: 12 }}>
                {benchModos.map(m => {
                  const speedup = m.ordem === 0 || m.ms === 0 ? null : ingenuo / m.ms;
                  return (
                    <div key={m.ordem} style={{ marginBottom: 8, fontFamily: 'JetBrains Mono', fontSize: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, color: 'var(--muted)' }}>
                        <span><b style={{ color: BAR_COR[m.ordem] }}>{m.nome}</b> {m.workers > 1 ? `· ${m.workers} cores` : '· 1 core'}{m.cacheHits > 0 ? ` · ${m.cacheHits} cache hits` : ''}</span>
                        <span style={{ color: '#fff' }}>{fmtMs(m.ms)} {speedup ? <span style={{ color: '#22c55e' }}>({speedup.toFixed(1)}× mais rápido)</span> : ''}</span>
                      </div>
                      <div style={{ height: 18, background: 'var(--surface-2)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ width: `${(m.ms / maxMs) * 100}%`, height: '100%', background: BAR_COR[m.ordem], transition: 'width .4s ease' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })()}

          {benchResult && (
            <div style={{
              display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center',
              padding: '12px 14px', background: 'var(--surface-2)', borderRadius: 6,
              fontFamily: 'JetBrains Mono', fontSize: 13, color: 'var(--muted)', marginBottom: 12,
            }}>
              <span>resultado: <b style={{ color: '#22c55e', fontSize: 18 }}>{benchResult.speedupTotal.toFixed(1)}×</b> mais rápido</span>
              <span style={{ color: '#555' }}>|</span>
              <span>MSE nos 5 modos: <b style={{ color: '#22c55e' }}>
                {benchResult.mesmoMse
                  ? `idêntico (${(benchResult.modos[benchResult.modos.length - 1]?.melhorMse ?? 0).toFixed(2)}) ✓`
                  : `praticamente idêntico (Δmax ${benchResult.maxDiffMse.toExponential(1)}) ✓`}</b></span>
              <span style={{ color: '#555' }}>|</span>
              <span>{benchResult.numCpu} cores · preset {benchResult.preset || '?'}</span>
            </div>
          )}

          {benchResult && (() => {
            const ehCheio = benchResult.preset === 'cheio';
            return (
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.7, marginBottom: 12, fontFamily: 'JetBrains Mono', padding: '8px 12px', background: 'rgba(255,59,59,0.06)', borderRadius: 6 }}>
                <b>{ehCheio ? 'No tamanho cheio (40×100 = 4.000 treinos, MEDIDO)' : 'Extrapolando pro tamanho cheio (40×100 = 4.000 treinos)'}:</b> o modo
                <b style={{ color: '#ff3b3b' }}> ingênuo</b> {ehCheio ? 'levou' : 'levaria'} <b style={{ color: '#ff3b3b' }}>≈ {fmtMs(benchResult.fullIngenuoMs)}</b>,
                enquanto o <b style={{ color: '#22c55e' }}>atual</b> {ehCheio ? 'rodou em' : 'roda em'} <b style={{ color: '#22c55e' }}>≈ {fmtMs(benchResult.fullOtimizadoMs)}</b>
                {ehCheio ? '.' : ' (estimativa linear; na prática o atual é ainda mais rápido — a memoização rende mais a cada geração).'}
              </div>
            );
          })()}

          <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
            <b>As técnicas (cada barra liga uma):</b>
            <ul style={{ marginLeft: 18, marginTop: 6 }}>
              <li><b style={{ color: BAR_COR[1] }}>Lib de matrizes (gonum/BLAS)</b> — no treino em lote (off-line) os 100 padrões viram uma matriz 100×15 e o matmul usa BLAS. Ajuda o off-line, mas como o custo dominante aqui é o treino <b>on-line</b> (padrão a padrão), o ganho é modesto — e o benchmark mostra isso honestamente.</li>
              <li><b style={{ color: BAR_COR[2] }}>Paralelismo</b> — os indivíduos da população são treinados ao mesmo tempo em todos os cores (goroutines), não um de cada vez.</li>
              <li><b style={{ color: BAR_COR[3] }}>Online sem alocação</b> — o backprop padrão-a-padrão reaproveita buffers em vez de alocar a cada padrão (zero pressão de GC).</li>
              <li><b style={{ color: BAR_COR[4] }}>Memoização</b> — arquiteturas repetidas (sobretudo os elites que sobrevivem a cada geração) não retreinam: o MSE fica em cache. Como os pesos têm seed determinístico por arquitetura, o valor em cache é exatamente o que o retreino daria.</li>
            </ul>
            E os próprios modelos são <b>minúsculos</b> (≤15 neurônios × ≤5 camadas), então cada treino é barato — o que pesa é a <b>quantidade</b> de treinos.
          </div>
        </div>
      </Card>

      {/* Educacional */}
      <Card title="Como funciona — AG procurando a arquitetura da RNA">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>A ideia (Aula 20):</b> em vez de escolher a arquitetura da rede na mão, deixamos um <b>algoritmo
          genético</b> procurar. Cada indivíduo é uma <b>configuração de MLP</b>, e o seu <b>fitness é o MSE</b>
          que essa rede atinge sobre os 100 padrões (15 entradas → 13 saídas). Menor MSE = melhor.
          <br /><br />
          <b>Cromossomo</b> = vetor de 6 genes: [neurônios/camada (2–15), nº camadas (2–5), taxa de aprendizagem
          (1e-5–0.1), máx. épocas (20–1000), on-line/off-line, normaliza/não]. Crossover de <b>1 ponto</b>,
          <b> mutação de 5%</b> (sorteia uma posição e gera um valor válido), seleção por <b>roleta</b>,
          substituição <b>elitista</b> (a melhor metade sobrevive), até <b>100 gerações</b>.
          <br /><br />
          <b>Avaliar = treinar:</b> a cada indivíduo a rede é treinada de verdade (tangente hiperbólica em todas
          as camadas, pesos iniciais aleatórios). É caro — então avaliamos os indivíduos <b>em paralelo</b>,
          <b> memoizamos</b> arquiteturas repetidas e usamos a lib de matrizes (gonum) no treino em lote (off-line).
          <br /><br />
          <b>O que o AG descobre:</b> como as saídas alvo (~58–312) não cabem na faixa do tanh (±1), arquiteturas
          <b> sem normalização</b> ficam péssimas (vermelho na tabela/heatmap), e o AG rapidamente migra para
          configurações que <b>normalizam</b> os dados — exatamente a lição do exercício.
        </div>
      </Card>
    </div>
  );
}
