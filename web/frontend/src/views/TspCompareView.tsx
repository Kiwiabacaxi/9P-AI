import { useState, useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Tooltip as LMTooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import Card from '../components/shared/Card';
import Select from '../components/shared/Select';
import { useToast } from '../components/shared/Toast';
import { apiGet, apiPost, apiSSE } from '../api/client';
import type {
  TspCidade, TspPreset, TspPresetMeta,
  TspBaselineResult, TspStep, TspResult,
  TspDistMode,
} from '../api/types';

// =============================================================================
// TSP Comparativo — roda Nearest Neighbor, 2-opt e GA na MESMA matriz e mostra
// as três rotas no mapa lado a lado, com tabela de distância + tempo.
// =============================================================================

const DIST_OPTIONS = [
  { value: 'haversine',  label: 'Haversine (linha reta)' },
  { value: 'osrm',       label: 'OSRM (estrada real)' },
  { value: 'euclidiana', label: 'Euclidiana (graus)' },
];

// Cores das polylines de cada algoritmo no mapa.
const COLORS: Record<string, string> = {
  nn:   '#00ccff', // ciano — Nearest Neighbor
  '2opt': '#ffaa00', // laranja — 2-opt
  ga:   '#ff00aa', // rosa — GA
};

const NAMES: Record<string, string> = {
  nn:   'Nearest Neighbor',
  '2opt': '2-opt',
  ga:   'GA (genético)',
};

interface AlgoState {
  tour: number[];
  distancia: number;
  tempoMs: number;
  detalhe?: string; // ex: "300 gerações" pra GA
}

function cityIcon(num: number, isStart: boolean): L.DivIcon {
  const bg = isStart ? '#ff00aa' : '#0a0a0a';
  const border = isStart ? '#ff00aa' : '#888';
  const color = isStart ? '#fff' : '#aaa';
  return L.divIcon({
    className: 'tsp-city-marker',
    html: `<div style="
      width: 22px; height: 22px;
      border-radius: 50%;
      background: ${bg};
      border: 2px solid ${border};
      color: ${color};
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      font-weight: bold;
      display: flex; align-items: center; justify-content: center;
    ">${num}</div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}

function FitBounds({ cidades }: { cidades: TspCidade[] }) {
  const map = useMap();
  const lastKey = useRef('');
  useEffect(() => {
    if (cidades.length === 0) return;
    const key = cidades.map(c => `${c.lat},${c.lng}`).join('|');
    if (key === lastKey.current) return;
    lastKey.current = key;
    const bounds = L.latLngBounds(cidades.map(c => [c.lat, c.lng]));
    map.fitBounds(bounds, { padding: [40, 40] });
  }, [cidades, map]);
  return null;
}

function tourToLatLngs(tour: number[], cidades: TspCidade[]): [number, number][] {
  if (tour.length === 0) return [];
  const byId = new Map(cidades.map(c => [c.id, c]));
  const pts: [number, number][] = [];
  for (const id of tour) {
    const c = byId.get(id);
    if (c) pts.push([c.lat, c.lng]);
  }
  if (pts.length > 0) pts.push(pts[0]);
  return pts;
}

export default function TspCompareView() {
  const { show } = useToast();

  const [presets, setPresets] = useState<TspPresetMeta[]>([]);
  const [preset, setPreset] = useState<string>('itambe-leite');
  const [presetMeta, setPresetMeta] = useState<TspPreset | null>(null);
  const [cidades, setCidades] = useState<TspCidade[]>([]);
  const [distMode, setDistMode] = useState<TspDistMode>('haversine');
  const [matrizPronta, setMatrizPronta] = useState(false);

  const [running, setRunning] = useState(false);
  const [nnRes, setNnRes] = useState<AlgoState | null>(null);
  const [twoOptRes, setTwoOptRes] = useState<AlgoState | null>(null);
  const [gaRes, setGaRes] = useState<AlgoState | null>(null);
  const [gaProgress, setGaProgress] = useState<{ geracao: number; melhor: number } | null>(null);

  const [show_, setShow] = useState({ nn: true, '2opt': true, ga: true });

  const closeSSE = useRef<(() => void) | null>(null);

  async function carregarPreset(name: string) {
    setMatrizPronta(false);
    setNnRes(null);
    setTwoOptRes(null);
    setGaRes(null);
    setGaProgress(null);
    const p = await apiGet<TspPreset>(`/tsp/preset?name=${encodeURIComponent(name)}`);
    await apiPost('/tsp/cities', p.cidades);
    await apiPost('/tsp/distancias', { modo: p.modoSugerido });
    setCidades(p.cidades);
    setPresetMeta(p);
    setDistMode(p.modoSugerido);
    setMatrizPronta(true);
    return p;
  }

  // Mount: lista presets + carrega default
  useEffect(() => {
    (async () => {
      try {
        const list = await apiGet<TspPresetMeta[]>('/tsp/presets');
        setPresets(list);
        await carregarPreset('itambe-leite');
      } catch (e) {
        show('Erro: ' + (e instanceof Error ? e.message : String(e)));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handlePresetChange(novo: string) {
    setPreset(novo);
    if (closeSSE.current) {
      closeSSE.current();
      closeSSE.current = null;
    }
    setRunning(false);
    try {
      const p = await carregarPreset(novo);
      show(`Cenário: ${p.nome}`);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  async function handleDistModeChange(novo: string) {
    const modo = novo as TspDistMode;
    setDistMode(modo);
    setMatrizPronta(false);
    setNnRes(null);
    setTwoOptRes(null);
    setGaRes(null);
    setGaProgress(null);
    try {
      await apiPost('/tsp/distancias', { modo });
      setMatrizPronta(true);
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
    }
  }

  async function handleRun() {
    if (!matrizPronta) return;
    setRunning(true);
    setNnRes(null);
    setTwoOptRes(null);
    setGaRes(null);
    setGaProgress(null);

    try {
      // 1) Nearest Neighbor (instantâneo)
      const nn = await apiPost<TspBaselineResult>('/tsp/baseline', { algoritmo: 'nn', inicio: 0 });
      setNnRes({
        tour: nn.tour, distancia: nn.distancia, tempoMs: nn.tempoMs,
        detalhe: 'greedy puro',
      });

      // 2) NN + 2-opt (busca local sobre o NN)
      const opt = await apiPost<TspBaselineResult>('/tsp/baseline', { algoritmo: 'nn+2opt', inicio: 0 });
      setTwoOptRes({
        tour: opt.tour, distancia: opt.distancia, tempoMs: opt.tempoMs,
        detalhe: 'NN + 2-opt',
      });

      // 3) GA via SSE (streaming)
      const cfg = {
        popSize: 80, maxGeracoes: 300,
        probCruzamento: 0.85, probMutacao: 0.15,
        selecao: 'torneio', tamanhoTorneio: 4,
        cruzamento: 'ox', mutacao: 'inversao',
        elitismo: 2, lambdaMaxLeg: 0, lastVisit: -1,
        gamma: 0, jornadaMaxSec: 36000, muOvertime: 0,
      };
      await apiPost('/tsp/config', cfg);

      const t0 = performance.now();
      closeSSE.current = apiSSE('/tsp/train', {
        onMessage(data) {
          const step = data as TspStep;
          setGaProgress({ geracao: step.geracao, melhor: step.melhorDist });
        },
        onDone(data) {
          const r = data as TspResult;
          const tempoMs = Math.round(performance.now() - t0);
          setGaRes({
            tour: r.melhorTour, distancia: r.melhorDist, tempoMs,
            detalhe: `${r.geracoes} gerações`,
          });
          setGaProgress(null);
          setRunning(false);
          closeSSE.current = null;
        },
        onError() {
          setRunning(false);
          closeSSE.current = null;
        },
      });
    } catch (e) {
      show('Erro: ' + (e instanceof Error ? e.message : String(e)));
      setRunning(false);
    }
  }

  const center: [number, number] = useMemo(() => {
    if (cidades.length === 0) return [-15, -55];
    const lat = cidades.reduce((a, c) => a + c.lat, 0) / cidades.length;
    const lng = cidades.reduce((a, c) => a + c.lng, 0) / cidades.length;
    return [lat, lng];
  }, [cidades]);

  // Tabela: ordena por distância e calcula diff% vs melhor
  const ranking = useMemo(() => {
    const rows: { algo: string; nome: string; res: AlgoState; diffPct: number | null }[] = [];
    if (nnRes) rows.push({ algo: 'nn', nome: NAMES.nn, res: nnRes, diffPct: 0 });
    if (twoOptRes) rows.push({ algo: '2opt', nome: NAMES['2opt'], res: twoOptRes, diffPct: 0 });
    if (gaRes) rows.push({ algo: 'ga', nome: NAMES.ga, res: gaRes, diffPct: 0 });
    if (rows.length === 0) return rows;
    const melhor = Math.min(...rows.map(r => r.res.distancia));
    return rows
      .map(r => ({ ...r, diffPct: ((r.res.distancia - melhor) / melhor) * 100 }))
      .sort((a, b) => a.res.distancia - b.res.distancia);
  }, [nnRes, twoOptRes, gaRes]);

  const unidade = distMode === 'euclidiana' ? 'graus' : 'km';

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">TSP <span>Comparativo</span></div>
          <div className="page-sub">
            Mesma matriz · 3 algoritmos lado a lado &mdash; Aula 12
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-primary" onClick={handleRun} disabled={running || !matrizPronta}>
            {running && <span className="spin" />}
            COMPARAR
          </button>
        </div>
      </div>

      <div className="grid-3" style={{ marginBottom: 16 }}>
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
        </Card>

        <Card style={{ padding: '16px 20px', gridColumn: 'span 2' }}>
          <div className="imgreg-select-label">Cenário ativo</div>
          {presetMeta ? (
            <div style={{ fontSize: 12, lineHeight: 1.6, fontFamily: 'JetBrains Mono', color: 'var(--muted)' }}>
              <div><b style={{ color: 'var(--cyan)' }}>{presetMeta.nome}</b></div>
              <div>{cidades.length} pontos · origem: <code>{presetMeta.origem}</code></div>
              <div>matriz pronta: <code>{matrizPronta ? '✓' : '…'}</code> · modo: <code>{distMode}</code></div>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--muted)' }}>carregando&hellip;</div>
          )}
        </Card>
      </div>

      {/* Toggle das polylines */}
      <Card style={{ marginBottom: 12, padding: '10px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 11, fontFamily: 'JetBrains Mono' }}>
          <span style={{ color: 'var(--muted)' }}>mostrar:</span>
          {(['nn', '2opt', 'ga'] as const).map(algo => (
            <label key={algo} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={show_[algo]}
                onChange={e => setShow(s => ({ ...s, [algo]: e.target.checked }))}
                style={{ accentColor: COLORS[algo] }}
              />
              <span style={{
                display: 'inline-block', width: 14, height: 3,
                background: COLORS[algo], borderRadius: 1,
              }} />
              <span style={{ color: COLORS[algo] }}>{NAMES[algo]}</span>
            </label>
          ))}
        </div>
      </Card>

      <Card title="Mapa comparativo (3 algoritmos sobrepostos)" style={{ marginBottom: 16 }}>
        <div style={{ height: 500, borderRadius: 6, overflow: 'hidden' }}>
          <MapContainer
            center={center}
            zoom={5}
            style={{ height: '100%', width: '100%', background: '#0a0a0a' }}
            attributionControl={false}
          >
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png" maxZoom={18} />
            <FitBounds cidades={cidades} />

            {nnRes && show_.nn && (
              <Polyline
                positions={tourToLatLngs(nnRes.tour, cidades)}
                pathOptions={{ color: COLORS.nn, weight: 3, opacity: 0.7 }}
              />
            )}
            {twoOptRes && show_['2opt'] && (
              <Polyline
                positions={tourToLatLngs(twoOptRes.tour, cidades)}
                pathOptions={{ color: COLORS['2opt'], weight: 3, opacity: 0.7 }}
              />
            )}
            {gaRes && show_.ga && (
              <Polyline
                positions={tourToLatLngs(gaRes.tour, cidades)}
                pathOptions={{ color: COLORS.ga, weight: 3, opacity: 0.85 }}
              />
            )}

            {cidades.map(c => (
              <Marker key={c.id} position={[c.lat, c.lng]} icon={cityIcon(c.id + 1, c.id === 0)}>
                <LMTooltip direction="top" offset={[0, -12]} opacity={0.95}>
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11 }}>
                    <b>{c.nome}</b>{c.uf ? ` / ${c.uf}` : ''}
                  </div>
                </LMTooltip>
              </Marker>
            ))}
          </MapContainer>
        </div>
        {gaProgress && (
          <div style={{
            marginTop: 8, padding: '6px 12px',
            background: 'var(--surface-2)', borderRadius: 6,
            fontSize: 11, color: 'var(--muted)', fontFamily: 'JetBrains Mono',
          }}>
            <span className="spin" /> GA: geração {gaProgress.geracao}/300 — melhor atual {gaProgress.melhor.toFixed(1)} {unidade}
          </div>
        )}
      </Card>

      {/* Tabela comparativa */}
      {ranking.length > 0 && (
        <Card title="Resultados" style={{ marginBottom: 16 }}>
          <div style={{ padding: 12, overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'JetBrains Mono' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--surface-2)', color: 'var(--muted)' }}>
                  <th style={{ textAlign: 'left',  padding: '8px 12px' }}>Algoritmo</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Distância</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Δ vs melhor</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Tempo</th>
                  <th style={{ textAlign: 'left',  padding: '8px 12px' }}>Detalhe</th>
                </tr>
              </thead>
              <tbody>
                {ranking.map((row, i) => (
                  <tr key={row.algo} style={{ borderBottom: '1px solid var(--surface-2)' }}>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        display: 'inline-block', width: 10, height: 10,
                        borderRadius: 2, background: COLORS[row.algo], marginRight: 8,
                      }} />
                      <span style={{ color: COLORS[row.algo] }}>{row.nome}</span>
                      {i === 0 && <span style={{ marginLeft: 6, color: 'var(--primary-glow)' }}>★ melhor</span>}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                      {row.res.distancia.toFixed(1)} {unidade}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right', color: row.diffPct === 0 ? 'var(--primary-glow)' : 'var(--muted)' }}>
                      {row.diffPct === 0 ? '—' : `+${row.diffPct?.toFixed(2)}%`}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                      {row.res.tempoMs < 10 ? '<10 ms' : `${row.res.tempoMs.toLocaleString()} ms`}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--muted)' }}>
                      {row.res.detalhe ?? ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Como cada algoritmo decide a rota">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <ul style={{ marginLeft: 18 }}>
            <li>
              <b style={{ color: COLORS.nn }}>Nearest Neighbor</b> — começa na cidade 0 (depot) e a cada
              passo vai pra <i>cidade ainda não visitada mais próxima</i>. Greedy puro,
              O(N²), instantâneo. Tipicamente 25% pior que o ótimo, mas é uma "rota
              razoável" muito rapidamente. Sofre quando dois pontos estão perto no início
              mas exigem volta longa no fim.
            </li>
            <li>
              <b style={{ color: COLORS['2opt'] }}>2-opt (sobre NN)</b> — pega o tour do NN
              e tenta inversões: pra cada par (i, j), reverte o segmento entre eles e
              verifica se o tour fica mais curto. Repete até estabilizar. Conserta
              "cruzamentos" no traçado — quase sempre o que torna um tour ruim. Tipicamente
              5–10% pior que o ótimo, e ainda sub-segundo pra N pequeno.
            </li>
            <li>
              <b style={{ color: COLORS.ga }}>GA (algoritmo genético)</b> — população de
              80 tours aleatórios evolui por 300 gerações usando torneio + OX + inversão
              (operadores idênticos aos que estudamos nas aulas 10–11). É <i>estocástico</i>
              (varia entre runs), mais lento (segundos) e o resultado depende dos
              parâmetros. Em compensação, escapa de mínimos locais que prendem 2-opt.
            </li>
          </ul>
          <br />
          <b>Quando o GA brilha:</b> instâncias grandes (N &gt; 50), geometrias
          enviesadas (clusters), problemas com restrições adicionais (janelas, capacidade,
          múltiplos veículos). Pra TSP "bem-comportado" pequeno, NN+2-opt já entrega
          quase a mesma coisa em milissegundos. Faz parte do papel didático aqui:
          mostrar que <i>nem sempre o algoritmo mais sofisticado é o que vence</i> —
          mas vence quando o problema se complica.
        </div>
      </Card>
    </div>
  );
}
