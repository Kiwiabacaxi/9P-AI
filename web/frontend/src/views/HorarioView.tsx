import { useState, useRef, useMemo } from 'react';
import Card from '../components/shared/Card';
import MetricCard from '../components/shared/MetricCard';
import Select from '../components/shared/Select';
import { GaEvoChart } from '../components/viz/GaChart';
import { useToast } from '../components/shared/Toast';
import { apiPost, apiSSE } from '../api/client';
import type {
  HorarioConfig, HorarioStep, HorarioResult, HorarioIndividuo, HorarioProfessor,
} from '../api/types';

// =============================================================================
// HorarioView — GA com cromossomo MATRICIAL (Aula 12)
//
// Cromossomo = matriz [slot × turma], cada célula guarda o ID do professor.
// Cruzamento: troca de LINHAS inteiras entre pais.
// Fitness: + aulas encadeadas, − choques, − matérias faltando.
// =============================================================================

const PROF_OPTIONS = [
  { value: '10', label: '10' },
  { value: '20', label: '20' },
  { value: '29', label: '29 (default)' },
  { value: '40', label: '40' },
  { value: '60', label: '60' },
];

const TURMA_OPTIONS = [
  { value: '2', label: '2 turmas' },
  { value: '3', label: '3 turmas (default)' },
  { value: '4', label: '4 turmas' },
  { value: '5', label: '5 turmas' },
  { value: '6', label: '6 turmas' },
];

const AULAS_OPTIONS = [
  { value: '4', label: '4 aulas/dia' },
  { value: '5', label: '5 aulas/dia (default)' },
  { value: '6', label: '6 aulas/dia' },
];

const DIAS_OPTIONS = [
  { value: '2', label: '2 dias (default)' },
  { value: '3', label: '3 dias' },
  { value: '5', label: '5 dias (sem.)' },
];

const MAT_OPTIONS = [
  { value: '5',  label: '5 matérias' },
  { value: '8',  label: '8 matérias' },
  { value: '10', label: '10 matérias' },
  { value: '12', label: '12 matérias' },
];

const POP_OPTIONS = [
  { value: '20',  label: '20' },
  { value: '50',  label: '50' },
  { value: '100', label: '100 (default)' },
  { value: '200', label: '200' },
];

const GERACOES_OPTIONS = [
  { value: '100',  label: '100' },
  { value: '300',  label: '300' },
  { value: '500',  label: '500' },
  { value: '1000', label: '1000' },
  { value: '2000', label: '2000' },
];

const PC_OPTIONS = [
  { value: '0.6', label: 'Pc 0.60' },
  { value: '0.7', label: 'Pc 0.70' },
  { value: '0.85', label: 'Pc 0.85' },
  { value: '0.95', label: 'Pc 0.95' },
];

const PM_OPTIONS = [
  { value: '0.02', label: 'Pm 0.02' },
  { value: '0.05', label: 'Pm 0.05' },
  { value: '0.10', label: 'Pm 0.10 (default)' },
  { value: '0.20', label: 'Pm 0.20' },
];

const ELITE_OPTIONS = [
  { value: '0', label: 'sem elite' },
  { value: '2', label: 'p = 2' },
  { value: '4', label: 'p = 4' },
  { value: '8', label: 'p = 8' },
];

// Presets de fitness — pedagógicos, ilustram como mudar os pesos altera
// completamente o comportamento do AG (Aula 12 "a criatividade da modelagem").
const FITNESS_PRESETS: Record<string, { bonus: number; choque: number; variedade: number }> = {
  equilibrado: { bonus: 3,  choque: 10, variedade: 1  },
  choque:      { bonus: 1,  choque: 50, variedade: 0  },
  encadeada:   { bonus: 20, choque: 5,  variedade: 0  },
  curriculo:   { bonus: 3,  choque: 10, variedade: 10 },
};

const FITNESS_PRESET_OPTIONS = [
  { value: 'equilibrado', label: 'Equilibrado (default)' },
  { value: 'choque',      label: 'Anti-conflito' },
  { value: 'encadeada',   label: 'Aulas encadeadas' },
  { value: 'curriculo',   label: 'Cobertura completa' },
  { value: 'custom',      label: 'Personalizado' },
];

const DIAS_NOMES = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'];

// paleta determinística por matéria — cores legíveis no tema dark.
const PALETA = [
  '#00ccff', '#ff00aa', '#00ff88', '#ffff00', '#ff8800',
  '#aa66ff', '#ff66aa', '#66ffcc', '#ffaa66', '#88ff00',
  '#0088ff', '#ff4488',
];

function corMateria(materiaId: number): string {
  return PALETA[materiaId % PALETA.length];
}

// Abreviações curtas pra caber na célula da matriz sem cortar acentos.
const MATERIA_ABREV: Record<string, string> = {
  'Matemática': 'Mat',
  'Português': 'Por',
  'História': 'Hist',
  'Geografia': 'Geo',
  'Biologia': 'Bio',
  'Física': 'Fis',
  'Química': 'Quim',
  'Inglês': 'Ing',
  'Educação Física': 'Educ',
  'Artes': 'Art',
  'Filosofia': 'Filo',
  'Sociologia': 'Soc',
  'Literatura': 'Lit',
  'Informática': 'Info',
  'Espanhol': 'Esp',
};

function abreviarMateria(nome: string): string {
  return MATERIA_ABREV[nome] ?? nome.slice(0, 4);
}

export default function HorarioView() {
  const { show } = useToast();

  // Config
  const [numProfs, setNumProfs] = useState('29');
  const [numTurmas, setNumTurmas] = useState('3');
  const [aulasPorDia, setAulasPorDia] = useState('5');
  const [diasDaSemana, setDiasDaSemana] = useState('2');
  const [numMaterias, setNumMaterias] = useState('10');
  const [popSize, setPopSize] = useState('100');
  const [maxGeracoes, setMaxGeracoes] = useState('300');
  const [probCruz, setProbCruz] = useState('0.85');
  const [probMut, setProbMut] = useState('0.10');
  const [elitismo, setElitismo] = useState('2');

  // Fitness — preset + 3 pesos editáveis em modo Custom.
  const [fitnessPreset, setFitnessPreset] = useState('equilibrado');
  const [bonusGeminada, setBonusGeminada] = useState(3);
  const [penChoque, setPenChoque] = useState(10);
  const [penSemVariedade, setPenSemVariedade] = useState(1);

  function applyPreset(key: string) {
    setFitnessPreset(key);
    if (key === 'custom') return;
    const p = FITNESS_PRESETS[key];
    if (p) {
      setBonusGeminada(p.bonus);
      setPenChoque(p.choque);
      setPenSemVariedade(p.variedade);
    }
  }

  // Training state
  const [training, setTraining] = useState(false);
  const [geracao, setGeracao] = useState<string>('—');
  const [melhorFit, setMelhorFit] = useState<string>('—');
  const [choques, setChoques] = useState<string>('—');
  const [bonus, setBonus] = useState<string>('—');
  const [statusColor, setStatusColor] = useState<string>('var(--on-surface)');

  // Dados pra renderizar a matriz
  const [melhorIndiv, setMelhorIndiv] = useState<HorarioIndividuo | null>(null);
  const [professores, setProfessores] = useState<HorarioProfessor[]>([]);

  // Histórico
  const [histMelhor, setHistMelhor] = useState<number[]>([]);
  const [histMedia, setHistMedia] = useState<number[]>([]);

  // Replay
  const histIndivRef = useRef<HorarioIndividuo[]>([]);
  const [maxGen, setMaxGen] = useState(0);
  const [displayGen, setDisplayGen] = useState(0);
  const [userScrub, setUserScrub] = useState(false);

  // Toggle de exibição do nome: false = código curto (P01..Pnn), true = nome brasileiro.
  const [nomesReais, setNomesReais] = useState(false);

  const closeSSE = useRef<(() => void) | null>(null);

  const slots = parseInt(diasDaSemana) * parseInt(aulasPorDia);
  const turmasNum = parseInt(numTurmas);

  async function handleTrain() {
    setTraining(true);
    setStatusColor('var(--cyan)');
    setHistMelhor([]);
    setHistMedia([]);
    setMelhorIndiv(null);
    histIndivRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);

    const cfg: HorarioConfig = {
      numProfessores: parseInt(numProfs),
      numTurmas: parseInt(numTurmas),
      aulasPorDia: parseInt(aulasPorDia),
      diasDaSemana: parseInt(diasDaSemana),
      numMaterias: parseInt(numMaterias),
      popSize: parseInt(popSize),
      maxGeracoes: parseInt(maxGeracoes),
      probCruzamento: parseFloat(probCruz),
      probMutacao: parseFloat(probMut),
      tamanhoTorneio: 4,
      elitismo: parseInt(elitismo),
      bonusGeminada: bonusGeminada,
      penChoque: penChoque,
      penSemVariedade: penSemVariedade,
    };

    try {
      await apiPost('/horario/config', cfg);
    } catch (e) {
      show('Erro ao configurar: ' + (e instanceof Error ? e.message : String(e)));
      setTraining(false);
      setStatusColor('var(--pink)');
      return;
    }

    closeSSE.current = apiSSE('/horario/train', {
      onMessage(data) {
        const step = data as HorarioStep;
        histIndivRef.current.push(step.melhorIndiv);
        setGeracao(step.geracao.toLocaleString());
        setMelhorFit(step.melhorFit.toFixed(1));
        setChoques(`${step.choques}`);
        setBonus(`${step.bonus}`);
        setMaxGen(step.geracao);
        if (!userScrub) {
          setDisplayGen(step.geracao);
          setMelhorIndiv(step.melhorIndiv);
        }
        setHistMelhor(prev => [...prev, step.melhorFit]);
        setHistMedia(prev => [...prev, step.mediaFit]);
      },
      onDone(data) {
        const r = data as HorarioResult;
        setProfessores(r.professores);
        setGeracao(r.geracoes.toLocaleString());
        setMelhorFit(r.melhorIndiv.fitness.toFixed(1));
        setChoques(`${r.melhorIndiv.choques}`);
        setBonus(`${r.melhorIndiv.bonus}`);
        setMelhorIndiv(r.melhorIndiv);
        setHistMelhor(r.histMelhor);
        setHistMedia(r.histMedia);
        if (!userScrub) {
          setDisplayGen(r.geracoes);
        }
        setStatusColor('var(--primary-glow)');
        setTraining(false);
        closeSSE.current = null;
        show('Horário evoluído — arraste a barra pra ver a evolução');
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
      await apiPost('/horario/reset');
    } catch {
      // ignore
    }
    setGeracao('—');
    setMelhorFit('—');
    setChoques('—');
    setBonus('—');
    setStatusColor('var(--on-surface)');
    setMelhorIndiv(null);
    setHistMelhor([]);
    setHistMedia([]);
    histIndivRef.current = [];
    setMaxGen(0);
    setDisplayGen(0);
    setUserScrub(false);
    setTraining(false);
    show('Horário resetado');
  }

  function handleSlider(gen: number) {
    setUserScrub(true);
    setDisplayGen(gen);
    const ind = histIndivRef.current[gen - 1];
    if (ind) setMelhorIndiv(ind);
  }

  // Lookup rápido prof → {nome, materia, cor}
  const profMap = useMemo(() => {
    const m = new Map<number, HorarioProfessor>();
    for (const p of professores) m.set(p.id, p);
    return m;
  }, [professores]);

  // Agrupa professores por matéria pra renderizar o catálogo em colunas
  // separadas por disciplina (mais fácil de ler com 30+ profs).
  const profsPorMateria = useMemo(() => {
    const grupos = new Map<number, HorarioProfessor[]>();
    for (const p of professores) {
      const arr = grupos.get(p.materia);
      if (arr) arr.push(p);
      else grupos.set(p.materia, [p]);
    }
    return Array.from(grupos.entries()).sort(([a], [b]) => a - b);
  }, [professores]);

  // Detecta choques por slot (mesmo prof em + de uma turma no mesmo slot)
  // para destacar células em vermelho na visualização.
  const choquesPorSlot = useMemo(() => {
    if (!melhorIndiv) return new Set<number>();
    const conflitos = new Set<number>();
    for (let s = 0; s < slots; s++) {
      const seen = new Map<number, number>();
      for (let t = 0; t < turmasNum; t++) {
        const p = melhorIndiv.matriz[s * turmasNum + t];
        seen.set(p, (seen.get(p) ?? 0) + 1);
      }
      for (let t = 0; t < turmasNum; t++) {
        const p = melhorIndiv.matriz[s * turmasNum + t];
        if ((seen.get(p) ?? 0) > 1) conflitos.add(s * turmasNum + t);
      }
    }
    return conflitos;
  }, [melhorIndiv, slots, turmasNum]);

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Horário <span>Escolar</span></div>
          <div className="page-sub">
            AG com cromossomo matricial · cruzamento por troca de linhas &mdash; Aula 12
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            className="btn"
            onClick={() => setNomesReais(v => !v)}
            style={{ fontSize: 11, padding: '6px 12px' }}
            title="Alternar entre código (P01) e nome brasileiro (Patrício)"
          >
            {nomesReais ? '⇋ NOMES' : '⇋ CÓDIGOS'}
          </button>
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

      {/* Config — linha 1: dimensões do problema */}
      <div className="grid-3" style={{ marginBottom: 12 }}>
        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Professores · Turmas</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={PROF_OPTIONS} value={numProfs} onChange={setNumProfs} style={{ flex: 1 }} />
            <Select options={TURMA_OPTIONS} value={numTurmas} onChange={setNumTurmas} style={{ flex: 1 }} />
          </div>
          <div style={{ marginTop: 10 }}>
            <Select
              label="Matérias distintas"
              options={MAT_OPTIONS}
              value={numMaterias}
              onChange={setNumMaterias}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <div className="imgreg-select-label">Aulas/dia · Dias</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Select options={AULAS_OPTIONS} value={aulasPorDia} onChange={setAulasPorDia} style={{ flex: 1 }} />
            <Select options={DIAS_OPTIONS} value={diasDaSemana} onChange={setDiasDaSemana} style={{ flex: 1 }} />
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)', fontFamily: 'JetBrains Mono' }}>
            Total de slots: <b style={{ color: 'var(--cyan)' }}>{slots}</b>
            <br />
            Tamanho do cromossomo: <b style={{ color: 'var(--cyan)' }}>{slots} × {turmasNum} = {slots * turmasNum}</b>
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
            <Select
              label="Gerações"
              options={GERACOES_OPTIONS}
              value={maxGeracoes}
              onChange={setMaxGeracoes}
              style={{ width: '100%' }}
            />
          </div>
        </Card>
      </div>

      {/* Config — linha 2: operadores do AG */}
      <div className="grid-3" style={{ marginBottom: 16 }}>
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
              label="Taxa de Mutação (por célula)"
              options={PM_OPTIONS}
              value={probMut}
              onChange={setProbMut}
              style={{ width: '100%' }}
            />
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Elitismo (p melhores intactos)"
            options={ELITE_OPTIONS}
            value={elitismo}
            onChange={setElitismo}
            style={{ width: '100%' }}
          />
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--muted)', fontFamily: 'JetBrains Mono', lineHeight: 1.6 }}>
            <div>Seleção: <span style={{ color: 'var(--cyan)' }}>torneio (k=4)</span></div>
            <div>Cruz.: <span style={{ color: 'var(--cyan)' }}>troca de linhas</span></div>
            <div>Mut.: <span style={{ color: 'var(--cyan)' }}>flip por célula</span></div>
          </div>
        </Card>

        <Card style={{ padding: '16px 20px' }}>
          <Select
            label="Fitness ativa"
            options={FITNESS_PRESET_OPTIONS}
            value={fitnessPreset}
            onChange={applyPreset}
            style={{ width: '100%' }}
          />

          {fitnessPreset === 'custom' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: 6,
              marginTop: 8,
            }}>
              <FitnessInput
                label="encad"
                sign="+"
                color="var(--green)"
                value={bonusGeminada}
                onChange={setBonusGeminada}
              />
              <FitnessInput
                label="choque"
                sign="−"
                color="var(--pink)"
                value={penChoque}
                onChange={setPenChoque}
              />
              <FitnessInput
                label="variedade"
                sign="−"
                color="var(--pink)"
                value={penSemVariedade}
                onChange={setPenSemVariedade}
              />
            </div>
          )}

          <div style={{
            fontSize: 11,
            lineHeight: 1.7,
            fontFamily: 'JetBrains Mono',
            color: 'var(--muted)',
            marginTop: 10,
          }}>
            <div>
              <span style={{ color: 'var(--green)' }}>+{bonusGeminada}</span>
              {' por aula encadeada (mesma matéria 2× consecutivas)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>−{penChoque}</span>
              {' por choque (prof em > 1 turma no mesmo slot)'}
            </div>
            <div>
              <span style={{ color: 'var(--pink)' }}>−{penSemVariedade}</span>
              {' por matéria não-coberta numa turma'}
            </div>
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
          title="Fitness"
          value={melhorFit}
          label={`encadeadas: ${bonus}`}
          valueStyle={{ color: statusColor }}
        />
        <MetricCard
          title="Choques"
          value={choques}
          label="conflitos no melhor"
          color="pink"
        />
      </div>

      {/* Matriz visual */}
      <Card title="Matriz de horário · melhor indivíduo" style={{ marginBottom: 16 }}>
        <div style={{ padding: '12px 16px', overflowX: 'auto' }}>
          {melhorIndiv ? (
            <HorarioMatrix
              indiv={melhorIndiv}
              profMap={profMap}
              slots={slots}
              turmas={turmasNum}
              aulasPorDia={parseInt(aulasPorDia)}
              choquesSet={choquesPorSlot}
              nomesReais={nomesReais}
            />
          ) : (
            <div style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'JetBrains Mono' }}>
              aguardando evolução&hellip;
            </div>
          )}
        </div>

        {maxGen > 0 && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '8px 12px',
            margin: '0 12px 12px',
            background: 'var(--surface-2)',
            borderRadius: 6,
          }}>
            <button
              className="btn btn-ghost"
              style={{ fontSize: 10, padding: '4px 10px' }}
              onClick={() => {
                setUserScrub(false);
                setDisplayGen(maxGen);
                const last = histIndivRef.current[maxGen - 1];
                if (last) setMelhorIndiv(last);
              }}
              disabled={training && !userScrub}
            >
              ⏭ latest
            </button>
            <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: 'var(--muted)', minWidth: 110 }}>
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
          </div>
        )}
      </Card>

      {/* Evolução */}
      <Card title="Evolução da fitness — melhor e médio" style={{ marginBottom: 16 }}>
        <GaEvoChart histMelhor={histMelhor} histMedia={histMedia} />
      </Card>

      {/* Catálogo de professores — agrupado por matéria */}
      {professores.length > 0 && (
        <Card title={`Catálogo de Professores (${professores.length})`} style={{ marginBottom: 16 }}>
          <div style={{
            padding: 12,
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: 10,
            fontSize: 11,
            fontFamily: 'JetBrains Mono',
          }}>
            {profsPorMateria.map(([materiaId, profs]) => {
              const cor = corMateria(materiaId);
              return (
                <div
                  key={materiaId}
                  style={{
                    background: 'var(--surface-2)',
                    borderLeft: `4px solid ${cor}`,
                    borderRadius: 4,
                    padding: '8px 10px',
                  }}
                >
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    marginBottom: 6,
                    paddingBottom: 6,
                    borderBottom: '1px solid var(--surface-top)',
                  }}>
                    <span style={{
                      display: 'inline-block',
                      width: 10, height: 10,
                      background: cor,
                      borderRadius: 2,
                    }} />
                    <span style={{ color: cor, fontWeight: 600 }}>{profs[0].materiaNome}</span>
                    <span style={{ color: 'var(--muted)', fontSize: 10, marginLeft: 'auto' }}>
                      {profs.length}
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {profs.map(p => (
                      <div
                        key={p.id}
                        style={{ color: 'var(--on-surface)' }}
                        title={`${p.nome} · ${p.nomeReal}`}
                      >
                        {nomesReais ? p.nomeReal : p.nome}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Educational */}
      <Card title="Como funciona — Aula 12">
        <div style={{ padding: 12, fontSize: 14, color: 'var(--muted)', lineHeight: 1.7 }}>
          <b>Cromossomo matricial.</b> Cada indivíduo é uma matriz <code>slot × turma</code>,
          onde cada célula guarda o ID de um professor. Diferente da aula 11 (bits) e da aula 13
          (permutação do TSP), aqui o cromossomo é um <i>grid</i> bidimensional — exatamente o
          ponto da Aula 12: "outros tipos de codificação".
          <br /><br />
          <b>Cruzamento por troca de linhas.</b> Para cada <i>linha</i> da matriz (um slot de
          horário inteiro), o filho sorteia de qual pai herda — preservando consistência dentro
          do mesmo horário. É o operador específico pra esse encoding.
          <br /><br />
          <b>Fitness composta (a "criatividade" da modelagem):</b>
          <ul style={{ marginLeft: 18 }}>
            <li><b>+3 por aula encadeada:</b> bônus quando a mesma matéria aparece em 2 horários consecutivos do mesmo dia numa turma.</li>
            <li><b>−10 por choque:</b> mesmo professor em &gt; 1 turma no mesmo slot — fisicamente impossível.</li>
            <li><b>−1 por matéria faltante:</b> cada matéria do catálogo deveria aparecer ao menos 1× em cada turma.</li>
          </ul>
          <b>Como ler a matriz:</b> linhas são slots de horário (agrupados por dia), colunas são turmas,
          células mostram o professor (P01..Pnn) com cor pela matéria. Choques ficam destacados em vermelho.
        </div>
      </Card>
    </div>
  );
}

// =============================================================================
// Input compacto pra editar peso de um termo da fitness (modo Custom)
// =============================================================================

interface FitnessInputProps {
  label: string;
  sign: string;
  color: string;
  value: number;
  onChange: (v: number) => void;
}

function FitnessInput({ label, sign, color, value, onChange }: FitnessInputProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 2,
      alignItems: 'center',
    }}>
      <span style={{ fontSize: 9, color: 'var(--muted)', fontFamily: 'JetBrains Mono' }}>
        {sign} {label}
      </span>
      <input
        type="number"
        min={0}
        step={1}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          onChange(Number.isFinite(v) ? Math.max(0, v) : 0);
        }}
        style={{
          width: '100%',
          padding: '3px 6px',
          fontSize: 12,
          fontFamily: 'JetBrains Mono',
          background: 'var(--surface-2)',
          border: `1px solid ${color}`,
          color: color,
          borderRadius: 3,
          textAlign: 'center',
        }}
      />
    </div>
  );
}

// =============================================================================
// Componente da matriz visual
// =============================================================================

interface MatrixProps {
  indiv: HorarioIndividuo;
  profMap: Map<number, HorarioProfessor>;
  slots: number;
  turmas: number;
  aulasPorDia: number;
  choquesSet: Set<number>;
  nomesReais: boolean;
}

function HorarioMatrix({ indiv, profMap, slots, turmas, aulasPorDia, choquesSet, nomesReais }: MatrixProps) {
  const headerStyle: React.CSSProperties = {
    padding: '6px 10px',
    background: 'var(--surface-2)',
    fontSize: 11,
    fontFamily: 'JetBrains Mono',
    color: 'var(--muted)',
    textAlign: 'center' as const,
    borderBottom: '1px solid var(--surface-top)',
  };
  const cellBase: React.CSSProperties = {
    padding: '8px 6px',
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    textAlign: 'center' as const,
    borderRight: '1px solid var(--surface-2)',
    borderBottom: '1px solid var(--surface-2)',
    minWidth: 80,
  };

  return (
    <table style={{ borderCollapse: 'collapse', width: '100%' }}>
      <thead>
        <tr>
          <th style={{ ...headerStyle, width: 60 }}>Dia</th>
          <th style={{ ...headerStyle, width: 40 }}>#</th>
          {Array.from({ length: turmas }).map((_, t) => (
            <th key={t} style={headerStyle}>Turma {String(t + 1).padStart(2, '0')}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: slots }).map((_, s) => {
          const dia = Math.floor(s / aulasPorDia);
          const aulaNoDia = (s % aulasPorDia) + 1;
          const primeiroDoDia = s % aulasPorDia === 0;
          return (
            <tr key={s} style={primeiroDoDia ? { borderTop: '2px solid var(--surface-top)' } : undefined}>
              <td style={{ ...cellBase, color: 'var(--muted)', background: 'var(--surface-2)' }}>
                {primeiroDoDia ? (DIAS_NOMES[dia] ?? `D${dia + 1}`) : ''}
              </td>
              <td style={{ ...cellBase, color: 'var(--muted)', background: 'var(--surface-2)' }}>
                {aulaNoDia}
              </td>
              {Array.from({ length: turmas }).map((_, t) => {
                const flatIdx = s * turmas + t;
                const profId = indiv.matriz[flatIdx];
                const prof = profMap.get(profId);
                const choque = choquesSet.has(flatIdx);
                const cor = prof ? corMateria(prof.materia) : '#444';
                return (
                  <td
                    key={t}
                    style={{
                      ...cellBase,
                      background: choque ? 'rgba(255, 0, 0, 0.25)' : 'transparent',
                      color: 'var(--on-surface)',
                      borderLeft: `3px solid ${cor}`,
                    }}
                    title={prof ? `${prof.nome} · ${prof.nomeReal} · ${prof.materiaNome}${choque ? ' · ⚠ CHOQUE' : ''}` : `prof ${profId}`}
                  >
                    <div style={{ color: cor, fontWeight: 600, fontSize: nomesReais ? 9 : 11 }}>
                      {prof ? (nomesReais ? prof.nomeReal : prof.nome) : `P${profId}`}
                    </div>
                    <div style={{ fontSize: 9, color: 'var(--muted)' }}>{prof ? abreviarMateria(prof.materiaNome) : '—'}</div>
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
