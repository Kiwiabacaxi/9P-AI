// Status
export interface AppStatus {
  mlpTrained: boolean;
  letrasTrained: boolean;
  ltrTraining: boolean;
  hebbCount: number;
  percPortaCount: number;
  percLetrasDone: boolean;
  madTrained: boolean;
  madTraining: boolean;
  imgregTrained: boolean;
  imgregTraining: boolean;
  igorTrained: boolean;
  igorTraining: boolean;
  imatTrained: boolean;
  imatTraining: boolean;
  imbTrained: boolean;
  imbTraining: boolean;
  mlpFuncTrained: boolean;
  mlpFuncTraining: boolean;
  ortTrained: boolean;
  ortTraining: boolean;
  cnnTrained: boolean;
  cnnTraining: boolean;
  tsTrained: boolean;
  tsTraining: boolean;
  gaTrained: boolean;
  gaTraining: boolean;
  ga2Trained: boolean;
  ga2Training: boolean;
  tspTrained: boolean;
  tspTraining: boolean;
}

// Hebb
export interface HebbResult {
  porta: string;
  pesos: number[];
  bias: number;
  tabela: { x1: number; x2: number; target: number; saida: number; correto: boolean }[];
  convergiu: boolean;
}

// Perceptron Portas
export interface PercPortasResult {
  porta: string;
  pesos: number[];
  bias: number;
  ciclos: number;
  convergiu: boolean;
  tabela: { x1: number; x2: number; target: number; saida: number; correto: boolean }[];
}

// Perceptron Letras
export interface PercLetrasResult {
  convergiu: boolean;
  ciclos: number;
  acuracia: number;
  resultados: { nome: string; correto: boolean }[];
}

export interface PercLetrasDataset {
  letras: { nome: string; grade: number[] }[];
}

// MADALINE
export interface MadStep {
  ciclo: number;
  letraIdx: number;
  letra: string;
  erroTotal: number;
}

export interface MadResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  acertos: number;
  total: number;
  acuracia: number;
}

export interface MadClassifyResp {
  letraIdx: number;
  letra: string;
  confidencias: number[];
  top5: { letra: string; confidencia: number; idx: number }[];
}

export interface MadDataset {
  letras: { nome: string; grade: number[] }[];
}

// MLP Desafio
export interface MlpResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  padroes: { entrada: number[]; target: number[]; saida: number[] }[];
  steps: MlpStep[];
}

export interface MlpStep {
  ciclo: number;
  padrao: number;
  entrada: number[];
  target: number[];
  zIn: number[];
  z: number[];
  yIn: number[];
  y: number[];
  deltaK: number[];
  deltaJ: number[];
  wAntes: number[][];
  wDepois: number[][];
  vAntes: number[][];
  vDepois: number[][];
  erroTotal: number;
}

// MLP Letras
export interface LetrasResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  acertos: number;
  total: number;
  acuracia: number;
}

export interface LetrasClassifyResp {
  letraIdx: number;
  letra: string;
  confidencias: number[];
  top5: { letra: string; confidencia: number; idx: number }[];
}

export interface LetrasDataset {
  letras: { nome: string; grade: number[] }[];
}

// MLP Funcoes
export interface FuncConfig {
  funcao: string;
  hiddenLayers?: number[];
  nHid?: number;
  alfa: number;
  maxCiclo: number;
  ativacao: string;
}

export interface FuncPoint {
  x: number;
  y: number;
  yPred: number;
}

export interface FuncStep {
  ciclo: number;
  erroTotal: number;
  pontos: FuncPoint[];
  activeLayer: number;
}

export interface FuncResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  pontos: FuncPoint[];
  funcao: string;
}

// MLP Ortogonal
export interface OrtConfig {
  nHid: number;
  alfa: number;
  maxCiclo: number;
}

export interface OrtStep {
  ciclo: number;
  letraIdx: number;
  letra: string;
  erroTotal: number;
  activeLayer: number;
}

export interface OrtResult {
  convergiu: boolean;
  ciclos: number;
  erroFinal: number;
  erroHistorico: number[];
  acertos: number;
  total: number;
  acuracia: number;
  vetores: number[][];
}

export interface OrtClassifyResp {
  letraIdx: number;
  letra: string;
  distancias: number[];
  top5: { letra: string; distancia: number; idx: number }[];
  saidaRede: number[];
}

export interface OrtDatasetInfo {
  letras: { nome: string; grade: number[]; vetor: number[] }[];
  vetores: number[][];
}

// Image Regression
export interface ImgregConfig {
  hiddenLayers: number;
  neuronsPerLayer: number;
  learningRate: number;
  imagem: string;
  maxEpocas: number;
  batchSize?: number;
  numWorkers?: number;
}

export interface ImgregStep {
  epoca: number;
  maxEpocas: number;
  loss: number;
  outputPixels: [number, number, number][];
  activeLayer: number;
  done?: boolean;
  convergiu?: boolean;
  lossHistorico?: number[];
  elapsedMs?: number;
  epochMs?: number;
}

export interface BenchConfig {
  hiddenLayers: number;
  neuronsPerLayer: number;
  maxEpocas: number;
  imagem: string;
}

export interface BenchStep {
  backend: string;
  step: ImgregStep;
}

export interface BenchResult {
  metodo: string;
  tempoMs: number;
  loss: number;
  convergiu: boolean;
  epocas: number;
}

// CNN (EMNIST Letters)
export interface CnnConfig {
  alfa: number;
  maxEpocas: number;
  batchSize: number;
  trainLimit: number;
}

export interface CnnStep {
  epoca: number;
  batch: number;
  totalBatch: number;
  loss: number;
  acuracia: number;
}

export interface CnnResult {
  epocas: number;
  lossFinal: number;
  lossHistorico: number[];
  acuracia: number;
  acuraciaTest: number;
  tempoMs: number;
}

export interface CnnClassifyResp {
  letraIdx: number;
  letra: string;
  scores: number[];
  top5: { letra: string; score: number; idx: number }[];
}

export interface CnnVisualizeResp {
  input: number[];
  conv1Maps: number[][][];
  pool1Maps: number[][][];
  conv2Maps: number[][][];
  pool2Maps: number[][][];
  filters1: number[][][][];
  filters2: number[][][][];
  probs: number[];
  letraIdx: number;
  letra: string;
  top5: { letra: string; score: number; idx: number }[];
}

export interface CnnModelMeta {
  id: string;
  nome: string;
  criadoEm: string;
  epocas: number;
  trainLimit: number;
  acuracia: number;
  acuraciaTest: number;
  lossFinal: number;
}

// Time Series (Previsão de ações)
export interface TsStockData {
  ticker: string;
  dates: string[];
  close: number[];
  open: number[];
  high: number[];
  low: number[];
  volume: number[];
}

export interface TsStep {
  ciclo: number;
  mseTreino: number;
  mseValid: number;
}

export interface TsPoint {
  data: string;
  preco: number;
  predito: number;
}

export interface TsForecastPoint {
  dia: number;
  predito: number;
  upper: number;
  lower: number;
}

export interface TsResult {
  ciclos: number;
  mseFinal: number;
  rmseFinal: number;
  maeFinal: number;
  mseHistorico: number[];
  pontos: TsPoint[];
  pontosValid: TsPoint[];
  predicaoAmanha: number;
  forecast: TsForecastPoint[];
  ticker: string;
  tempoMs: number;
}

export interface TsModelMeta {
  id: string;
  nome: string;
  criadoEm: string;
  ticker: string;
  windowSize: number;
  hiddenSize: number;
  ciclos: number;
  rmseFinal: number;
  maeFinal: number;
  predicaoAmanha: number;
}

// Algoritmo Genético (Aula 10)
export interface GAConfig {
  bits: number;
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  seed?: number;
}

export interface GAIndividuo {
  bits: number[];
  dec: number;
  x: number;
  fitness: number;
  fx: number;
}

export interface GAStep {
  geracao: number;
  melhorX: number;
  melhorFx: number;
  mediaFx: number;
  piorFx: number;
  populacao: GAIndividuo[];
  melhorIndiv: GAIndividuo;
}

export interface GAResult {
  geracoes: number;
  melhorX: number;
  melhorFx: number;
  melhorIndiv: GAIndividuo;
  histMelhorFx: number[];
  histMediaFx: number[];
  bits: number;
  popSize: number;
  probCruzamento: number;
  probMutacao: number;
}

// Algoritmo Genético v2 (Aula 11)
export type GA2Selecao = 'roleta' | 'torneio';

export interface GA2Config {
  bits: number;
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  selecao: GA2Selecao;
  tamanhoTorneio: number;
  pontosCorte: 1 | 2;
  elitismo: number;
  dominioMin: number;
  dominioMax: number;
  seed?: number;
}

export interface GA2Step {
  geracao: number;
  melhorX: number;
  melhorFx: number;
  mediaFx: number;
  piorFx: number;
  diversidade: number;
  populacao: GAIndividuo[];
  melhorIndiv: GAIndividuo;
}

export interface GA2Result {
  geracoes: number;
  melhorX: number;
  melhorFx: number;
  melhorIndiv: GAIndividuo;
  histMelhorFx: number[];
  histMediaFx: number[];
  histDiversidade: number[];
  bits: number;
  popSize: number;
  probCruzamento: number;
  probMutacao: number;
  selecao: string;
  tamanhoTorneio: number;
  pontosCorte: number;
  elitismo: number;
  dominioMin: number;
  dominioMax: number;
}

// TSP — Caixeiro Viajante (Aula 12)
export interface TspCidade {
  id: number;
  nome: string;
  uf?: string;
  lat: number;
  lng: number;
}

// Preset temático: cidades + narrativa + parâmetros sugeridos.
export interface TspPreset {
  id: string;
  nome: string;
  descricao: string;
  narrativa: string;
  origem: string;
  cidades: TspCidade[];
  lambdaSugerido: number;
  modoSugerido: TspDistMode;
  fitnessNota: string;
  lastVisit: number;
  lastVisitNome?: string;
  gammaSugerido: number;
  muOvertimeSugerido: number;
}

// Versão resumida pra dropdown (sem cidades).
export interface TspPresetMeta {
  id: string;
  nome: string;
  descricao: string;
  origem: string;
  lambdaSugerido: number;
  modoSugerido: TspDistMode;
  numCidades: number;
}

export type TspSelecao = 'roleta' | 'torneio';
export type TspCrossover = 'ox' | 'pmx';
export type TspMutacao = 'swap' | 'inversao';
export type TspDistMode = 'euclidiana' | 'haversine' | 'osrm';

export interface TspConfig {
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  selecao: TspSelecao;
  tamanhoTorneio: number;
  cruzamento: TspCrossover;
  mutacao: TspMutacao;
  elitismo: number;
  lambdaMaxLeg: number;
  lastVisit: number;
  gamma: number;          // peso do tempo (km/h equivalente)
  jornadaMaxSec: number;  // jornada máxima em segundos (default 36000 = 10h)
  muOvertime: number;     // coef. da penalidade quadrática de overtime
  seed?: number;
}

export interface TspStep {
  geracao: number;
  melhorTour: number[];
  melhorDist: number;
  melhorMaxLeg: number;
  melhorTempoSec: number;
  melhorCusto: number;
  mediaDist: number;
  piorDist: number;
  diversidade: number;
  melhorGlobal: number[];
  melhorGlobalDist: number;
}

export interface TspResult {
  geracoes: number;
  melhorTour: number[];
  melhorDist: number;
  melhorMaxLeg: number;
  melhorTempoSec: number;
  melhorCusto: number;
  histMelhor: number[];
  histMedia: number[];
  histDiversidade: number[];
  cfg: TspConfig;
}

// Uma "perna" da rota — entre duas cidades consecutivas no tour.
// Permite animar o truck seguindo as curvas reais dentro de cada perna.
export interface TspLegGeometry {
  polyline: [number, number][]; // [[lat, lng], ...]
  distancia: number;            // km
  duracao: number;              // segundos
  deId: number;                 // id da cidade no início da perna
  paraId: number;               // id da cidade no fim da perna
}

// Resultado de um baseline determinístico (Nearest Neighbor / 2-opt).
export interface TspBaselineResult {
  algoritmo: string;        // "nn" | "2opt" | "nn+2opt"
  tour: number[];
  distancia: number;        // km (ou graus se modo euclidiana)
  maxLeg: number;
  tempoMs: number;
  tempoUs: number;
}

// Geometria curvada da rota retornada pelo OSRM (estradas reais).
export interface TspRouteGeometry {
  polyline: [number, number][];  // [[lat, lng], ...] tour fechado completo
  legs: TspLegGeometry[];        // breakdown por perna
  distancia: number;             // km totais
  duracao: number;               // segundos totais
}

// TSP Multi-populacional — modelo de ilhas (Aula 14 / Trabalho 12)
export interface TspMultiConfig {
  numIlhas: number;
  tamIlha: number;
  maxGeracoes: number;
  intervaloMigracao: number;
  numMigrantes: number;
  topologia: string;          // "anel"
  compararPopUnica: boolean;
  seed?: number;
  ga: TspConfig;
}

export interface TspIlhaStep {
  ilha: number;
  melhorTour: number[];
  melhorDist: number;
  melhorCusto: number;
  mediaDist: number;
  diversidade: number;
}

export interface TspMigracao {
  de: number;
  para: number;
  migranteTour?: number[];
}

export interface TspMultiStep {
  geracao: number;
  ilhas: TspIlhaStep[];
  melhorGlobalTour: number[];
  melhorGlobalDist: number;
  ilhaVencedora: number;
  geracoesSemMelhora: number;
  diversidadeGlobal: number;
  migrou: boolean;
  migracoes?: TspMigracao[];
  refUnicaDist?: number;
  refUnicaDiv?: number;
}

export interface TspMultiResult {
  geracoes: number;
  melhorGlobalTour: number[];
  melhorGlobalDist: number;
  ilhaVencedora: number;
  histGlobal: number[];
  histIlhas: number[][];
  histDiversidade: number[];
  geracoesMigracao: number[];
  histRefUnica?: number[];
  histRefUnicaDiv?: number[];
  melhorRefUnicaDist?: number;
  cfg: TspMultiConfig;
}

// AG Rastrigin — cromossomos reais (Aula 15 / Trabalho 13)
export type RastSelecao = 'roleta' | 'torneio';
export type RastCruzamento = 'radcliff' | 'wright';

export interface RastConfig {
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  selecao: RastSelecao;
  tamanhoTorneio: number;
  cruzamento: RastCruzamento;
  elitismo: number;
  dominioMin: number;
  dominioMax: number;
  seed?: number;
}

export interface RastIndividuo {
  x: number[];        // [x, y, z]
  fx: number;         // valor de Rastrigin(x)
  fitness?: number;
}

export interface RastStep {
  geracao: number;
  melhorFx: number;
  melhorX: number[];
  mediaFx: number;
  piorFx: number;
  diversidade: number;
  populacao: RastIndividuo[];
  melhorGlobalFx: number;
  melhorGlobalX: number[];
}

export interface RastResult {
  geracoes: number;
  melhorFx: number;
  melhorX: number[];
  histMelhor: number[];
  histMedia: number[];
  histDiversidade: number[];
  cfg: RastConfig;
}

// AG + RNA — busca de arquitetura de MLP (Trabalho 15 / Aula 20)
export interface RnaGaConfig {
  popSize: number;
  maxGeracoes: number;
  probMutacao: number;
  tetoEpocas: number;
  seed?: number;
}

export interface RnaGaIndividuo {
  genes: number[];      // 6 genes
  string: string;       // codificação "8 | 3 | 0.01 | 500 | online | normaliza"
  mse: number;
  neuronios: number;
  camadas: number;
  online: boolean;
  normaliza: boolean;
}

export interface RnaGaStep {
  geracao: number;
  melhorMse: number;
  melhorGlobalMse: number;
  mediaMse: number;
  melhorCromossomo: RnaGaIndividuo;
  populacao: RnaGaIndividuo[];
  gradeMse: number[][];   // [neuronios-2][camadas-2], -1 = não visitada
}

export interface RnaGaResult {
  geracoes: number;
  melhorCromossomo: { genes: number[] };
  melhorView: RnaGaIndividuo;
  melhorMse: number;
  histMelhor: number[];
  histMedia: number[];
  cfg: RnaGaConfig;
}

export interface RnaGaBenchModo {
  ordem: number;     // 0=ingênuo … 3=atual
  nome: string;
  ms: number;
  melhorMse: number;
  cacheHits: number;
  workers: number;
}

export interface RnaGaBenchResult {
  preset: string;
  modos: RnaGaBenchModo[];
  numCpu: number;
  speedupTotal: number;
  mesmoMse: boolean;
  maxDiffMse: number;
  benchCfg: RnaGaConfig;
  fullCfg: RnaGaConfig;
  fullIngenuoMs: number;
  fullOtimizadoMs: number;
  timestampUnix: number;
}

export interface RnaGaBenchSaved {
  nome: string;
  preset: string;
  speedupTotal: number;
  mesmoMse: boolean;
  numCpu: number;
  popSize: number;
  maxGeracoes: number;
  tetoEpocas: number;
  timestampUnix: number;
}

// AG com Ranking — TSP (Trabalho 14 / Aulas 13 + 16)
export type TspRankSelecao = 'rankingLinear' | 'rankingExp' | 'torneio' | 'roleta';
export type TspRankCruzamento = 'ox' | 'pmx';
export type TspRankMutacao = 'swap' | 'inversao';

export interface TspRankConfig {
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  selecao: TspRankSelecao;
  tamanhoTorneio: number;
  etaMax: number;       // ranking linear: pressão máxima (η_min = 2 − η_max)
  cExp: number;         // ranking exponencial: base c > 1
  cruzamento: TspRankCruzamento;
  mutacao: TspRankMutacao;
  elitismo: number;
  seed?: number;
}

// Resposta de /tspranking/cidades — cenário fixo do Triângulo Mineiro.
export interface TspRankMapa {
  cidades: TspCidade[];
  matriz: number[][];
  fonte: boolean[][];   // true = veio da tabela da Aula 13; false = preenchido (Haversine·fator)
  fator: number;        // fator de calibração estrada/reta dos pares preenchidos
}

export interface TspRankStep {
  geracao: number;
  melhorTour: number[];
  melhorDist: number;
  mediaDist: number;
  piorDist: number;
  diversidade: number;
  melhorGlobalTour: number[];
  melhorGlobalDist: number;
  popDist: number[];    // distâncias da população ordenadas asc (rank 1..N)
}

export interface TspRankResult {
  geracoes: number;
  melhorTour: number[];
  melhorDist: number;
  histMelhor: number[];
  histMedia: number[];
  histDiversidade: number[];
  cfg: TspRankConfig;
}

export type ViewId =
  | 'hebb' | 'perceptron' | 'madaline'
  | 'mlp' | 'letras' | 'mlpfunc' | 'mlport'
  | 'imgreg' | 'imgreg-goroutines' | 'imgreg-matrix' | 'imgreg-minibatch' | 'imgreg-bench'
  | 'cnn' | 'timeseries'
  | 'genetico' | 'genetico2' | 'horario' | 'tsp' | 'tsp-compare' | 'tsp-multi'
  | 'rastrigin' | 'rna-ga' | 'tsp-ranking'
  | 'about';

// Horário Escolar (Aula 12 — cromossomo matricial)
export interface HorarioProfessor {
  id: number;
  nome: string;       // código curto P01..Pnn
  nomeReal: string;   // nome brasileiro
  materia: number;
  materiaNome: string;
}

export interface HorarioConfig {
  numProfessores: number;
  numTurmas: number;
  aulasPorDia: number;
  diasDaSemana: number;
  numMaterias: number;
  popSize: number;
  maxGeracoes: number;
  probCruzamento: number;
  probMutacao: number;
  tamanhoTorneio: number;
  elitismo: number;
  bonusGeminada: number;
  penChoque: number;
  penSemVariedade: number;
  seed?: number;
}

export interface HorarioIndividuo {
  matriz: number[];   // flat slot-major: M[slot*numTurmas + turma]
  fitness: number;
  choques: number;
  bonus: number;
  faltando: number;
}

export interface HorarioStep {
  geracao: number;
  melhorFit: number;
  melhorIndiv: HorarioIndividuo;
  mediaFit: number;
  choques: number;
  bonus: number;
  faltando: number;
}

export interface HorarioResult {
  geracoes: number;
  melhorIndiv: HorarioIndividuo;
  histMelhor: number[];
  histMedia: number[];
  histChoques: number[];
  histBonus: number[];
  professores: HorarioProfessor[];
  cfg: HorarioConfig;
}
