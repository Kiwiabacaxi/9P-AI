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

// Time Series Comparison
export interface TsCompareModelResult {
  modelo: string;
  result: TsResult;
  cor: string;
  erro?: string;
}

export interface TsCompareProgress {
  modelo: string;
  ciclo: number;
  mseTreino: number;
  mseValid: number;
}

export interface TsAvailableModel {
  nome: string;
  categoria: string;
  cor: string;
  needsPython: boolean;
}

export type ViewId =
  | 'hebb' | 'perceptron' | 'madaline'
  | 'mlp' | 'letras' | 'mlpfunc' | 'mlport'
  | 'imgreg' | 'imgreg-goroutines' | 'imgreg-matrix' | 'imgreg-minibatch' | 'imgreg-bench'
  | 'cnn' | 'timeseries' | 'ts-compare'
  | 'about';
