# Design: Image Regression Otimizada (3 Backends + Benchmark)

**Data:** 2026-03-24
**Status:** Aprovado

---

## Contexto

O pacote `web/server/imgreg` implementa treinamento de MLP para regressão de imagem (256 pixels 16×16, coordenadas → RGB). O loop de treinamento é sequencial: para cada época, processa pixel a pixel com forward + backward + update de pesos. O objetivo é criar 3 variantes otimizadas e um painel de benchmark comparativo, mantendo o backend original intacto.

---

## Arquitetura

### Novos packages Go

Quatro packages independentes em `web/server/`, cada um com seu próprio arquivo `.go`:

```
web/server/
├── imgreg/              ← existente, não muda
├── imgreg_goroutines/   ← novo
├── imgreg_matrix/       ← novo
├── imgreg_minibatch/    ← novo
└── imgreg_bench/        ← novo (orquestrador do benchmark)
```

Cada package expõe:
- Tipos `Config` e `Step` próprios
- Função `Treinar(ctx context.Context, cfg Config, ch chan<- Step) Net`

### Contratos de tipos

**Config base** (todos os backends):
```go
type Config struct {
    HiddenLayers    int     `json:"hiddenLayers"`
    NeuronsPerLayer int     `json:"neuronsPerLayer"`
    LearningRate    float64 `json:"learningRate"`
    Imagem          string  `json:"imagem"`
    MaxEpocas       int     `json:"maxEpocas"`
}
```

**Config extra em `imgreg_minibatch`:**
```go
BatchSize  int `json:"batchSize"`   // 8, 16, 32, 64
NumWorkers int `json:"numWorkers"`  // 2, 4, 8, runtime.NumCPU()
```

**Step** (todos os backends adicionam campos de timing):
```go
type Step struct {
    // campos existentes do imgreg original
    Epoca        int          `json:"epoca"`
    MaxEpocas    int          `json:"maxEpocas"`
    Loss         float64      `json:"loss"`
    OutputPixels [][3]float64 `json:"outputPixels"`
    ActiveLayer  int          `json:"activeLayer"`
    Done         bool         `json:"done"`
    Convergiu    bool         `json:"convergiu"`
    LossHistorico []float64   `json:"lossHistorico"`
    // campos novos de timing
    ElapsedMs    int64        `json:"elapsedMs"`   // ms desde início do treino
    EpochMs      int64        `json:"epochMs"`     // duração da última época
}
```

---

## Backends em Detalhe

### `imgreg_goroutines`

- Paralelismo no forward+backward por pixel dentro de cada época
- Usa `sync.WaitGroup` + canal de gradientes para coletar resultados
- Acumula gradientes de todos os 256 pixels, faz um único update ao final da época (batch gradient descent)
- Diferença de comportamento vs original: SGD → batch GD (gradiente mais estável, convergência diferente)
- Sem dependências externas

### `imgreg_matrix`

- Reescreve forward/backward usando `gonum.org/v1/gonum/mat`
- Cada camada é uma multiplicação matricial: `Z = A * W + B` vetorizado para todos os 256 pixels simultaneamente
- Processa a época inteira em uma única passagem matricial, sem loop por pixel
- Dependência: `gonum.org/v1/gonum/mat`
- Potencialmente mais rápido para redes com muitos neurônios (BLAS por baixo)

### `imgreg_minibatch`

- Divide os 256 pixels em batches de `BatchSize` pixels
- Pool de `NumWorkers` goroutines, uma por batch
- Acumula gradientes por batch, atualiza pesos após cada batch
- Meio-termo entre SGD (update por pixel) e batch GD (update por época)
- Config extra: `BatchSize` (8/16/32/64) e `NumWorkers` (2/4/8/NumCPU)

---

## Endpoints da API

### Por backend (padrão imgreg existente)

```
POST /api/imgreg-goroutines/config
GET  /api/imgreg-goroutines/train    (SSE)
POST /api/imgreg-goroutines/reset
GET  /api/imgreg-goroutines/status

POST /api/imgreg-matrix/config
GET  /api/imgreg-matrix/train        (SSE)
POST /api/imgreg-matrix/reset
GET  /api/imgreg-matrix/status

POST /api/imgreg-minibatch/config
GET  /api/imgreg-minibatch/train     (SSE)
POST /api/imgreg-minibatch/reset
GET  /api/imgreg-minibatch/status
```

### Benchmark

```
POST /api/imgreg-bench/config
GET  /api/imgreg-bench/run           (SSE)
```

**Benchmark config:**
```json
{
  "hiddenLayers": 3,
  "neuronsPerLayer": 32,
  "learningRate": 0.01,
  "imagem": "coracao",
  "maxEpocas": 500,
  "batchSize": 32,
  "numWorkers": 4,
  "parallel": true
}
```

**Benchmark SSE step:** inclui campo `backend: "goroutines"|"matrix"|"minibatch"` para o frontend identificar qual painel atualizar. Steps `done: true` por backend contêm métricas finais.

---

## Frontend

### Sidebar — accordion collapsible

```
▼ IMG REGRESSION
   ├─ Standard        (view existente, sem mudanças)
   ├─ Goroutines
   ├─ Matrix
   ├─ Mini-batch
   └─ Benchmark
```

O item pai `IMG REGRESSION` é clicável para expandir/colapsar o grupo. Cada filho abre sua view.

### Views dos backends (Goroutines / Matrix / Mini-batch)

Idênticas ao painel `Standard` existente:
- Controles: imagem, camadas ocultas, neurônios, lr, épocas
- `Mini-batch` adiciona: `Batch Size` e `Workers`
- Stat-bar em tempo real: `época`, `loss`, `ms/época`, `throughput px/s`
- Visualização: grid target + viz de rede + grid output + gráfico de loss

### View Benchmark

- Toggle `PARALELO / SEQUENCIAL`
- Config compartilhada: imagem, camadas, neurônios, lr, épocas, batch size, workers
- Botão `RODAR BENCHMARK`
- Em modo **paralelo**: 3 colunas side-by-side, cada uma com grid output + stat em tempo real via 3 SSE connections simultâneas
- Em modo **sequencial**: roda um por vez, exibe resultado ao terminar cada um
- Ao final: cards comparativos com `tempo total (ms)`, `ms/época médio`, `throughput (px/s)`, `loss final`
- Gráfico de barras comparativo entre os 3 backends

---

## Observações

- O package `imgreg` original **não é modificado**
- Estado global em `main.go` é replicado para cada backend (`imgreg_goroutinesRede`, `imgreg_matrixRede`, etc.)
- O `go.mod` precisará adicionar `gonum.org/v1/gonum` como dependência
- Os campos de timing `ElapsedMs` e `EpochMs` são medidos com `time.Since` dentro de cada função `Treinar`
