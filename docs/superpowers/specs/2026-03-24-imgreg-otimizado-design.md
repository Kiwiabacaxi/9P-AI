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
- Tipos `Config`, `Step` e `Net` próprios (cada package é autossuficiente, sem import entre eles)
- Função `Treinar(ctx context.Context, cfg Config, ch chan<- Step) Net`

### Contratos de tipos

**Config base** (todos os backends têm estes campos):
```go
type Config struct {
    HiddenLayers    int     `json:"hiddenLayers"`
    NeuronsPerLayer int     `json:"neuronsPerLayer"`
    LearningRate    float64 `json:"learningRate"`
    Imagem          string  `json:"imagem"`
    MaxEpocas       int     `json:"maxEpocas"`
}
```

**Config extra em `imgreg_minibatch`** (estende a config base):
```go
BatchSize  int `json:"batchSize"`   // 8, 16, 32, 64
NumWorkers int `json:"numWorkers"`  // 2, 4, 8, runtime.NumCPU()
```

**Step** (todos os backends — `LossHistorico` apenas no step `Done=true`):
```go
type Step struct {
    Epoca         int          `json:"epoca"`
    MaxEpocas     int          `json:"maxEpocas"`
    Loss          float64      `json:"loss"`
    OutputPixels  [][3]float64 `json:"outputPixels"`
    ActiveLayer   int          `json:"activeLayer"`
    Done          bool         `json:"done"`
    Convergiu     bool         `json:"convergiu"`
    LossHistorico []float64    `json:"lossHistorico"` // populado APENAS quando Done=true
    ElapsedMs     int64        `json:"elapsedMs"`     // ms desde início do treino
    EpochMs       int64        `json:"epochMs"`       // duração da última época em ms
}
```

**BenchStep** — tipo exclusivo do `imgreg_bench`, envolve um Step com identificação do backend:
```go
// BenchStep é o tipo serializado no SSE do benchmark
type BenchStep struct {
    Backend string `json:"backend"` // "goroutines" | "matrix" | "minibatch"
    Step    Step   `json:"step"`    // Step do backend correspondente (tipos locais copiados)
}
```

O package `imgreg_bench` define seu próprio `Step` struct idêntico aos outros backends (sem importar os packages individuais) e usa `BenchStep` como envelope no canal SSE.

**Frequência de steps (epochStep):** todos os backends (incluindo benchmark) usam `const epochStep = 5` — envia step a cada 5 épocas, igual ao original. O benchmark envia `BenchStep` com esta mesma frequência por backend.

**Net** (cada package define seu próprio tipo `Net` internamente — o `main.go` armazena como ponteiro opaco e não acessa campos diretamente):
- `imgreg_goroutines.Net` e `imgreg_minibatch.Net`: mesma estrutura do `imgreg.Net` original (`LayerSizes []int`, `W [][][]float64`, `B [][]float64`)
- `imgreg_matrix.Net`: internamente usa `[]gonum/mat.Dense` por camada; expõe apenas `LayerSizes []int` no JSON para a viz de rede

---

## Backends em Detalhe

### `imgreg_goroutines`

- Paralelismo no forward+backward por pixel dentro de cada época
- Canal buffered de tamanho 256: `gradCh := make(chan gradResult, 256)` onde `gradResult` contém `(gradW, gradB)` de um pixel
- Lança 256 goroutines, cada uma processa 1 pixel e envia seu resultado no canal; `sync.WaitGroup` garante que o main loop aguarde todas antes de fechar o canal
- O loop principal coleta os 256 pares via `range gradCh` e os soma em buffers de acumulação pré-alocados (sem mutex, pois a soma é feita em goroutine única)
- Faz um único update de pesos ao final da época com os gradientes acumulados (batch gradient descent)
- Diferença de comportamento vs original: SGD → batch GD (gradiente mais estável, convergência diferente)
- Sem dependências externas além da stdlib

### `imgreg_matrix`

- Reescreve forward/backward usando `gonum.org/v1/gonum/mat`
- Representa cada época como operações matriciais sobre todos os 256 pixels simultaneamente
- Layout das matrizes por camada de transição:
  - `A`: 256×fanIn (ativações de todos os pixels para esta camada)
  - `W`: fanIn×fanOut (pesos, transposto da convenção `W[i][j]` existente)
  - `B`: 1×fanOut (bias, adicionado linha a linha via loop — gonum não faz broadcast)
  - `Z = A * W + B`: 256×fanOut (pré-ativações)
- Backprop também operado matricialmente: `dA = dZ * Wᵀ`, `dW = (Aᵀ * dZ) / 256` — dividido por 256 para manter a mesma escala de gradiente que os outros backends (média, não soma)
- Dependência: `gonum.org/v1/gonum/mat`

### `imgreg_minibatch`

- Divide os 256 pixels embaralhados em batches de `BatchSize` pixels
- Pool de `NumWorkers` goroutines processa batches via canal de trabalho
- Cada worker: recebe um batch, calcula gradientes acumulados do batch, envia resultado
- Main loop: coleta resultados, atualiza pesos após cada batch completo
- Meio-termo entre SGD (update por pixel) e batch GD (update por época)
- Config extra: `BatchSize` (8/16/32/64) e `NumWorkers` (2/4/8/NumCPU)

---

## Endpoints da API

### Por backend (mesmo padrão do imgreg existente)

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
POST /api/imgreg-bench/config        ← salva BenchConfig em global benchCfg (mesmo padrão dos outros backends)
GET  /api/imgreg-bench/train         (SSE — stream único multiplexado, nome consistente com demais)
POST /api/imgreg-bench/reset
```

**BenchConfig** (superset da Config base, armazenada como `benchCfg *imgreg_bench.BenchConfig` no `main.go`):
```go
type BenchConfig struct {
    HiddenLayers    int     `json:"hiddenLayers"`
    NeuronsPerLayer int     `json:"neuronsPerLayer"`
    LearningRate    float64 `json:"learningRate"`
    Imagem          string  `json:"imagem"`
    MaxEpocas       int     `json:"maxEpocas"`
    BatchSize       int     `json:"batchSize"`
    NumWorkers      int     `json:"numWorkers"`
    Parallel        bool    `json:"parallel"`
}
```

**`imgreg_bench` expõe:**
```go
func Rodar(ctx context.Context, cfg BenchConfig, ch chan<- BenchStep)
```
Sem retorno de `Net` (benchmark não persiste redes). O `main.go` chama `imgreg_bench.Rodar(ctx, *benchCfg, benchCh)` no handler.

**Cancelamento do benchmark:** segue o mesmo padrão dos outros backends. O `main.go` armazena um `benchCancel context.CancelFunc` global (protegido pelo mesmo `sync.RWMutex`). O handler `/api/imgreg-bench/train` cria `ctx, cancel := context.WithCancel(r.Context())`, armazena `benchCancel = cancel`, e passa `ctx` para `imgreg_bench.Rodar`. O handler `/api/imgreg-bench/reset` chama `benchCancel()` se não-nil e limpa o estado (igual ao `handleImgregReset` existente).

**Benchmark SSE — stream único multiplexado (servidor faz o fanout):**

O endpoint `/api/imgreg-bench/train` é uma única SSE connection. O servidor lança os backends em goroutines internas e serializa todos os steps em um único stream. Cada step inclui o campo `backend: "goroutines"|"matrix"|"minibatch"` para o frontend saber qual coluna atualizar.

- **Modo `parallel: true`**: os 3 backends rodam simultaneamente em goroutines separadas; steps de todos chegam intercalados no stream
- **Modo `parallel: false`**: os 3 backends rodam sequencialmente (goroutines → matrix → minibatch); steps de um backend chegam completos antes do próximo começar; o frontend pode exibir progresso em tempo real para o backend em execução e resultados finais dos anteriores

Steps intermediários de cada backend incluem `ElapsedMs`, `EpochMs`, `Loss`, `OutputPixels`. O step final por backend (`Done=true`) inclui também `LossHistorico`, `Convergiu` e as métricas de timing para os cards comparativos.

---

## Frontend

### Sidebar — accordion collapsible

O item `IMG REGRESSION` existente na sidebar é convertido em accordion:

```
▼ IMG REGRESSION        ← clicável, expande/colapsa
   ├─ Standard          ← item filho, substitui o link atual
   ├─ Goroutines
   ├─ Matrix
   ├─ Mini-batch
   └─ Benchmark
```

- Ao clicar no pai, o grupo expande ou colapsa (os filhos aparecem/somem)
- Ao clicar em um filho, abre a view correspondente e marca o filho como `active`
- Apenas um item filho pode estar `active` por vez
- O accordion pai não navega para uma view própria
- A view `Standard` é a view `imgreg` existente, sem mudanças no HTML interno

### Views dos backends (Goroutines / Matrix / Mini-batch)

Cópia visual do painel `Standard` com novos IDs de elementos. Cada view tem:
- Controles: imagem, camadas ocultas, neurônios, lr, épocas
- `Mini-batch` adiciona dois dropdowns extras: `Batch Size` e `Workers`
- Stat-bar em tempo real: `época`, `loss`, `ms/época`, `throughput px/s`
- Visualização: grid target + viz de rede + grid output + gráfico de loss

**Definição de throughput:** `pixels_por_epoca / (epochMs / 1000)` onde `pixels_por_epoca = 256`. Fórmula idêntica para todos os backends para comparabilidade.

### View Benchmark

- Toggle `PARALELO / SEQUENCIAL` (controla o campo `parallel` da config)
- Config compartilhada: imagem, camadas, neurônios, lr, épocas, batch size, workers
- Botão `RODAR BENCHMARK`
- 3 colunas (Goroutines / Matrix / Mini-batch), cada uma com: grid output 16×16 + stat em tempo real
- Em modo paralelo: as 3 colunas animam simultaneamente
- Em modo sequencial: a coluna ativa anima, as outras aguardam; ao terminar cada uma, exibe resultado e passa para a próxima
- Ao final de todos os 3: cards comparativos com `tempo total (ms)`, `ms/época médio`, `throughput (px/s)`, `loss final`
- Gráfico de barras comparativo entre os 3 backends

---

## Observações

- O package `imgreg` original **não é modificado**
- Estado global em `main.go` é replicado para cada backend (`imgreg_goroutinesRede`, `imgreg_matrixRede`, etc.) seguindo o padrão existente com `sync.RWMutex`
- O `go.mod` precisará adicionar `gonum.org/v1/gonum` como dependência
- Os campos de timing `ElapsedMs` e `EpochMs` são medidos com `time.Since` dentro de cada `Treinar`
- `LossHistorico` é enviado **apenas** no step `Done=true`, para evitar payload crescente nos steps intermediários
