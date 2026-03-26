# Redes Neurais Artificiais

Repositorio com os trabalhos praticos da disciplina de **Redes Neurais Artificiais** (Inteligencia Artificial).

Todos os algoritmos rodam **direto no browser** via WebAssembly — sem backend, sem instalacao.

> **[Abrir o Mission Control](https://kiwiabacaxi.github.io/9P-AI/)**

## Algoritmos implementados

| # | Algoritmo | Aula | Arquitetura | Ativacao | Regra de atualizacao |
|---|-----------|------|-------------|----------|---------------------|
| 1 | **Hebb** | 02 | 2 -> 1 | sign(y_in) | w <- w + x*t (sempre) |
| 2 | **Perceptron Portas** | 03 | 2 -> 1 | sign(y_in) | w <- w + a(t-y)x (so no erro) |
| 3 | **Perceptron Letras** | 03 | 49 -> 1 (7x7) | sign(y_in) | w <- w + a(t-y)x (so no erro) |
| 4 | **MADALINE** | 04 | 35 -> 13 ADALINE -> 13 | sign / argmin | MRII — atualiza unidade com menor \|y_in\| |
| 5 | **MLP Desafio** | 05 | 3 -> 2 -> 3 | tanh | Backpropagation |
| 6 | **MLP Letras** | 05 | 35 -> 15 -> 26 (A-Z) | tanh | Backpropagation |
| 7 | **MLP Image Regression** | 05 | 2 -> [NxM] -> 3 | ReLU / Sigmoid | SGD estocastico / He init / MSE |

### Backends de Image Regression

O Image Regression tem 4 implementacoes que demonstram diferentes estrategias de otimizacao:

| Backend | Descricao | Disponivel no WASM |
|---------|-----------|-------------------|
| **Standard** | SGD online, 1 pixel por vez | Sim |
| **Goroutines** | Batch GD com goroutines paralelas | Sim |
| **Matrix** | Batch GD matricial com gonum/mat (BLAS) | Nao (cgo) |
| **Mini-batch** | Mini-batch GD com workers configuravel | Sim |

## Estrutura do projeto

```
├── cli/                          Programas TUI interativos (Go + Charm)
│   ├── trab01-hebb/              Regra de Hebb — portas logicas
│   ├── trab02-perceptron-letras/ Perceptron — reconhecimento A/B
│   ├── trab02-perceptron-portas/ Perceptron — portas logicas
│   ├── trab03-madaline/          MADALINE — reconhecimento A-M
│   ├── desafio-mlp/              MLP 3->2->3 — backpropagation
│   └── desafio-mlp-letras/       MLP 35->15->26 — reconhecimento A-Z
├── slides/                       PDFs das aulas
└── web/
    ├── static/
    │   └── index.html            Frontend (HTML/CSS/JS, sem frameworks)
    └── server/
        ├── main.go               HTTP server + SSE handlers
        ├── hebb/                  package hebb
        ├── perceptron_portas/    package perceptronportas
        ├── perceptron_letras/    package perceptronletras
        ├── madaline/             package madaline
        ├── mlp/                  package mlp
        ├── letras/               package letras
        ├── imgreg/               package imgreg (Standard)
        ├── imgreg_goroutines/    package igoroutines
        ├── imgreg_matrix/        package imatrix (gonum/mat)
        ├── imgreg_minibatch/     package iminibatch
        ├── imgreg_bench/         package ibench (benchmark)
        └── wasm/                 WebAssembly bridge (syscall/js)
```

Para detalhes sobre cada programa CLI, veja o [README da pasta cli](cli/README.md).

## Como usar

### GitHub Pages (sem instalar nada)

Acesse **https://kiwiabacaxi.github.io/9P-AI/** — tudo roda no seu browser via WebAssembly.

### Servidor local (desenvolvimento)

```bash
cd web
make run   # compila Go, inicia servidor na :8080 e abre o browser
```

### TUI (terminal interativo)

Cada trabalho tem uma TUI em Go. A partir da pasta `cli/`:

```bash
go run ./trab01-hebb
go run ./trab02-perceptron-letras
go run ./trab02-perceptron-portas
go run ./trab03-madaline
go run ./desafio-mlp
go run ./desafio-mlp-letras
```

## Arquitetura web

O projeto usa **dual-mode** — o mesmo `index.html` funciona em dois modos:

| Modo | Quando | Como funciona |
|------|--------|---------------|
| **Servidor** | `localhost:8080` | Frontend faz `fetch()` / `EventSource` para o Go HTTP server |
| **WASM** | GitHub Pages / qualquer host estatico | Go compilado para WebAssembly roda em **Web Worker** (thread separada, nao trava a UI) |

A deteccao e automatica: se a porta e 8080, usa o servidor; senao, carrega o WASM.

### Build do WASM

```bash
cd web/server
GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o ../static/main.wasm ./wasm/
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" ../static/
```

O GitHub Actions faz isso automaticamente a cada push para `main`.

## Tecnologias

- **Go 1.24** — algoritmos + HTTP server + WebAssembly
- **gonum/mat** — multiplicacao matricial otimizada (backend Matrix)
- **syscall/js** — bridge Go <-> JavaScript para WASM
- **Web Workers** — execucao em thread separada no browser
- **Charm** (Bubble Tea + Lipgloss) — TUI interativa
- **HTML/CSS/JS** — frontend sem frameworks
- **GitHub Actions** — CI/CD para deploy automatico no Pages
