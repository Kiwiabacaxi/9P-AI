# Inteligência Artificial — RNA + Algoritmos Genéticos + Fuzzy

Projeto da disciplina de **Inteligência Artificial** na faculdade: uma coleção de
trabalhos que implementam, do zero, os algoritmos clássicos da área — de **redes
neurais** (Hebb → Perceptron → MADALINE → MLP → CNN) a **algoritmos genéticos** e
**otimização** (função matemática, caixeiro viajante, e até um AG que descobre a
melhor arquitetura de uma rede neural), fechando com **lógica fuzzy** (inferência
Mamdani para qualidade da água).

A ideia é entender cada algoritmo **pela descoberta**: várias implementações são
propositalmente "ingênuas" (com laços aninhados e tudo mais) porque o objetivo é
ver o algoritmo funcionando, não otimizar performance. Cada trabalho tem uma
**visualização web interativa** — mapas, gráficos de convergência, diagramas de
rede animados, etc.

> Dentro da aplicação, a aba **"Arquitetura" → Mapa dos Trabalhos** lista onde
> encontrar cada um dos 16 trabalhos (em qual item do menu clicar e qual pacote o
> implementa).

## Os trabalhos

**Redes Neurais**

| # | Trabalho | Menu |
|---|----------|------|
| 1 | Regra de Hebb | Hebb |
| 2 | Perceptron (portas lógicas + letras A/B) | Perceptron |
| 3 | MADALINE (letras A–M) | MADALINE |
| 4 | MLP — reprodução do exemplo dos slides | MLP Desafio |
| 5 | MLP — reconhecimento de letras A–Z | MLP Letras |
| 6 | Rede Convolucional (EMNIST) | CNN EMNIST |
| 7 | Investimentos — séries temporais | MLP Ações |

**Algoritmos Genéticos & Otimização**

| # | Trabalho | Menu |
|---|----------|------|
| 8 | AG com função matemática f(x) | GA · Aula 10 |
| 9 | AG parametrizável (torneio, elitismo, 2 pontos) | GA · Aula 11 |
| 10 | AG para grade de horários (cromossomo matricial) | GA · Horário (Aula 12) |
| 11 | Caixeiro viajante — 10 cidades | GA · TSP |
| 12 | Caixeiro viajante multipopulacional | GA · TSP Multi-ilhas |
| 13 | AG com cromossomos reais — Rastrigin 3D | GA · Rastrigin 3D |
| 14 | AG com seleção por Ranking — TSP | GA · TSP Ranking |
| 15 | AG que descobre a arquitetura de uma RNA | GA · Arquitetura RNA |

**Lógica Fuzzy**

| # | Trabalho | Menu |
|---|----------|------|
| 16 | Qualidade da água — inferência Mamdani (SABESP) | Fuzzy · Água |

Bônus (além dos 16): **MLP Funções**, **MLP Ortogonal**, **IMG_REGRESSION** (MLP
que "pinta" uma imagem, com variações goroutines/matriz/mini-batch/benchmark) e
**GA · TSP Comparativo**.

## Como rodar (recomendado — roda tudo)

Backend em Go + frontend React, ligados por HTTP/SSE:

```bash
cd web
make run          # compila o frontend e sobe o servidor em http://localhost:8080
```

Precisa de **Go 1.24+** e **Node 18+** (para o build do frontend).

Também há uma **TUI no terminal** para alguns dos primeiros trabalhos:

```bash
cd cli
go run ./trab01-hebb
go run ./desafio-mlp-letras
# etc.
```

## Versão online (subconjunto, via WebAssembly)

O deploy no GitHub Pages compila os algoritmos como **WebAssembly** e roda os
**trabalhos clássicos de RNA** (Hebb, Perceptron, MADALINE, MLP, CNN, image
regression) direto no navegador. Os trabalhos de **AG/otimização** (TSP,
Rastrigin, Ranking, RNA+AG) e o **Fuzzy** usam o backend Go, então a experiência
completa é **local** com `make run`.

## Estrutura

```
web/
  server/                Backend Go — um pacote por trabalho
    main.go              Rotas HTTP + streaming SSE
    hebb/ madaline/ mlp/ cnn/ timeseries/ …        (redes neurais)
    genetico/ genetico2/ horario/ tsp/ tspmulti/   (algoritmos genéticos)
    agrastrigin/ tspranking/ rnaga/                (AG reais, ranking, RNA+AG)
    fuzzy/                                         (lógica fuzzy — qualidade da água)
    wasm/                Build WebAssembly (subconjunto p/ o navegador)
    cmd/rnabench/        CLI de benchmark do Trabalho 15
  frontend/              Frontend React + TypeScript (Vite)
    src/views/           Uma tela por trabalho
    src/components/       Componentes compartilhados (mapa, gráficos, layout)
  static/                Saída do build (servida pelo servidor / Pages)

cli/                     TUIs no terminal (Go + Charm) de alguns trabalhos
slides/                  PDFs das aulas
docs/                    Specs de design dos trabalhos
```

## Tecnologias

- **Go** — algoritmos, servidor HTTP, WebAssembly, TUIs
- **gonum** — multiplicação de matrizes (BLAS) nos trabalhos que precisam de velocidade
- **React + TypeScript + Vite** — frontend
- **Leaflet · Plotly · Recharts** — mapas e gráficos interativos
- **Charm** (Bubble Tea + Lipgloss) — interfaces de terminal
- **GitHub Actions** — build WASM + deploy no Pages
