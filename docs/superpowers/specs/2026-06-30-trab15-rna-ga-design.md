# Trabalho 15 — AG que descobre a melhor arquitetura de uma RNA (RNA + AG)

**Data:** 2026-06-30 · **Branch:** `feat/trab15-rna-ga`
**Aula:** 20 (O desafio de integrar RNA e AG)

> Empresa de engenharia elétrica quer uma MLP para modelar um processo (15
> entradas → 13 saídas). Um **Algoritmo Genético** procura a **melhor arquitetura**
> da rede. O fitness de cada indivíduo é o **MSE** da MLP treinada com a
> configuração que o cromossomo define. Minimizar o MSE.

## Objetivo

Implementar, em Go com visualização web "MUITO boa", um AG que evolui
arquiteturas de MLP. Cada indivíduo = uma configuração de rede; avaliar =
treinar a rede e medir o MSE sobre 100 padrões. Mesmo padrão de qualidade dos
trabalhos anteriores (pacote Go isolado + view dedicada).

## Não-objetivos

- Não é fuzzy (fuzzy são as Aulas 17–19, outro trabalho).
- Não reaproveitar `mlpfunc` (especializado a 1-in/1-out, função-aprox); escrevo
  um MLP próprio e isolado no pacote novo.

## Dados (fixos por run, seeded)

100 padrões: 15 entradas ~ U(3, 1457), 13 saídas ~ U(58, 312), aleatórios.
Treino = teste (o enunciado pede isso para simplificar). As saídas são aleatórias
(sem relação com as entradas) — então a tarefa é, na prática, medir quão bem cada
arquitetura consegue **memorizar/ajustar** 100 padrões.

## Cromossomo — vetor de 6 genes (typed) + `String()` na UI

`Genes [6]float64` com decodificadores; a UI mostra a codificação em String
(ex.: `8 | 3 | 0.0100 | 500 | online | normaliza`):

| pos | gene | faixa | tipo |
|-----|------|-------|------|
| 0 | neurônios por camada oculta | 2–15 | int |
| 1 | nº de camadas ocultas | 2–5 | int |
| 2 | taxa de aprendizagem | 1e-5–0.1 | real |
| 3 | nº máximo de épocas | 20–1000 | int |
| 4 | online (1) / offline (2) | {1,2} | int |
| 5 | normaliza (1) / não (2) | {1,2} | int |

Crossover e mutação operam neste vetor de 6 posições (honra o "vetor String").

## MLP própria (gonum `mat`, tanh em todas as camadas)

- Arquitetura: 15 → (camadas × neurônios) → 13, ativação **tanh** em todas as camadas.
- **Online** (atualiza pesos por padrão) ou **offline/batch** (acumula e atualiza por época).
- Forward/backward **batcheado com gonum** (`mat.NewDense`, `C.Mul`, `.T()`) no
  modo offline (matriz 100×15 — onde o BLAS realmente acelera); por-padrão no online.
- **Normalização** opcional: entradas e saídas escaladas para [-1, 1] quando o
  gene pedir; senão usa valores crus.
- **MSE em unidades reais (58–312):** a saída normalizada é denormalizada antes
  de medir o MSE, então a comparação entre indivíduos é justa e arquiteturas sem
  normalização ficam claramente ruins (tanh satura em ±1 vs alvos ~58–312) — o AG
  descobre que normalizar é essencial.
- **Pesos iniciais aleatórios, mas com seed derivado do cromossomo + seed do run**
  → fitness determinístico (memoização válida + execução reprodutível).

## AG (fiel ao enunciado)

- População inicial: **40** indivíduos aleatórios.
- Seleção: **roleta** (menor MSE → maior probabilidade; `fitness ∝ 1/(MSE+ε)`).
- Taxa de cruzamento **50%**: 20 indivíduos viram genitores → 10 casais → 20 filhos.
- Cruzamento: **1 ponto de corte**, dois filhos.
- Mutação **5%**: para cada filho, com prob. 5% sorteia 1 posição e gera um novo
  valor válido para ela.
- Substituição (escolha nossa, justificada): **elitista** — os 20 melhores
  sobrevivem e os 20 filhos substituem a pior metade.
- Critério de parada: **100 gerações** (limite do enunciado).

### Desempenho

- Avaliação dos indivíduos **em paralelo** (goroutines, pool = nº de CPUs).
- **Memoização** de fitness por cromossomo (chave = genes canônicos) — elites e
  duplicatas não retreinam.
- **Teto de épocas ajustável** na UI (ex.: 50–1000) para demo ágil vs. busca completa.
- Matmul batcheado (gonum) no modo offline.

> Nota honesta: em matrizes 15×15 no modo online o overhead do BLAS pode passar do
> ganho; o ganho real vem de paralelismo + memoização + batch no offline.

## Backend — `web/server/rnaga/rnaga.go`

Tipos: `Cromossomo` (Genes + decode + String), `Config`, `Individuo`, `Step`,
`Result`, `DefaultConfig()`, `Treinar(progressCh chan<- Step, cfg Config) Result`.
`Step` por geração: melhor cromossomo/MSE, média, população (genes + MSE), e a
**grade de MSE por (neurônios × camadas)** para o heatmap.
Endpoints em `main.go`: `/api/rnaga/{config,train,reset,result}`.

## Frontend — `web/frontend/src/views/RnaGaView.tsx`

1. **Diagrama animado da MELHOR rede** (15 → camadas×neurônios → 13) que se
   transforma quando o AG melhora, anotado com LR/épocas/online/normalização (SVG).
2. **Heatmap de MSE** sobre (neurônios 2–15 × nº de camadas 2–5) — o espaço de
   busca sendo varrido (melhor MSE por célula).
3. **Convergência** (melhor/média MSE por geração) + tabela/scatter da **população**.
4. **Cromossomo vencedor** (String + decodificado) + **MSE final** em destaque (entregável).
5. Controles (pop, gerações, teto de épocas, etc.) + card educacional.

## Testes (Go, TDD)

- Decode/round-trip dos genes dentro das faixas; mutação sempre gera valor válido
  por posição; crossover de 1 ponto preserva os 6 genes.
- Forward com dims corretas (15→…→13); MSE cai ao treinar uma arquitetura boa
  (normalizada, LR razoável) vs. uma ruim (sem normalização).
- AG reduz o melhor-MSE ao longo das gerações.
- Memoização determinística: mesmo cromossomo → mesmo MSE.

## Validação
`go test ./rnaga`, `go build`, build do frontend, rodar local e validar no
navegador com Playwright MCP (screenshots em `.playwright-mcp/screenshots/`).

## Nomes
pkg `rnaga` · view `RnaGaView` · rota `/api/rnaga/*` · viewId `rna-ga` ·
sidebar "GA · Arquitetura RNA".
