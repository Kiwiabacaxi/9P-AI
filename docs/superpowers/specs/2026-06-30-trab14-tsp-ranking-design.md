# Trabalho 14 — AG com seleção por Ranking aplicado ao TSP

**Data:** 2026-06-30 · **Branch:** `feat/trab14-tsp-ranking`
**Aulas:** 13 (Caixeiro Viajante) + 16 (AG com Ranking)

> Slide 19 da Aula 16 (literal): *"Volte no exercício do caixeiro viajante e
> altere-o para que ele tenha Ranking linear ou exponencial."*

## Objetivo

Resolver o TSP (roteamento de 10 cidades do Triângulo Mineiro, partindo de
Uberaba, minimizando a distância) com um Algoritmo Genético cuja **seleção é por
posição no ranking** (não pelo valor absoluto do fitness). Implementação Go
didática, sem toolbox/shortcut, com visualização web fiel — mesmo padrão de
qualidade do Trabalho 13 (Rastrigin / `agrastrigin`).

O ranking ataca os 3 problemas da seleção proporcional clássica citados na Aula
16: convergência prematura, domínio dos extremamente aptos e perda de
diversidade genética.

## Não-objetivos

- Não alterar o pacote `tsp` existente (versão "vida real" com OSRM/tempo).
- Sem WebAssembly (igual ao `agrastrigin`: backend-only via SSE, rodando local).

## Arquitetura

Pacote Go novo e independente espelhando `agrastrigin`; view nova espelhando
`RastriginView`; reaproveita o componente `TspMap` (react-leaflet) e o tipo
`TspCidade` já existentes.

### Backend — `web/server/tspranking/tspranking.go`

**Dados.** 10 cidades fixas do Triângulo Mineiro com coordenadas reais
(Uberaba = id 0 = depósito). Matriz de distâncias N×N simétrica:
- usa os números da **tabela da Aula 13** (slide 9) onde existem;
- preenche os pares com "—" por **1,3 × Haversine** (fator de desvio de estrada,
  pra ficar na mesma escala de km da tabela, não menor);
- guarda uma máscara `fonteTabela[i][j]` (tabela vs. preenchido) pra exibição.

**Encoding.** `Tour []int` = permutação de índices de cidade, ciclo fechado
(volta à origem). O custo do ciclo é invariante à rotação; pra exibição o tour é
rotacionado pra iniciar em Uberaba.

**Operadores (sem shortcut).**
- Cruzamento: **OX** (Order Crossover) e **PMX** (Partially Mapped) — ambos
  garantem permutação válida (sem cidade repetida/faltando), requisito da Aula 13.
- Mutação: **swap** (troca 2 cidades — exemplo do slide) e **inversão** (reverte
  segmento, 2-opt-like).
- Elitismo: mantém os `p` melhores.

**Seleção (o foco).**
- `rankingLinear` — Baker 1985:
  `P_i = (1/N)·[η_max − (η_max − η_min)·(i−1)/(N−1)]`, com `η_min = 2 − η_max`,
  `i` = posição 1-based (1 = melhor). `η_max ∈ [1,2]` configurável (default 1.5).
- `rankingExp` — `P_i = c^(N−i) / Σ_j c^(N−j)`, `c > 1` configurável.
  Pesos calculados de forma numericamente estável (normaliza pelo maior).
- `torneio` (k configurável) e `roleta` (proporcional, `fitness = maxDist − dist + ε`)
  — mantidos só para **comparação** com o ranking.

Seleção por ranking/roleta usa amostragem por roleta sobre as probabilidades.

**Streaming (SSE `Step` por geração):** geração, melhor tour + distância, média,
pior, diversidade, melhor global; **e `popDist []float64`** (distâncias da
população ordenadas asc) — alimenta o Laboratório do Ranking ao vivo.

**Endpoints** (em `main.go`, padrão dos outros): `/api/tspranking/{config,train,reset,result,cidades}`.
`cidades` devolve a lista de cidades + matriz + máscara de fonte (pro mapa e a tabela).

### Frontend — `web/frontend/src/views/TspRankingView.tsx`

Mesma régua do `RastriginView`:
1. Header + controles de config; cada seleção revela seu controle de pressão
   (η_max / c / k).
2. Métricas: geração, melhor distância (km), melhoria % vs. geração inicial.
3. **Mapa animado do melhor tour** (reusa `TspMap`) + player de replay
   (▶ / scrub / velocidade), igual ao Rastrigin.
4. Gráfico de **convergência** (melhor / média / melhor-acumulado por geração).
5. **🏆 Laboratório do Ranking (destaque):** curvas de `P_i` por posição (linear
   vs. exponencial) com sliders de pressão, reproduzindo as tabelas da Aula 16;
   sobrepõe as distâncias reais por rank da geração atual (do player) → mostra
   "ranking ignora o fitness absoluto, usa só a posição". Inclui a tabelinha
   `Rank | P_i | %` igual aos slides "Resultado Final".
6. Tabela da matriz de distâncias (Aula 13), marcando célula de tabela vs. preenchida.
7. Card educacional: TSP, OX/PMX, swap/inversão, e **Ranking** (fórmulas + porquê).

### Tipos — `web/frontend/src/api/types.ts`
`TspRankSelecao = 'rankingLinear'|'rankingExp'|'torneio'|'roleta'`,
`TspRankCruzamento = 'ox'|'pmx'`, `TspRankMutacao = 'swap'|'inversao'`,
`TspRankConfig`, `TspRankStep`, `TspRankResult`, `TspRankCidadesResp`.
ViewId += `'tsp-ranking'`.

### Wiring
`main.go` (import + globals + 5 handlers + 5 rotas), `App.tsx` (import +
viewComponents), `Sidebar.tsx` (nav item "AG · TSP Ranking"), `TopBar.tsx` (label).

## Testes (Go) — oráculos exatos dos slides
- Ranking linear N=5, η_max=1.5 → `P = [0.30, 0.25, 0.20, 0.15, 0.10]`.
- Ranking exponencial N=5, c=2 → pesos `[16,8,4,2,1]`, `P ≈ [.516,.258,.129,.064,.032]`.
- Probabilidades somam 1 e são monótonas decrescentes com o rank.
- OX, PMX, swap e inversão sempre produzem permutação válida (sem duplicar/perder cidade).
- Matriz de distâncias simétrica, diagonal zero, sem zeros fora da diagonal.

## Validação
`go test ./tspranking`, `go build`, build do frontend, rodar local e validar no
navegador com Playwright MCP (screenshots em `.playwright-mcp/screenshots/`).

## Nomes
pkg `tspranking` · view `TspRankingView` · rota `/api/tspranking/*` ·
viewId `tsp-ranking` · sidebar "AG · TSP Ranking".
