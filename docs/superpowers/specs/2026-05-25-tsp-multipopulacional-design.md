# Trabalho 12 — TSP com AG Multi-populacional (modelo de ilhas)

**Data:** 2026-05-25
**Aula base:** Aula 14 — "Algoritmo genético multi-populacional"
**Status:** Design aprovado, aguardando review da spec

## 1. Objetivo

Implementar o problema do Caixeiro Viajante (TSP) resolvido por um **Algoritmo
Genético multi-populacional** (modelo de ilhas), conforme o enunciado da Aula 14:

- Várias **ilhas** (subpopulações) evoluindo simultaneamente ("em paralelo").
- **Migração** periódica: a cada N gerações, o(s) melhor(es) indivíduo(s) de cada
  ilha migram para outra ilha (topologia em anel — a "dança de cadeiras").
- Objetivo pedagógico: aumentar diversidade e escapar de mínimos locais.
- Problema concreto: roteamento de um caminhão pelas **20 cidades do Triângulo
  Mineiro**, minimizando a distância percorrida.
- Seleção/cruzamento à escolha (reusamos roleta/torneio + OX/PMX já existentes).
- Requisito do enunciado: usar **threads** → no Go, goroutines.

Defaults do exemplo do professor: **3 ilhas × 20 indivíduos**, migração a cada
**10 gerações**, **≥50 gerações**, melhor fitness escolhido entre os 60 indivíduos.

## 2. Decisões de design (já alinhadas)

| Decisão | Escolha |
|---------|---------|
| Estrutura | Pacote `web/server/tspmulti` + view nova `TspMultiView` ("GA · TSP Multi-ilhas") |
| Cidades | 20 cidades do **mapa** do slide, coords reais, matriz via haversine/OSRM |
| Cidade nº 14 (rótulo ilegível) | Preenchida com **Nova Ponte** (trocável) |
| Modelo de paralelismo | **A — Lockstep paralelo** (orquestrador comanda gerações; ilhas evoluem 1 passo concorrentes via goroutines + WaitGroup; migração em intervalos fixos) |
| Visualização | **Rica + pedagógica** — mapa do melhor global + small-multiples por ilha + convergência multi-linha + animação de migração + tabela por ilha |
| Comparativo | **Pop única vs multi-ilhas** (toggle): roda em paralelo um AG de população única com o MESMO total de indivíduos e seed, sobrepõe as curvas para evidenciar a multi escapando do mínimo local |
| Reforços pedagógicos | Medidor de diversidade (salto pós-migração); anotações de "salto" na curva; cor de gene migrante; painel "melhor de todas" com contador de estagnação |

### Por que lockstep paralelo (A)?

Determinístico (com seed fixa), fácil de transmitir um snapshot coerente por
geração via SSE, e satisfaz o requisito de "threads" (cada geração roda as ilhas
em goroutines concorrentes). Alternativas descartadas: ilhas totalmente
assíncronas (não-determinístico, snapshot incoerente, complexidade sem ganho
pedagógico) e ilhas sequenciais (não satisfaz "com threads").

## 3. Backend — pacote `web/server/tspmulti`

### 3.1 Refactor de pré-requisito no pacote `tsp`

Extrair de `tsp.Treinar` a lógica de **uma geração** para uma função pública
reutilizável, sem alterar o comportamento do Trabalho 11:

```go
// EvoluirUmaGeracao recebe a população atual e devolve a próxima geração.
// Encapsula elitismo + seleção + cruzamento + mutação (a lógica hoje no corpo
// do loop de tsp.Treinar). tsp.Treinar passa a chamá-la dentro do seu loop.
func EvoluirUmaGeracao(pop []Individuo, cfg Config, rng *rand.Rand,
    matDist, matDur [][]float64) []Individuo
```

Os helpers já existentes continuam reaproveitados: `gerarPopulacaoInicial`,
`selecionarRoleta`, `selecionarTorneio`, OX/PMX, mutação swap/inversão,
`extrairElites`, cálculo de distância/custo, `Individuo`. Helpers hoje privados
necessários ao orquestrador (ex: `diversidade`) são **exportados** (`Diversidade`)
como parte do refactor; estatísticas de melhor/média usam os campos exportados de
`Individuo` (`.Distancia`, `.Custo`).

**Validação do refactor:** o Trabalho 11 (`tsp`) deve continuar produzindo
resultado idêntico com a mesma seed (teste de regressão de determinismo).

### 3.2 Config

```go
type MultiConfig struct {
    NumIlhas          int        `json:"numIlhas"`          // default 3
    TamIlha           int        `json:"tamIlha"`           // default 20 (pop por ilha)
    MaxGeracoes       int        `json:"maxGeracoes"`       // default 50
    IntervaloMigracao int        `json:"intervaloMigracao"` // default 10
    NumMigrantes      int        `json:"numMigrantes"`      // default 1
    Topologia         string     `json:"topologia"`         // "anel" (único por ora)
    CompararPopUnica  bool       `json:"compararPopUnica"`  // roda baseline de pop única em paralelo
    Seed              int64      `json:"seed,omitempty"`
    GA                tsp.Config `json:"ga"`                // seleção, cruzamento, mutação, probs, elitismo
}
```

`sanitizar` aplica mínimos: NumIlhas ≥ 2, TamIlha ≥ 4, MaxGeracoes ≥ 1,
1 ≤ IntervaloMigracao ≤ MaxGeracoes, 1 ≤ NumMigrantes ≤ TamIlha-1.

### 3.3 Tipos de progresso/resultado

```go
type IlhaStep struct {
    Ilha        int     `json:"ilha"`
    MelhorTour  []int   `json:"melhorTour"`
    MelhorDist  float64 `json:"melhorDist"`
    MelhorCusto float64 `json:"melhorCusto"`
    MediaDist   float64 `json:"mediaDist"`
    Diversidade int     `json:"diversidade"`
}

type MultiStep struct {
    Geracao          int        `json:"geracao"`
    Ilhas            []IlhaStep `json:"ilhas"`
    MelhorGlobalTour []int      `json:"melhorGlobalTour"`
    MelhorGlobalDist float64    `json:"melhorGlobalDist"`
    IlhaVencedora    int        `json:"ilhaVencedora"`     // qual ilha tem o melhor global agora
    GeracoesSemMelhora int      `json:"geracoesSemMelhora"` // estagnação do melhor global (painel "melhor de todas")
    DiversidadeGlobal  int      `json:"diversidadeGlobal"`  // tours únicos somando todas as ilhas
    Migrou           bool       `json:"migrou"`            // true na geração em que houve migração
    Migracoes        []Migracao `json:"migracoes,omitempty"` // de→para nesta geração (animação + cor de gene)
    RefUnicaDist     float64    `json:"refUnicaDist,omitempty"` // melhor da pop única nesta geração (se comparando)
}

type Migracao struct {
    De         int   `json:"de"`
    Para       int   `json:"para"`
    MigranteTour []int `json:"migranteTour,omitempty"` // tour do migrante (destaca o gene migrante no destino)
}

type MultiResult struct {
    Geracoes         int          `json:"geracoes"`
    MelhorGlobalTour []int        `json:"melhorGlobalTour"`
    MelhorGlobalDist float64      `json:"melhorGlobalDist"`
    IlhaVencedora    int          `json:"ilhaVencedora"`
    HistGlobal       []float64    `json:"histGlobal"`       // melhor global por geração
    HistIlhas        [][]float64  `json:"histIlhas"`        // [ilha][geracao] melhor da ilha
    HistDiversidade  []int        `json:"histDiversidade"`  // diversidade global por geração
    GeracoesMigracao []int        `json:"geracoesMigracao"` // gerações em que houve migração
    HistRefUnica     []float64    `json:"histRefUnica,omitempty"` // baseline pop única (se comparando)
    MelhorRefUnicaDist float64    `json:"melhorRefUnicaDist,omitempty"`
    Cfg              MultiConfig  `json:"cfg"`
}
```

### 3.4 Orquestração (lockstep paralelo)

```
Treinar(progressCh chan<- MultiStep, cfg MultiConfig, matDist, matDur):
    sanitiza cfg
    rng base com seed (deriva 1 sub-rng por ilha, determinístico)
    ilhas[i].pop = gerarPopulacaoInicial(...)   // i = 0..NumIlhas-1
    melhorGlobal = +inf

    para g em 1..MaxGeracoes:
        // 1) evolui todas as ilhas concorrentemente (goroutines + WaitGroup)
        para cada ilha i em paralelo:
            ilhas[i].pop = tsp.EvoluirUmaGeracao(ilhas[i].pop, cfg.GA, rngIlha[i], matDist, matDur)
        WaitGroup.Wait()

        // 2) migração em anel a cada IntervaloMigracao gerações (g % intervalo == 0)
        migrou = (g % IntervaloMigracao == 0)
        se migrou:
            coleta os NumMigrantes melhores de cada ilha (cópias)
            para cada ilha i: insere migrantes da ilha (i-1 mod N) substituindo os PIORES de i
            registra Migracao{de:i, para:(i+1)%N}

        // 3) estatísticas + melhor global
        para cada ilha: calcula melhor/média/diversidade
        atualiza melhorGlobal (clona se melhorou; nunca piora)

        // 4) emite MultiStep
        progressCh <- MultiStep{...}

    devolve MultiResult{...}
```

**Migração em anel:** ilha *i* envia seus `NumMigrantes` melhores para a ilha
*(i+1) mod N*; no destino, os migrantes **substituem os piores** indivíduos.
Migrantes são **cópias** (a ilha de origem mantém os seus). A coleta dos melhores
de todas as ilhas é feita **antes** de qualquer inserção, para a migração ser
simultânea ("dança de cadeiras") e não encadeada.

**Determinismo:** cada ilha recebe um `*rand.Rand` derivado da seed base
(ex: `seed + i`), garantindo reprodutibilidade independente do escalonamento das
goroutines (a estrutura lockstep evita corrida nos dados compartilhados).

### 3.5 Baseline de população única (comparativo)

Quando `CompararPopUnica = true`, o orquestrador mantém **uma população única
extra** de tamanho `NumIlhas × TamIlha` (mesmo total de indivíduos das ilhas
somadas), com a mesma seed-base e os mesmos params de GA, evoluída em lockstep
junto com as ilhas (mais uma goroutine na barreira de cada geração). Ela **não
participa de migração** — é a testemunha "sem multipopulacional". Seu melhor por
geração vai em `RefUnicaDist`/`HistRefUnica`. Reusa diretamente
`tsp.EvoluirUmaGeracao`, deixando a comparação justa: mesmo orçamento de
indivíduos e avaliações, única diferença = ter ilhas + migração.

## 4. Cidades — preset `triangulo20`

Novo preset (no pacote `tsp`, ao lado dos demais, reusado pelo `tspmulti`) com
as 20 cidades do mapa do slide. Coordenadas reais (aproximadas; OSRM resolve a
estrada real). Origem/depot = Uberlândia (maior cidade, hub natural).

> **Nota sobre numeração:** a coluna "#" abaixo é o rótulo do mapa do slide. No
> código os IDs são 0-indexados e o **depot recebe ID 0** (convenção do `tsp`,
> que ancora o tour na cidade 0). Logo, no preset Uberlândia vira ID 0 e as
> demais seguem; a tabela serve só para rastrear a origem de cada cidade.

| # | Cidade | Lat | Lng |
|---|--------|-----|-----|
| 1 | Tupaciguara | -18.5917 | -48.7053 |
| 2 | Araguari | -18.6486 | -48.1872 |
| 3 | Monte Carmelo | -18.7264 | -47.4986 |
| 4 | Patos de Minas | -18.5789 | -46.5181 |
| 5 | Uberlândia (depot) | -18.9128 | -48.2755 |
| 6 | Ituiutaba | -18.9686 | -49.4650 |
| 7 | Iturama | -19.7281 | -50.1958 |
| 8 | Prata | -19.3072 | -48.9244 |
| 9 | Campina Verde | -19.5378 | -49.4869 |
| 10 | Campo Florido | -19.7656 | -48.5703 |
| 11 | Itapagipe | -19.9047 | -49.3789 |
| 12 | Indianópolis | -19.0319 | -47.9181 |
| 13 | Iraí de Minas | -18.9869 | -47.4592 |
| 14 | Nova Ponte *(escolhida — trocável)* | -19.1419 | -47.6803 |
| 15 | Serra do Salitre | -19.1097 | -46.6914 |
| 16 | Araxá | -19.5933 | -46.9406 |
| 17 | Perdizes | -19.3525 | -47.2914 |
| 18 | Conceição das Alagoas | -19.9142 | -48.3878 |
| 19 | Campos Altos | -19.6953 | -46.1719 |
| 20 | Santa Juliana | -19.3119 | -47.5331 |

A matriz de distâncias é montada pelo pipeline já existente
`POST /api/tsp/distancias` (haversine, OSRM ou euclidiana), reaproveitado por
ambos os trabalhos.

## 5. API (espelha o padrão dos outros módulos)

| Rota | Método | Função |
|------|--------|--------|
| `/api/tspmulti/config` | POST | Salva a `MultiConfig` ativa |
| `/api/tspmulti/train` | GET (SSE) | Roda o AG multi-populacional, emite `MultiStep` por geração |
| `/api/tspmulti/reset` | POST | Limpa estado |
| `/api/tspmulti/result` | GET | Devolve o último `MultiResult` |

Reusa `/api/tsp/preset?name=triangulo20` (carregar cidades) e
`/api/tsp/distancias` (calcular matriz) antes de treinar — mesmo fluxo do TSP.

## 6. Frontend — view `TspMultiView` ("GA · TSP Multi-ilhas")

Nova entrada no `Sidebar` na seção "Algoritmo Genético", depois de "TSP
Comparativo". Novo `ViewId` `tsp-multi`, tipos em `api/types.ts`, consumo via
`apiSSE` (mesmo helper do TSP).

**Identidade de cor por ilha:** cada ilha recebe uma cor fixa (paleta de N
cores). Essa cor é usada de ponta a ponta — borda do small-multiple, linha na
curva de convergência, token de migração e destaque do gene migrante — para o
olho amarrar "ilha → cor" em todos os painéis.

Componentes (visual rico + pedagógico):

1. **Mapa** (reusa `TspMap`/leaflet): melhor rota **global**, ancorada no depot.
2. **Small multiples**: um mini-painel por ilha (na cor da ilha) mostrando a rota
   atual daquela ilha — dá pra ver cada ilha "se especializando" num caminho
   diferente.
3. **Convergência multi-linha** (reusa/estende `GaChart`): uma linha por ilha (na
   cor da ilha) + linha grossa do melhor **global**. Quando `CompararPopUnica`,
   sobrepõe a linha **pontilhada cinza da população única** — o contraste mostra
   a multi mergulhando abaixo enquanto a única empaca num platô (mínimo local).
   - **Marcadores verticais** nas gerações de migração.
   - **Anotações de "salto"**: quando uma migração é seguida de melhora do
     global dentro de poucas gerações, rotular o ponto ("migração → melhorou").
4. **Animação de migração**: na geração de migração, um **token colorido** (cor
   da ilha de origem) viaja entre os painéis no sentido do anel — a "dança de
   cadeiras". No painel de destino, o **gene migrante** (rota recém-chegada,
   `MigranteTour`) pisca na cor da origem por alguns frames.
5. **Medidor de diversidade**: sparkline/barra da diversidade global por geração,
   com realce do **salto logo após cada migração** (o efeito que o slide promete
   — injeção de diversidade evita estagnação).
6. **Painel "melhor de todas"**: card de destaque com a melhor distância global,
   **qual ilha** a detém (na cor dela) e **há quantas gerações não melhora**
   (contador de estagnação) — torna visível quando a migração "destrava" o ótimo.
7. **Tabela por ilha**: melhor distância/fitness e diversidade de cada ilha, com
   destaque na ilha vencedora.
8. **Controles**: nº de ilhas, tamanho da ilha, intervalo de migração, nº de
   migrantes, gerações, **toggle "comparar com população única"**, + params de GA
   (seleção, cruzamento, mutação, probs, elitismo) e modo de distância
   (haversine/OSRM).

## 7. Tratamento de erros

- Treinar exige cidades carregadas + matriz calculada; senão, SSE encerra com
  evento de erro (mesma convenção do TSP: "matriz de distâncias não calculada").
- `sanitizar` corrige configs inválidas para os mínimos (seção 3.2) em vez de
  falhar silenciosamente.
- Reset e troca de preset zeram a matriz, forçando recálculo antes do próximo
  treino (consistente com o TSP atual).

## 8. Testes (Go, pacote `tspmulti`)

1. **Migração em anel:** após uma migração, o melhor indivíduo da ilha *i*
   aparece na ilha *(i+1) mod N* e o pior anterior do destino foi removido.
2. **Migração simultânea:** a coleta dos melhores ocorre antes das inserções
   (uma migração não "vaza" para a próxima na mesma rodada).
3. **Monotonicidade do global:** `MelhorGlobalDist` nunca aumenta ao longo das
   gerações.
4. **Determinismo:** mesma seed ⇒ mesmo `MultiResult`, independente do
   escalonamento das goroutines.
5. **Regressão do `tsp`:** após extrair `EvoluirUmaGeracao`, `tsp.Treinar`
   produz resultado idêntico com a mesma seed (o Trabalho 11 não regride).
6. **Baseline comparativo:** com `CompararPopUnica`, o `HistRefUnica` tem o mesmo
   comprimento de `HistGlobal` e a pop única usa exatamente `NumIlhas × TamIlha`
   indivíduos e a mesma seed-base (comparação justa).

## 9. Fora de escopo (YAGNI)

- Topologias de migração além do anel (estrela, all-to-all, aleatória).
- Cluster físico / execução distribuída entre máquinas (o enunciado diz "de
  preferência"; goroutines satisfazem o requisito de threads).
- Migração assíncrona / ilhas com nº de gerações independentes.
- Edição manual da matriz de distâncias da tabela do slide (usamos coords).
