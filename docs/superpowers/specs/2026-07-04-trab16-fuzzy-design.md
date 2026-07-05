# Trabalho 16 — Qualidade da Água com Lógica Fuzzy

**Data:** 2026-07-04 · **Branch:** `trab16-fuzzy`
**Fonte:** apostila de Fuzzy (Jafelice · Barros · Bassanezi), exemplo 2.7.3 "Qualidade da Água"
**Aulas:** 17–19 (Lógica Fuzzy)

## Objetivo

Sistema de inferência fuzzy **Mamdani** que classifica a potabilidade da água a
partir de três entradas (cor aparente, pH, turbidez — limites da SABESP) e devolve
a qualidade em [0,1] com termos *boa / adequada / inadequada*. Implementação Go
didática (sem toolbox), com visualização web que mostra o **pipeline inteiro ao
vivo**: fuzzificação → ativação das 45 regras → agregação → defuzzificação por
centroide — mais uma **superfície 3D** da saída.

Validação canônica da apostila: **cor 15 UH, pH 7, turbidez 0 UT → Q = 0.6
(adequada)**.

## Não-objetivos

- Sem treino/evolução — inferência é instantânea, **não há SSE** (só REST).
- Sem WebAssembly (padrão dos trabalhos recentes: backend local).
- Sem editor de regras/trapézios na UI (as regras são as das Tabelas 2.6–2.8).

## Modelo fuzzy

**Funções de pertinência** — trapézios `(a, b, c, d)` (0 até `a`, sobe até `b`,
platô até `c`, desce até `d`). Os cantos foram fixados para que os cruzamentos
em μ=0.5 caiam exatamente nos limites da SABESP citados na apostila:

| Variável (domínio) | Termo | (a, b, c, d) |
|---|---|---|
| Cor aparente (0–30 UH) | boa | (0, 0, 4, 6) |
| | adequada | (4, 6, 14, 16) |
| | inadequada | (14, 16, 30, 30) |
| pH (0–14) | inadequado baixo | (0, 0, 5.75, 6.25) |
| | adequado baixo | (5.75, 6.25, 6.25, 6.75) |
| | bom | (6.25, 6.75, 8.25, 8.75) |
| | adequado alto | (8.25, 8.75, 9.75, 10.25) |
| | inadequado alto | (9.75, 10.25, 14, 14) |
| Turbidez (0–10 UT) | boa | (0, 0, 0.8, 1.2) |
| | adequada | (0.8, 1.2, 4.6, 5.4) |
| | inadequada | (4.6, 5.4, 10, 10) |
| Qualidade (0–1, saída) | inadequada | (0, 0, 0.35, 0.5) |
| | adequada | (0.35, 0.5, 0.7, 0.85) |
| | boa | (0.7, 0.85, 1, 1) |

Cruzamentos: cor 5 e 15 · pH 6, 6.5, 8.5 e 10 · turbidez 1 e 5.

**Base de regras** — as 45 regras das Tabelas 2.6/2.7/2.8 da apostila,
declaradas como **dados** (3 matrizes 5×3: aparência × pH × turbidez → saída).

**Inferência Mamdani** — AND = `min` (força da regra = min dos 3 antecedentes),
implicação = `min` (recorta o termo de saída na força), agregação = `max`
(envelope), defuzzificação = **centroide** discretizado em 1001 pontos de [0,1].
Classe final = termo de saída com maior μ no centroide.

**Por que o exemplo dá 0.6 exato:** cor 15 fica no cruzamento adequada/inadequada
(0.5/0.5), pH 7 → bom (1), turbidez 0 → boa (1). Disparam 2 regras, ambas →
*adequada* com força 0.5. O trapézio de *adequada* recortado em 0.5 é simétrico
em torno de (0.35+0.85)/2 = **0.6** → centroide 0.6.

## Backend — `web/server/fuzzy/fuzzy.go` (+ `fuzzy_test.go`)

Pacote **stateless** (sem mutex/estado global — diferente dos AGs). API:

- `GET /api/fuzzy/meta` — definição completa: variáveis de entrada/saída (nome,
  unidade, domínio, termos com trapézio e cor sugerida) + as 45 regras. O
  frontend **desenha tudo a partir disso** — zero lógica fuzzy duplicada em TS.
- `POST /api/fuzzy/evaluate {cor, ph, turbidez}` — trace completo:
  pertinências por termo das 3 entradas, força das 45 regras, força máxima por
  consequente, curva agregada amostrada (~121 pts, com as parcelas por termo),
  centroide, classe e nº de regras ativas.
- `GET /api/fuzzy/surface?eixoX=ph&eixoY=turbidez&cor=15&ph=7&turbidez=0` —
  grade 41×41 (`xs`, `ys`, `z`) para a superfície 3D; eixos escolhíveis, a
  variável restante vem fixa da query.

Testes Go: exemplo da apostila (0.6 ± 0.01, classe adequada), água perfeita
(→ boa), esgoto (→ inadequada), cobertura das 45 regras, trapézio (bordas,
platô, rampas), cobertura do domínio (∀x, algum termo com μ > 0), superfície
dentro de [0,1].

## Frontend — `web/frontend/src/views/FuzzyView.tsx`

Registro padrão: `App.tsx` (`viewComponents.fuzzy`), `api/types.ts`
(`ViewId += 'fuzzy'` + tipos `Fuzzy*`), `Sidebar.tsx` (seção nova
**"Lógica Fuzzy"** com item **"Fuzzy · Água"** 💧), `TopBar.tsx`
(`VIEW_INFO.fuzzy`, sem leitor de status — stateless).

Layout (cima → baixo), estética mission-control existente:

1. **Controle:** 3 sliders (cor 0–30 · pH 0–14 · turbidez 0–10) + presets
   (📖 Apostila 15/7/0 com selo "= 0.6 validado", 🚰 Torneira, 🏊 Piscina,
   🌧 Pós-chuva, 🏭 Rio poluído) + MetricCards: **Q**, **classe** (cor por
   termo), **regras ativas**.
2. **Fuzzificação:** 3 gráficos Recharts com os trapézios de cada entrada,
   linha de referência no valor atual e os μ correntes destacados.
   `evaluate` com debounce ~80 ms.
3. **Regras:** 3 tabelas 5×3 (uma por termo de aparência); célula = consequente,
   fundo aceso com intensidade ∝ força; clique abre o **inspetor da regra**
   (SE … E … ENTÃO …, μ de cada antecedente e o `min`).
4. **Saída:** termos recortados + envelope agregado preenchido + linha do
   centroide; Q grande + badge da classe.
5. **Superfície 3D** (Plotly, padrão RastriginView): seletor do par de eixos,
   3ª variável presa ao slider; debounce ~250 ms.
6. **Card didático:** fuzzificação, Mamdani (min/max), centroide — com fórmulas.

Cores dos termos: boa `#3ddc84` · adequada `#ffb020` · inadequada `#ff4d6d`
(consistentes em todos os gráficos, tabelas e badges).

## Integração e validação

- `AboutView.tsx`: Trabalho 16 sai de `pendente` → menu "Fuzzy · Água",
  pkg `fuzzy`; ajustar rodapé "15 dos 16".
- `README.md`: linha 16 da tabela + estrutura (`fuzzy/` na lista de pacotes).
- Teste de interface no navegador: abrir a view, clicar no preset da apostila
  e conferir Q = 0.60/adequada na tela.
- `go test ./fuzzy/` + `npm run build` (tsc) verdes antes do commit final.
