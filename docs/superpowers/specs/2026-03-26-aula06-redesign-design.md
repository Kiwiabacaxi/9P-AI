# Aula 06 — Redesign MLP Funcoes + MLP Ortogonal

## Contexto

A Aula 06 (MLP Part 2) do Prof. Jefferson Beethoven Martins cobre dois temas:
1. **Aproximacao de funcoes** — MLP aprende a reproduzir funcoes matematicas (sin(x)*sin(2x), etc.)
2. **Vetores bipolares ortogonais** — Classificacao A-Z usando distancia euclidiana em vez de threshold

A versao web atual tem problemas:
- **MLP Funcoes**: graficos lado a lado (pequenos), sem configuracao de hiperparametros, sem reset
- **MLP Ortogonal**: pagina quase vazia, sem visualizacao dos conceitos (vetores ortogonais, distancia euclidiana), sem configuracao

## Objetivo

Redesenhar ambas as paginas para:
- Permitir customizacao de hiperparametros (estilo img_regression)
- Adicionar visualizacoes educacionais dos conceitos do slide
- Melhorar layout e experiencia interativa

---

## MLP Funcoes

### Configuracao (estilo img_regression)

Painel de cards com dropdowns:

| Parametro | Opcoes | Default |
|-----------|--------|---------|
| Funcao | sin(x)*sin(2x), sin(x), x², x³ | sin(x)*sin(2x) |
| Neuronios ocultos | 50, 100, 200, 300 | 200 |
| Learning Rate (alfa) | 0.001, 0.005, 0.01, 0.02 | 0.005 |
| Max Epocas | 10000, 50000, 100000 | 100000 |

Botoes: **TREINAR** + **RESETAR**

### Layout dos graficos

Graficos **empilhados verticalmente** (largura total) em vez de lado a lado:

1. **Funcao Original vs Aproximacao** — grafico grande, largura total, enfase visual
2. **Curva de Erro (escala log)** — abaixo, mesma largura

Isso da mais espaco para ver a rede convergindo em tempo real.

### Metricas

Manter os 3 cards: Ciclos, Erro, Status.

### Backend

`mlpfunc.Treinar()` passa a aceitar um struct de config em vez de usar constantes:

```go
type FuncConfig struct {
    Funcao   string  `json:"funcao"`
    NHid     int     `json:"nHid"`
    Alfa     float64 `json:"alfa"`
    MaxCiclo int     `json:"maxCiclo"`
}
```

O endpoint POST `/api/mlpfunc/train` recebe o config no body (ou query params para compatibilidade SSE).

---

## MLP Ortogonal

### Configuracao (estilo img_regression)

Painel de cards com dropdowns:

| Parametro | Opcoes | Default |
|-----------|--------|---------|
| Neuronios ocultos | 10, 15, 20, 30 | 15 |
| Learning Rate (alfa) | 0.005, 0.01, 0.02, 0.05 | 0.01 |
| Max Epocas | 10000, 50000, 100000 | 50000 |

Botoes: **TREINAR REDE** + **RESETAR**

### Secao 1: Construcao dos Vetores Bipolares Ortogonais

Visualizacao educacional da expansao recursiva (Fausett 1994):

- Tabs ou botoes para navegar entre passos: **Passo 0** (2 vetores, 2 dims) → **Passo 1** (4 vetores, 4 dims) → ... → **Passo 4** (32 vetores, 32 dims)
- Tabela com celulas coloridas:
  - **Amarelo**: valores do vetor original (v,v)
  - **Verde**: valores invertidos (v,-v)
- Headers das colunas mostram a notacao do slide: (a,a), (a,-a), (b,b), (b,-b)
- Esta secao eh estatica (nao depende de treinamento)

### Secao 2: Mapa Letra → Vetor Ortogonal

Grid compacto mostrando cada uma das 26 letras:
- Mini-grid 5x7 da letra (visual bipolar: preenchido/vazio)
- Vetor ortogonal de 32 dims atribuido a essa letra (compacto, talvez com +/- coloridos)

Dados vem de `/api/mlport/dataset` que ja retorna tudo isso.

### Metricas

3 cards: Ciclos, Acuracia (%), Status.

### Curva de erro

Grafico de erro (escala log), largura total.

### Secao 3: Demo Distancia Euclidiana (aparece apos treinar)

Tabela exatamente como no slide do professor:
- **Coluna esquerda**: "saida da rede" — valores reais da saida tanh para um padrao
- **Colunas a, b, c, ...**: vetores ortogonais target de cada letra candidata
- **Linha inferior**: distancia euclidiana calculada para cada letra
- **Menor distancia** destacada em amarelo

Dropdown para selecionar qual letra do dataset usar como exemplo.

Formula: D = sqrt( sum_k (t_k - y_k)^2 )

### Teste Interativo

Grid 5x7 clicavel (como hoje) + botao CLASSIFICAR + botao LIMPAR.

Resultado mostra:
- Letra reconhecida (grande)
- Top-5 candidatos com barras de distancia
- A distancia euclidiana de cada candidato

### Backend

`mlport.Treinar()` passa a aceitar config:

```go
type OrtConfig struct {
    NHid     int     `json:"nHid"`
    Alfa     float64 `json:"alfa"`
    MaxCiclo int     `json:"maxCiclo"`
}
```

Novo endpoint ou extensao do existente: retornar a saida bruta da rede para um padrao especifico (para alimentar a tabela de distancia euclidiana). O `Classificar()` ja retorna distancias, mas precisa tambem retornar o vetor de saida da rede (valores tanh).

---

## Mudancas no Backend (resumo)

1. **mlpfunc/mlpfunc.go**: converter constantes NHid, alfa, maxCiclo em parametros de `Treinar()`
2. **mlport/mlport.go**: converter constantes NHid, alfa, maxCiclo em parametros de `Treinar()`
3. **mlport/mlport.go**: `Classificar()` retornar tambem o vetor de saida bruta `y [NOrt]float64`
4. **server/main.go**: endpoints de train recebem config via query params (para SSE) ou body
5. **server/wasm/main.go**: mesmas mudancas para versao WASM

## Mudancas no Frontend (resumo)

1. **view-mlpfunc**: adicionar painel de config, empilhar graficos, botao reset
2. **view-mlport**: adicionar painel de config, 3 secoes visuais, botao reset
3. **JS mlpfunc**: enviar config na request, logica de reset
4. **JS mlport**: enviar config, renderizar tabelas de vetores/distancias, logica de reset
