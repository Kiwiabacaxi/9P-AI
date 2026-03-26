# Desafio MLP — Multilayer Perceptron com Backpropagation

Implementacao do exemplo numerico exato dos slides da **Aula 05** — MLP 3->2->3 com backpropagation, com TUI interativa.

## Arquitetura

```
Entradas (3) -> Camada Oculta (2 neuronios) -> Saida (3 neuronios)
```

| Parametro | Valor |
|-----------|-------|
| Entradas | 3 |
| Neuronios ocultos | 2 |
| Saidas | 3 |
| Ativacao | tanh em todas as camadas |
| Taxa de aprendizado (a) | 0.01 |
| Criterio de parada | Erro total <= 0.001 ou 50000 ciclos |

Os **pesos iniciais sao exatos do slide** (nao aleatorios), garantindo reprodutibilidade.

## Os 3 Padroes de Treinamento

| Padrao | Entrada `x` | Target `t` |
|--------|-------------|------------|
| 1 | `[+1.0, +0.5, -1.0]` | `[+1, -1, -1]` |
| 2 | `[+1.0, +0.5, +1.0]` | `[-1, +1, -1]` |
| 3 | `[+1.0, -0.5, -1.0]` | `[-1, -1, +1]` |

## Formulas Implementadas

**Forward Pass:**
```
zin_j = v0_j + Sum x_i * v[i][j]     (pre-ativacao oculta)
z_j   = tanh(zin_j)                   (saida oculta)
yin_k = w0_k + Sum z_j * w[j][k]     (pre-ativacao saida)
y_k   = tanh(yin_k)                   (saida final)
```

**Backpropagation:**
```
d_k    = (t_k - y_k) * (1+y_k)(1-y_k)      (erro saida)
din_j  = Sum d_k * w[j][k]                  (propaga para oculta)
d_j    = din_j * (1+z_j)(1-z_j)             (erro oculta)
```

## Como executar

```bash
cd cli
go run ./desafio-mlp
```

### Navegacao

| Tecla | Acao |
|---|---|
| `cima` / `baixo` ou `k` / `j` | Navegar menu |
| `enter` | Selecionar |
| `direita` / `esquerda` | Proximo/anterior passo (walkthrough e slides) |
| `tab` | Proximo campo (teste manual) |
| `esc` / `q` | Voltar ao menu |
| `ctrl+c` | Sair |

## Estados da TUI

| Estado | Descricao |
|---|---|
| Menu | Menu principal com 6 opcoes |
| Training | Diagrama animado de neuronios (forward -> backprop alternados) |
| TrainingDone | Pesos finais + curva de erro ASCII |
| Slides | 6 slides explicativos com revelacao progressiva |
| Result | Tabela com os 3 padroes e saidas da rede treinada |
| Walkthrough | Passo a passo manual — 22 sub-passos por padrao |
| Test | Insercao manual de x1, x2, x3 e classificacao em tempo real |

## Convergencia

Com os pesos iniciais do slide e a=0.01, a rede converge em **~27.000 ciclos**. O limite e 50.000 ciclos.

## Arquivos

| Arquivo | Conteudo |
|---------|----------|
| `main.go` | Algoritmo MLP: tipos, forward, backward, treino |
| `tui.go` | Interface Bubble Tea: menu, animacao, slides, walkthrough |
