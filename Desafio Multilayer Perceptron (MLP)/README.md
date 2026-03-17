# Desafio MLP — Multilayer Perceptron com Backpropagation

Implementação em Go do exemplo numérico exato dos slides da **Aula 05** do Prof. Jefferson — MLP 3→2→3 com backpropagation, com TUI interativa neon/cyberpunk.

---

## Arquitetura

```
Entradas (3) → Camada Oculta (2 neurônios) → Saída (3 neurônios)
```

| Parâmetro       | Valor                          |
|-----------------|--------------------------------|
| Entradas        | 3                              |
| Neurônios ocultos | 2                            |
| Saídas          | 3                              |
| Ativação        | tanh em todas as camadas       |
| Taxa de aprendizado (α) | 0.01               |
| Critério de parada | Erro total ≤ 0.001 ou 50000 ciclos |

Os **pesos iniciais são exatos do slide** (não aleatórios), garantindo reprodutibilidade.

---

## Os 3 Padrões de Treinamento

| Padrão | Entrada `x`         | Target `t`    |
|--------|---------------------|---------------|
| 1      | `[+1.0, +0.5, -1.0]` | `[+1, -1, -1]` |
| 2      | `[+1.0, +0.5, +1.0]` | `[-1, +1, -1]` |
| 3      | `[+1.0, -0.5, -1.0]` | `[-1, -1, +1]` |

---

## Fórmulas Implementadas

**Forward Pass:**
```
zin_j = v0_j + Σ x_i · v[i][j]     (pré-ativação oculta)
z_j   = tanh(zin_j)                  (saída oculta)
yin_k = w0_k + Σ z_j · w[j][k]     (pré-ativação saída)
y_k   = tanh(yin_k)                  (saída final)
```

**Erro:**
```
E = ½ · Σ (t_k − y_k)²
```

**Backpropagation:**
```
δ_k    = (t_k − y_k) · (1+y_k)(1−y_k)      (erro saída)
δin_j  = Σ δ_k · w[j][k]                    (propaga para oculta)
δ_j    = δin_j · (1+z_j)(1−z_j)             (erro oculta)

Δw_jk  = α · δ_k · z_j                      (update saída)
Δw0_k  = α · δ_k
Δv_ij  = α · δ_j · x_i                      (update oculta)
Δv0_j  = α · δ_j
```

---

## Estrutura dos Arquivos

```
Desafio MLP/
├── main.go   — algoritmo MLP: tipos, forward, backward, treino
├── tui.go    — interface Bubble Tea: menu, animação, slides, walkthrough
├── go.mod    — módulo desafioMLP
└── go.sum
```

### `main.go`

| Função/Tipo | Descrição |
|---|---|
| `MLP` | Struct com pesos `v`, `v0`, `w`, `w0` |
| `ForwardResult` | `zin`, `z`, `yin`, `y` calculados no forward |
| `BackwardResult` | Todos os deltas e incrementos Δ |
| `TrainingStep` | Snapshot de um padrão num ciclo (para animação) |
| `ResultadoTreino` | Resultado completo: convergência, pesos, histórico |
| `inicializarPesosSlide()` | Pesos exatos do slide da Aula 05 |
| `forward()` | Calcula z e y para uma entrada x |
| `backward()` | Calcula δ e Δpesos para um padrão |
| `atualizarPesos()` | Aplica os Δ nos pesos |
| `treinarMLP()` | Loop completo até convergência |
| `classificar()` | Argmax da saída y |

### `tui.go` — Estados da Interface

| Estado | Descrição |
|---|---|
| `stateMenu` | Menu principal com 6 opções |
| `stateTraining` | Diagrama animado de neurônios (forward → backprop alternados) |
| `stateTrainingDone` | Pesos finais + curva de erro ASCII |
| `stateSlide` | 6 slides explicativos com revelação progressiva |
| `stateResult` | Tabela com os 3 padrões e saídas da rede treinada |
| `stateWalkthrough` | Passo a passo manual — 22 sub-passos por padrão |
| `stateTest` | Inserção manual de x1, x2, x3 e classificação em tempo real |

---

## Como Executar

```bash
cd "Desafio MLP"
go run .
```

### Navegação

| Tecla | Ação |
|---|---|
| `↑` / `↓` ou `k` / `j` | Navegar menu |
| `enter` | Selecionar |
| `→` / `←` | Próximo/anterior passo (walkthrough e slides) |
| `tab` | Próximo campo (teste manual) |
| `esc` / `q` | Voltar ao menu |
| `ctrl+c` | Sair |

---

## Modo Passo a Passo (Walkthrough)

O modo mais didático do programa. Mostra **cada conta individualmente**:

1. `zin₁`, `z₁` (neurônio oculto 1)
2. `zin₂`, `z₂` (neurônio oculto 2)
3. `yin₁`, `y₁`, `yin₂`, `y₂`, `yin₃`, `y₃` (3 saídas)
4. `E = ½Σ(t−y)²`
5. `δ₁`, `δ₂`, `δ₃` (camada de saída)
6. `δin₁`, `δin₂` (propagação para oculta)
7. `δ₁_oc`, `δ₂_oc` (erro oculta)
8. Todos os `Δw` e `Δv` com contas explícitas
9. Pesos novos após o update

Ao terminar o Padrão 1, os pesos são atualizados e o Padrão 2 começa com os pesos novos — exatamente como o algoritmo real funciona.

---

## Convergência

Com os pesos iniciais do slide e α=0.01, a rede converge em **~27.000 ciclos**. O limite é 50.000 ciclos. Após convergência, todos os 3 padrões são classificados corretamente (saídas acima de 0.98 em módulo).
