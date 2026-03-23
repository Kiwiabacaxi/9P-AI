# MLP Letras — Reconhecimento de Letras A–Z com Backpropagation

Aplicação do algoritmo MLP (Multilayer Perceptron) para reconhecimento das 26 letras do alfabeto (A–Z), usando a mesma arquitetura e regras de backpropagation do **Desafio MLP**, agora com entradas reais de pixels 5×7.

---

## Diferença em relação ao Desafio MLP

| | Desafio MLP | MLP Letras |
|---|---|---|
| Problema | Exemplo numérico do slide | Reconhecimento de letras |
| Entradas | 3 valores fixos | 35 pixels (grade 5×7 bipolar) |
| Saídas | 3 classes | 26 classes (A–Z) |
| Neurônios ocultos | 2 | 15 |
| Pesos iniciais | Fixos (do slide) | Aleatórios [-0.5, +0.5] |
| Padrões | 3 | 26 (um por letra) |

O **algoritmo é idêntico** — mesma fórmula de forward, mesma backpropagation, mesmo critério de parada.

---

## Arquitetura

```
Entradas (35)  →  Camada Oculta (15 neurônios)  →  Saída (26 neurônios)
 pixels 5×7        tanh                              tanh, one-hot A–Z
```

| Parâmetro | Valor |
|---|---|
| Entradas | 35 (grade 5×7 pixels, bipolar -1/+1) |
| Neurônios ocultos | 15 |
| Saídas | 26 (uma por letra A–Z) |
| Ativação | tanh em todas as camadas |
| Taxa de aprendizado (α) | 0.01 |
| Critério de parada | Erro total ≤ 0.5 ou 50.000 ciclos |
| Codificação target | One-hot: +1 na letra correta, -1 nas outras 25 |

---

## Representação das Letras

Cada letra é uma grade **5 colunas × 7 linhas** em bipolar:
- `+1` (ou `1`) → pixel aceso (`█`)
- `-1` (ou `-1`) → pixel apagado (`·`)

Exemplo da letra **A**:
```
· █ █ █ ·
█ · · · █
█ · · · █
█ █ █ █ █
█ · · · █
█ · · · █
█ · · · █
```

---

## Estrutura dos Arquivos

```
MLP Letras/
├── main.go   — algoritmo MLP + dataset das 26 letras
├── tui.go    — interface Bubble Tea: menu, treino, resultado, teste interativo
├── go.mod    — módulo mlpLetras
└── go.sum
```

### `main.go`

| Função/Tipo | Descrição |
|---|---|
| `MLP` | Pesos `v[35][15]`, `v0[15]`, `w[15][26]`, `w0[26]` |
| `ForwardResult` | `zin[15]`, `z[15]`, `yin[26]`, `y[26]` |
| `BackwardResult` | Todos os deltas e incrementos Δ |
| `letrasDataset()` | 26 padrões 5×7 em float64 bipolar |
| `targetVetor(idx)` | One-hot: +1.0 na posição idx, -1.0 nas demais |
| `inicializarPesos()` | Pesos aleatórios em [-0.5, +0.5] |
| `forward()` | Calcula z e y para entrada de 35 pixels |
| `backward()` | Calcula δ e Δpesos para um padrão |
| `treinarMLP()` | Loop completo sobre as 26 letras |
| `classificar()` | Argmax de y → índice da letra reconhecida |
| `formataLetraGrid()` | Renderiza grade 5×7 com █ e · coloridos |

### `tui.go` — Estados da Interface

| Estado | Descrição |
|---|---|
| `stateMenu` | Menu principal |
| `stateTrainingDone` | Resultado do treino: acurácia, erro, pesos |
| `stateResult` | Navega pelas 26 letras — grid + classificação |
| `stateTest` | Grade 5×7 interativa — pinta e vê a classificação ao vivo |

---

## Como Executar

```bash
cd "MLP Letras"
go run .
```

### Navegação

| Tecla | Ação |
|---|---|
| `↑` / `↓` | Navegar menu |
| `enter` | Selecionar |
| `←` / `→` | Mudar letra (tela de resultado) |
| `↑↓←→` | Mover cursor na grade (teste interativo) |
| `space` | Acender/apagar pixel (teste interativo) |
| `r` | Resetar grade (teste interativo) |
| `esc` / `q` | Voltar ao menu |
| `ctrl+c` | Sair |

---

## Modo Teste Interativo

O modo mais interessante: uma **grade 5×7 em branco** onde você:

1. Move o cursor com as setas
2. Pressiona `space` para acender/apagar pixels
3. A rede classifica em **tempo real** conforme você desenha
4. Mostra o top-3 de letras mais prováveis com as ativações `y`

Permite testar letras com ruído, variações e desenhos livres — mostrando como a rede generaliza (ou não) além dos padrões de treino.

---

## Notas sobre Convergência

- Com 26 letras e 15 neurônios ocultos, o treino costuma convergir em **algumas centenas a poucos milhares de ciclos**
- Pesos aleatórios → resultado varia a cada execução
- Se não convergir no limite de 50.000 ciclos, a tela mostra a acurácia obtida
- Para aumentar a chance de convergência: aumentar `N_HID` em `main.go`

---

## Relação com os Outros Trabalhos

| Trabalho | Rede | Entradas | Letras |
|---|---|---|---|
| Trab 03 | MADALINE (Regra Delta) | 35 pixels | A–M (13) |
| MLP Letras | MLP + Backpropagation | 35 pixels | A–Z (26) |
| Desafio MLP | MLP + Backpropagation | 3 valores | — |

MLP Letras usa o mesmo dataset de pixels do Trab 03, mas com **algoritmo diferente** (backpropagation em vez de regra delta) e **cobertura completa** do alfabeto.
