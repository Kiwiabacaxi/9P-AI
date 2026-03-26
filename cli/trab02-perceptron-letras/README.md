# Trabalho 02 (Parte 1) — Perceptron: Reconhecimento de Letras

## Descricao

A rede reconhece as letras **A** e **B** representadas em matrizes **7x7** (49 pixels).

```
Letra A          Letra B
. # # # # # .    # # # # # . .
# . . . . . #    # . . . . # .
# . . . . . #    # . . . . # .
# # # # # # #    # # # # # . .
# . . . . . #    # . . . . # .
# . . . . . #    # . . . . # .
# . . . . . #    # # # # # . .
```

## Hebb vs Perceptron

| | Hebb | Perceptron |
|---|------|-----------|
| Atualizacao | **Sempre** (toda amostra) | **So quando erra** |
| Regra | `w += x * target` | `w += a * (target - y) * x` |
| Convergencia | Nao garantida | Garantida (se linearmente separavel) |
| Parametro extra | — | Taxa de aprendizagem (a = 0.01) |

## Como executar

```bash
cd cli
go run ./trab02-perceptron-letras
```

### Menu interativo (TUI)

- **Treinar passo a passo** — animacao dos ciclos de treinamento
- **Operar** — testa a rede com os pesos aprendidos
- **Treinar e Operar** — faz ambos automaticamente
- **Sair**

## Arquivos

| Arquivo | Conteudo |
|---------|----------|
| `main.go` | Logica do Perceptron com dataset das letras |
| `tui.go` | Interface visual TUI (Bubbletea + Lipgloss) |
