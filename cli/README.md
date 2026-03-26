# CLI — Programas TUI Interativos

Interfaces de terminal interativas para cada algoritmo de rede neural, construidas com [Bubble Tea](https://github.com/charmbracelet/bubbletea) + [Lipgloss](https://github.com/charmbracelet/lipgloss).

## Programas

| Pasta | Algoritmo | Arquitetura | Descricao |
|-------|-----------|-------------|-----------|
| [`trab01-hebb`](trab01-hebb/README.md) | Regra de Hebb | 2 -> 1 | Portas logicas (AND, OR, NAND, NOR, XOR) |
| [`trab02-perceptron-letras`](trab02-perceptron-letras/README.md) | Perceptron | 49 -> 1 (7x7) | Reconhecimento das letras A e B |
| [`trab02-perceptron-portas`](trab02-perceptron-portas/README.md) | Perceptron | 2 -> 1 | Portas logicas com TUI e tabela de resultados |
| [`trab03-madaline`](trab03-madaline/README.md) | MADALINE | 35 -> 13 -> 13 | Reconhecimento das letras A-M |
| [`desafio-mlp`](desafio-mlp/README.md) | MLP + Backprop | 3 -> 2 -> 3 | Exemplo numerico exato dos slides |
| [`desafio-mlp-letras`](desafio-mlp-letras/README.md) | MLP + Backprop | 35 -> 15 -> 26 | Reconhecimento de A-Z |

## Como executar

Todos compartilham o mesmo `go.mod`. A partir desta pasta:

```bash
go run ./trab01-hebb
go run ./trab02-perceptron-letras
go run ./trab02-perceptron-portas
go run ./trab03-madaline
go run ./desafio-mlp
go run ./desafio-mlp-letras
```

## Dependencias

- [Bubble Tea](https://github.com/charmbracelet/bubbletea) — framework TUI
- [Lipgloss](https://github.com/charmbracelet/lipgloss) — estilizacao
- [Bubbles](https://github.com/charmbracelet/bubbles) — componentes (spinner, progress bar)
