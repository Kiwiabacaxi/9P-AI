# Redes Neurais Artificiais

Repositório com os trabalhos práticos da disciplina de **Redes Neurais Artificiais** (Inteligência Artificial).

## Estrutura

| Pasta | Algoritmo | Descrição |
|-------|-----------|-----------|
| [`Trab 01`](./Trab%2001/) | Regra de Hebb | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 02 - PT 1`](./Trab%2002%20-%20PT%201/) | Perceptron | Reconhecimento de Letras A e B (grade 7×7) |
| [`Trab 02 - PT 2`](./Trab%2002%20-%20PT%202/) | Perceptron | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 03`](./Trab%2003/) | MADALINE | Reconhecimento de Letras A–M (grade 5×7) |
| [`Desafios MLP/Desafio Multilayer Perceptron (MLP)`](./Desafios%20MLP/Desafio%20Multilayer%20Perceptron%20(MLP)/) | MLP | Treinamento e visualização passo a passo do MLP com TUI |
| [`Desafios MLP/Multilayer Perceptron (MLP) Letras`](./Desafios%20MLP/Multilayer%20Perceptron%20(MLP)%20Letras/) | MLP | Reconhecimento de letras com MLP e entrada manual via TUI |
| `slides/` | — | Slides das aulas em PDF |

## Web — Mission Control

Todos os algoritmos rodam via interface web em [`web/`](./web/):

```bash
cd web
make run   # compila, mata porta 8080 e abre o browser
```

## Como rodar (TUI)

Os trabalhos individuais têm TUI em **Go**:

```bash
cd "Trab XX"
go run .
```

> Requisito: [Go 1.24+](https://go.dev/dl/)

## Tecnologias

- **Go** — linguagem principal (backend HTTP/SSE + algoritmos)
- **Charm** (Bubble Tea + Lipgloss + Bubbles) — TUI interativa (Trab 02 e 03)
- **HTML/CSS/JS** — frontend web (sem frameworks)
