# Redes Neurais Artificiais

Repositório com os trabalhos práticos da disciplina de **Redes Neurais Artificiais** (Inteligência Artificial).

## Estrutura

Cada projeto individual tem subpasta `cli/` com a TUI Go interativa.
O servidor web centraliza todos os algoritmos em [`web/`](./web/), organizado por pacotes.

### Projetos individuais (CLI)

| Pasta | Algoritmo | Descrição |
|-------|-----------|-----------|
| [`Trab 01/cli`](./Trab%2001/cli/) | Regra de Hebb | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 02 - PT 1/cli`](./Trab%2002%20-%20PT%201/cli/) | Perceptron | Reconhecimento de Letras A e B (grade 7×7) |
| [`Trab 02 - PT 2/cli`](./Trab%2002%20-%20PT%202/cli/) | Perceptron | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 03/cli`](./Trab%2003/cli/) | MADALINE | Reconhecimento de Letras A–M (grade 5×7) |
| [`Desafios MLP/Desafio Multilayer Perceptron (MLP)/cli`](./Desafios%20MLP/Desafio%20Multilayer%20Perceptron%20(MLP)/cli/) | MLP | Treinamento e visualização passo a passo com TUI |
| [`Desafios MLP/Multilayer Perceptron (MLP) Letras/cli`](./Desafios%20MLP/Multilayer%20Perceptron%20(MLP)%20Letras/cli/) | MLP | Reconhecimento de letras A–Z com entrada manual via TUI |

### Servidor web (`web/`)

```
web/
├── main.go                   ← HTTP server, rotas, handlers (SSE)
├── static/index.html         ← frontend (HTML/CSS/JS, sem frameworks)
└── server/
    ├── main.go
    ├── hebb/                 ← package hebb        (Aula 02)
    ├── perceptron_portas/    ← package perceptronportas (Aula 03)
    ├── perceptron_letras/    ← package perceptronletras (Aula 03)
    ├── madaline/             ← package madaline    (Aula 04)
    ├── mlp/                  ← package mlp         (Aula 05)
    ├── letras/               ← package letras      (Aula 05)
    └── imgreg/               ← package imgreg      (Aula 05 — Aproximação Universal)
```

Slides das aulas em PDF estão em `slides/`.

## Web — Mission Control

Todos os algoritmos rodam via interface web em [`web/`](./web/):

```bash
cd web
make run   # compila, mata porta 8080 e abre o browser
```

## Como rodar (TUI)

Cada trabalho tem TUI em **Go** na subpasta `cli/`:

```bash
cd "Trab 01/cli"
go run .
```

> Requisito: [Go 1.24+](https://go.dev/dl/)

## Tecnologias

- **Go** — linguagem principal (backend HTTP/SSE + algoritmos)
- **Charm** (Bubble Tea + Lipgloss + Bubbles) — TUI interativa (Trab 02 e 03)
- **HTML/CSS/JS** — frontend web (sem frameworks)
