# Redes Neurais Artificiais

Repositório com os trabalhos práticos da disciplina de **Redes Neurais Artificiais** (Inteligência Artificial).

## Estrutura

| Pasta | Algoritmo | Descrição |
|-------|-----------|-----------|
| [`Trab 01`](./Trab%2001/) | Regra de Hebb | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 02 - PT 1`](./Trab%2002%20-%20PT%201/) | Perceptron | Reconhecimento de Letras A e B (grade 7×7) |
| [`Trab 02 - PT 2`](./Trab%2002%20-%20PT%202/) | Perceptron | Portas Lógicas (AND, OR, NAND, NOR, XOR) |
| [`Trab 03 - PT 1`](./Trab%2003%20-%20PT%201/) | MADALINE | Reconhecimento de Letras A–M (grade 5×7) |
| `slides/` | — | Slides das aulas em PDF |

## Como rodar

Todos os trabalhos são escritos em **Go**. Para executar qualquer um:

```bash
cd "Trab XX"
go run .
```

> Requisito: [Go 1.24+](https://go.dev/dl/)

## Trab 03 - PT 1: MADALINE

**MADALINE** (Multiple ADALINE) — rede com 13 unidades ADALINE em paralelo, uma por letra.

### Arquitetura

```
Entrada (35 pixels)      Camada 1 (ADALINE)     Saída
  x₀ = ±1  ──────────▶  ◉ ADALINE-A  y_in_A ──┐
  x₁ = ±1  ──────────▶  ◉ ADALINE-B  y_in_B   │
  x₂ = ±1  ──────────▶  ◉ ADALINE-C  y_in_C   ├─▶ argmax ──▶ letra
  ⋮                      ⋮                      │
  x₃₄= ±1  ──────────▶  ◉ ADALINE-M  y_in_M ──┘
```

- **Entradas**: grade 5×7 = 35 pixels em bipolar (−1/+1)
- **Camada 1**: 13 unidades ADALINE, cada uma com 35 pesos + 1 bias
- **Saída**: argmax dos y_in — a ADALINE com maior potencial vence
- **Codificação**: One-of-N — letra correta = +1, todas as outras = −1
- **Aprendizado**: Regra Delta — `wᵢ += α·(t − y)·xᵢ`, atualiza apenas em erro
- **Ativação**: degrau bipolar — `y_in ≥ 0 → +1`, `y_in < 0 → −1`
- **Hiperparâmetros**: α = 0.01, máx. 10.000 ciclos, pesos iniciais ∈ [−0.5, +0.5]

### Funcionalidades da TUI

| Opção | Descrição |
|-------|-----------|
| Treinar (barra) | Animação com barra de progresso e tabela de ADALINEs |
| Treinar (arquitetura) | Diagrama com bolinhas + painel matemático detalhado em tempo real |
| Slides interativos | 7 slides animados explicando cada etapa da MADALINE |
| Desenhar e reconhecer | Grade 5×7 interativa — teclado ou mouse (click/drag) |

### Slides (navegação com ←→)

| Slide | Conteúdo |
|-------|----------|
| 1 — Arquitetura | Diagrama das camadas revelado progressivamente |
| 2 — Entrada | Grade 5×7 e codificação bipolar, target One-of-N |
| 3 — ADALINE | Estrutura interna, pesos sinápticos com barras visuais |
| 4 — Somatório | Fórmula expandida + mapa de calor das contribuições wᵢ·xᵢ |
| 5 — Ativação | Gráfico do degrau bipolar + exemplos reais do treinamento |
| 6 — Regra Delta | Passos do aprendizado + cálculo numérico com valores reais |
| 7 — Saída | Barras de y_in para todas as ADALINEs + decisão argmax |

## Tecnologias

- **Go** — linguagem principal
- **Charm** (Bubble Tea + Lipgloss + Bubbles) — TUI interativa
