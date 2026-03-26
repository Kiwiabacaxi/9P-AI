# Trabalho 02 (Parte 2) — Perceptron: Portas Logicas

## Descricao

Programa com **interface grafica TUI** que treina um **Perceptron** para cada porta logica, mostrando os pesos finais e bias apos o treinamento.

## Portas Disponiveis

| Porta | Linearmente separavel? | Perceptron converge? |
|-------|----------------------|---------------------|
| AND   | Sim | Sim |
| OR    | Sim | Sim |
| NAND  | Sim | Sim |
| NOR   | Sim | Sim |
| XOR   | Nao | **Nao** |

> **XOR**: O Perceptron simples nao consegue resolver o XOR porque ele nao e linearmente separavel. O programa detecta isso e mostra o aviso.

## Conceitos

- **Perceptron** so corrige pesos quando erra: `w += a * (target - y) * x`
- **Taxa de aprendizagem** (a): 0.01
- **Pesos iniciais**: aleatorios em [-0.5, +0.5]
- **Convergencia**: repete ciclos ate acertar tudo (exceto XOR)

## Como executar

```bash
cd cli
go run ./trab02-perceptron-portas
```

### Menu TUI

- Escolha uma porta individual ou **"Treinar TODAS"**
- Animacao passo a passo do treinamento
- Ao final: pesos (W1, W2, Bias), ciclos e acuracia
- Tabela resumo ao treinar todas as portas

## Arquivos

| Arquivo | Conteudo |
|---------|----------|
| `main.go` | Logica do Perceptron + definicao das portas |
| `tui.go` | Interface TUI (Bubbletea + Lipgloss) |
