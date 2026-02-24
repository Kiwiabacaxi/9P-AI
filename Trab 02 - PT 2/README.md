# Trabalho 02 (Parte 2) — Perceptron: Portas Lógicas

## 📖 Descrição

Programa com **interface gráfica TUI** que treina um **Perceptron** para cada porta lógica, mostrando os pesos finais e bias após o treinamento.

## ⚡ Portas Disponíveis

| Porta | Linearmente separável? | Perceptron converge? |
|-------|----------------------|---------------------|
| AND   | ✅ Sim | ✅ Sim |
| OR    | ✅ Sim | ✅ Sim |
| NAND  | ✅ Sim | ✅ Sim |
| NOR   | ✅ Sim | ✅ Sim |
| XOR   | ❌ Não | ❌ **Não** |

> ⚠ **XOR**: O Perceptron simples não consegue resolver o XOR porque ele não é linearmente separável. O programa detecta isso e mostra o aviso.

## 🧮 Conceitos

- **Perceptron** só corrige pesos quando erra: `w += α * (target - y) * x`
- **Taxa de aprendizagem** (α): 0.01
- **Pesos iniciais**: aleatórios em [-0.5, +0.5]
- **Convergência**: repete ciclos até acertar tudo (exceto XOR)

## ▶️ Como executar

```bash
go run .
```

### Menu TUI

- Escolha uma porta individual ou **"Treinar TODAS"**
- Animação passo a passo do treinamento
- Ao final: pesos (W1, W2, Bias), ciclos e acurácia
- Tabela resumo ao treinar todas as portas

## 📁 Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `main.go` | Lógica do Perceptron + definição das portas |
| `tui.go` | Interface TUI (Bubbletea + Lipgloss) |
