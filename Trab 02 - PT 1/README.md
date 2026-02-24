# Trabalho 02 (Parte 1) — Perceptron: Reconhecimento de Letras

## 📖 Descrição

Adaptação do Trabalho 01 para usar o **Perceptron** no lugar da Regra de Hebb.  
A rede reconhece as letras **A** e **B** representadas em matrizes **7×7** (49 pixels).

```
Letra A          Letra B
· █ █ █ █ █ ·    █ █ █ █ █ · ·
█ · · · · · █    █ · · · · █ ·
█ · · · · · █    █ · · · · █ ·
█ █ █ █ █ █ █    █ █ █ █ █ · ·
█ · · · · · █    █ · · · · █ ·
█ · · · · · █    █ · · · · █ ·
█ · · · · · █    █ █ █ █ █ · ·
```

## 🧮 Hebb vs Perceptron

| | Hebb | Perceptron |
|---|------|-----------|
| Atualização | **Sempre** (toda amostra) | **Só quando erra** |
| Regra | `w += x * target` | `w += α * (target - y) * x` |
| Convergência | Não garantida | Garantida (se linearmente separável) |
| Parâmetro extra | — | Taxa de aprendizagem (α = 0.01) |

## ▶️ Como executar

```bash
go run .
```

### Menu interativo (TUI)

- ⚡ **Treinar passo a passo** — animação dos ciclos de treinamento
- 🔍 **Operar** — testa a rede com os pesos aprendidos
- 🚀 **Treinar e Operar** — faz ambos automaticamente
- 🚪 **Sair**

## 📁 Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `main.go` | Lógica do Perceptron com comentários de apresentação |
| `tui.go` | Interface visual TUI (Bubbletea + Lipgloss) |

## 🛠️ Dependências

- [Bubbletea](https://github.com/charmbracelet/bubbletea) — framework TUI
- [Lipgloss](https://github.com/charmbracelet/lipgloss) — estilização
- [Bubbles](https://github.com/charmbracelet/bubbles) — componentes (spinner)
