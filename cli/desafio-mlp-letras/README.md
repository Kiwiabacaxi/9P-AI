# MLP Letras — Reconhecimento de Letras A-Z com Backpropagation

Aplicacao do algoritmo MLP para reconhecimento das 26 letras do alfabeto (A-Z), usando backpropagation com entradas de pixels 5x7.

## Diferenca em relacao ao Desafio MLP

| | Desafio MLP | MLP Letras |
|---|---|---|
| Problema | Exemplo numerico do slide | Reconhecimento de letras |
| Entradas | 3 valores fixos | 35 pixels (grade 5x7 bipolar) |
| Saidas | 3 classes | 26 classes (A-Z) |
| Neuronios ocultos | 2 | 15 |
| Pesos iniciais | Fixos (do slide) | Aleatorios [-0.5, +0.5] |
| Padroes | 3 | 26 (um por letra) |

O **algoritmo e identico** — mesma formula de forward, mesma backpropagation, mesmo criterio de parada.

## Arquitetura

```
Entradas (35)  ->  Camada Oculta (15 neuronios)  ->  Saida (26 neuronios)
 pixels 5x7        tanh                               tanh, one-hot A-Z
```

| Parametro | Valor |
|---|---|
| Entradas | 35 (grade 5x7 pixels, bipolar -1/+1) |
| Neuronios ocultos | 15 |
| Saidas | 26 (uma por letra A-Z) |
| Ativacao | tanh em todas as camadas |
| Taxa de aprendizado (a) | 0.01 |
| Criterio de parada | Erro total <= 0.5 ou 50.000 ciclos |
| Codificacao target | One-hot: +1 na letra correta, -1 nas outras 25 |

## Como executar

```bash
cd cli
go run ./desafio-mlp-letras
```

### Navegacao

| Tecla | Acao |
|---|---|
| `cima` / `baixo` | Navegar menu |
| `enter` | Selecionar |
| `esquerda` / `direita` | Mudar letra (tela de resultado) |
| `setas` | Mover cursor na grade (teste interativo) |
| `space` | Acender/apagar pixel (teste interativo) |
| `r` | Resetar grade (teste interativo) |
| `esc` / `q` | Voltar ao menu |
| `ctrl+c` | Sair |

## Modo Teste Interativo

Grade 5x7 em branco onde voce:

1. Move o cursor com as setas
2. Pressiona `space` para acender/apagar pixels
3. A rede classifica em **tempo real** conforme voce desenha
4. Mostra o top-3 de letras mais provaveis com as ativacoes `y`

## Notas sobre Convergencia

- Com 26 letras e 15 neuronios ocultos, o treino costuma convergir em **algumas centenas a poucos milhares de ciclos**
- Pesos aleatorios -> resultado varia a cada execucao
- Se nao convergir no limite de 50.000 ciclos, a tela mostra a acuracia obtida

## Relacao com os Outros Trabalhos

| Trabalho | Rede | Entradas | Letras |
|---|---|---|---|
| trab03-madaline | MADALINE (Regra Delta) | 35 pixels | A-M (13) |
| desafio-mlp-letras | MLP + Backpropagation | 35 pixels | A-Z (26) |
| desafio-mlp | MLP + Backpropagation | 3 valores | — |

## Arquivos

| Arquivo | Conteudo |
|---------|----------|
| `main.go` | Algoritmo MLP + dataset das 26 letras |
| `tui.go` | Interface Bubble Tea: menu, treino, resultado, teste interativo |
