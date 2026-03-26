# Trabalho 03 — MADALINE

Reconhecimento das letras **A-M** usando uma rede **MADALINE** (Multiple ADALINE).

## Arquitetura

```
Entrada (35 pixels)      Camada 1 (ADALINE)     Saida
  x0 = +/-1  ---------> ADALINE-A  y_in_A ---+
  x1 = +/-1  ---------> ADALINE-B  y_in_B    |
  x2 = +/-1  ---------> ADALINE-C  y_in_C    +-> argmax --> letra
  ...                    ...                   |
  x34= +/-1  ---------> ADALINE-M  y_in_M ---+
```

- **Entradas**: grade 5x7 = 35 pixels em bipolar (-1/+1)
- **Camada 1**: 13 unidades ADALINE, cada uma com 35 pesos + 1 bias
- **Saida**: argmax dos y_in — a ADALINE com maior potencial vence
- **Codificacao**: One-of-N — letra correta = +1, todas as outras = -1
- **Aprendizado**: Regra Delta — `w_i += a*(t - y)*x_i`, atualiza apenas em erro
- **Ativacao**: degrau bipolar — `y_in >= 0 -> +1`, `y_in < 0 -> -1`
- **Hiperparametros**: a = 0.01, max. 10.000 ciclos, pesos iniciais em [-0.5, +0.5]

## Como executar

```bash
cd cli
go run ./trab03-madaline
```

## Funcionalidades da TUI

| Opcao | Descricao |
|-------|-----------|
| Treinar (barra) | Animacao com barra de progresso e tabela de ADALINEs |
| Treinar (arquitetura) | Diagrama com bolinhas + painel matematico detalhado em tempo real |
| Slides interativos | 7 slides animados explicando cada etapa da MADALINE |
| Desenhar e reconhecer | Grade 5x7 interativa — teclado ou mouse (click/drag) |

## Slides (navegacao com setas)

| Slide | Conteudo |
|-------|----------|
| 1 — Arquitetura | Diagrama das camadas revelado progressivamente |
| 2 — Entrada | Grade 5x7 e codificacao bipolar, target One-of-N |
| 3 — ADALINE | Estrutura interna, pesos sinapticos com barras visuais |
| 4 — Somatorio | Formula expandida + mapa de calor das contribuicoes w_i*x_i |
| 5 — Ativacao | Grafico do degrau bipolar + exemplos reais do treinamento |
| 6 — Regra Delta | Passos do aprendizado + calculo numerico com valores reais |
| 7 — Saida | Barras de y_in para todas as ADALINEs + decisao argmax |
