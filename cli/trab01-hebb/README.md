# Trabalho 01 — Regra de Hebb: Portas Logicas

## Descricao

Implementacao da **Regra de Hebb** para treinar e testar um neuronio artificial nas portas logicas classicas:

| Porta | Linearmente separavel? | Resultado |
|-------|----------------------|-----------|
| AND   | Sim | Converge |
| OR    | Sim | Converge |
| NAND  | Sim | Converge |
| NOR   | Sim | Converge |
| XOR   | Nao | **Nao converge** |

## Conceitos

- **Representacao bipolar**: entradas e saidas usam -1 e +1 (nao binario)
- **Regra de Hebb**: `w = w + (x * target)` — pesos sao atualizados para toda amostra
- **Funcao de ativacao**: degrau bipolar (`soma >= 0 -> 1`, `soma < 0 -> -1`)

## Como executar

```bash
cd cli
go run ./trab01-hebb
```

Um menu interativo permite selecionar portas individuais ou executar todas de uma vez.

## Arquivos

- `main.go` — logica completa da Regra de Hebb com menu CLI
