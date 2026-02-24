# Trabalho 01 — Regra de Hebb: Portas Lógicas

## 📖 Descrição

Implementação da **Regra de Hebb** para treinar e testar um neurônio artificial nas portas lógicas clássicas:

| Porta | Linearmente separável? | Resultado |
|-------|----------------------|-----------|
| AND   | ✅ Sim | Converge |
| OR    | ✅ Sim | Converge |
| NAND  | ✅ Sim | Converge |
| NOR   | ✅ Sim | Converge |
| XOR   | ❌ Não | **Não converge** |

## 🧮 Conceitos

- **Representação bipolar**: entradas e saídas usam -1 e +1 (não binário)
- **Regra de Hebb**: `w = w + (x * target)` — pesos são atualizados para toda amostra
- **Função de ativação**: degrau bipolar (`soma ≥ 0 → 1`, `soma < 0 → -1`)

## ▶️ Como executar

```bash
go run .
```

Um menu interativo permite selecionar portas individuais ou executar todas de uma vez.

## 📁 Arquivos

- `main.go` — lógica completa da Regra de Hebb com menu CLI
- `appemrust.rs` / `appemrust` — versão alternativa em Rust
