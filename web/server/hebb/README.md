# Package `hebb`

Implements the **Hebb Learning Rule** for binary logic gates.

## Algorithm

The Hebb rule is a one-pass, unsupervised-style learning rule:

```
w_i ← w_i + x_i · y    (for each weight)
bias ← bias + y
```

Weights are updated once per training sample — there is no iterative convergence loop. After a single pass through all four bipolar samples, the final weights are used to classify the same samples.

## Architecture

- **Inputs:** 2 (bipolar: −1 or +1)
- **Hidden layers:** none (single-layer)
- **Output:** 1 (bipolar via sign activation)
- **Activation:** sign function — returns +1 if net input ≥ 0, else −1

## Dataset

Logic gate truth tables in bipolar encoding (0 → −1):

| Gate | Learnable? |
|------|-----------|
| AND  | Yes |
| OR   | Yes |
| NAND | Yes |
| NOR  | Yes |
| XOR  | No (not linearly separable) |

## Source

**Aula 02 — Regra de Hebb**
