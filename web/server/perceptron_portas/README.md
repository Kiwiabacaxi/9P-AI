# Package `perceptronportas`

Implements the **Perceptron learning algorithm** for binary logic gates.

## Algorithm

Iterative error-correction learning rule (Rosenblatt Perceptron):

```
if y ≠ target:
    Δ = α · (target − y)
    w_i ← w_i + Δ · x_i
    bias ← bias + Δ
```

Training repeats until all samples are classified correctly (convergence) or the maximum cycle limit is reached.

## Architecture

- **Inputs:** 2 (bipolar: −1 or +1)
- **Hidden layers:** none (single-layer)
- **Output:** 1 (bipolar via sign activation)
- **Learning rate (α):** 0.01
- **Max cycles:** 1000
- **Activation:** sign function — returns +1 if net input ≥ 0, else −1

## Dataset

Logic gate truth tables in bipolar encoding (0 → −1):

| Gate | Converges? |
|------|-----------|
| AND  | Yes |
| OR   | Yes |
| NAND | Yes |
| NOR  | Yes |
| XOR  | No (not linearly separable — will not converge) |

## Source

**Aula 03 — Perceptron**
