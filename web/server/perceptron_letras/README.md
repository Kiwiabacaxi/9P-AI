# Package `perceptronletras`

Implements the **Perceptron learning algorithm** for letter recognition (A vs B).

## Algorithm

Iterative error-correction learning rule applied to a 7×7 pixel grid:

```
if y ≠ target:
    Δ = α · (target − y)
    w_i ← w_i + Δ · x_i    (for each of 49 inputs)
    bias ← bias + Δ
```

Training repeats until both letters are classified correctly (convergence) or the maximum cycle limit is reached.

## Architecture

- **Inputs:** 49 (7×7 pixel grid, bipolar: −1 or +1)
- **Hidden layers:** none (single-layer)
- **Output:** 1 (bipolar — +1 = letter B, −1 = letter A)
- **Learning rate (α):** 0.01
- **Max cycles:** 10000
- **Activation:** sign function

## Dataset

Two hand-crafted 7×7 bipolar letter patterns:

- **A** → target −1
- **B** → target +1

## Source

**Aula 03 — Perceptron**
