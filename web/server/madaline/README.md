# Package `madaline`

Implements **MADALINE** (Multiple ADALINE) for multi-class letter recognition (A–M).

## Algorithm

MADALINE uses one ADALINE unit per output class. Each unit is trained independently with the delta/LMS rule:

```
Δ = α · (target_j − y_j)
w_ij ← w_ij + Δ · x_i    (if y_j ≠ target_j)
bias_j ← bias_j + Δ
```

Training repeats until all letters are correctly classified in a full cycle (convergence) or the maximum cycle limit is reached.

## Architecture

- **Inputs:** 35 (5×7 pixel grid, bipolar: −1 or +1)
- **Hidden layers:** none
- **Output units:** 13 (one ADALINE per letter A–M, one-vs-rest bipolar targets)
- **Learning rate (α):** 0.01
- **Max cycles:** 10000
- **Activation:** sign function (bipolar)

## Dataset

13 hand-crafted 5×7 bipolar letter patterns: A, B, C, D, E, F, G, H, I, J, K, L, M.

One-vs-rest targets: the target unit for the correct letter is +1, all others are −1.

## Source

**Aula 04 — MADALINE/ADALINE**
