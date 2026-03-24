# Package `mlp`

Implements a **Multilayer Perceptron (MLP)** with backpropagation for the classic 3-class challenge dataset.

## Algorithm

Full backpropagation with tanh activation:

1. **Forward pass:** compute hidden activations Z = tanh(V·x + V0), output Y = tanh(W·Z + W0)
2. **Backward pass:** compute deltas at output (δ_k) and propagate to hidden layer (δ_j)
3. **Weight update:** ΔW = α · δ_k · Z_j, ΔV = α · δ_j · x_i

## Architecture

- **Inputs:** 3
- **Hidden layer:** 2 neurons (tanh activation)
- **Output layer:** 3 neurons (tanh activation)
- **Learning rate (α):** 0.01
- **Max cycles:** 50000
- **Error target:** 0.001 (sum of squared errors × 0.5)

## Dataset

Three hand-crafted patterns with one-hot (bipolar) targets:

| Pattern | x1 | x2  | x3 | Target class |
|---------|----|-----|----|--------------|
| P1      | 1  | 0.5 | −1 | Class 1 |
| P2      | 1  | 0.5 | 1  | Class 2 |
| P3      | 1  | −0.5| −1 | Class 3 |

Weights are initialized to fixed values for reproducibility.

## Source

**Aula 05 — MLP Backpropagation**
