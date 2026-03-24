# Package `letras`

Implements a **Multilayer Perceptron (MLP)** with backpropagation for full A–Z letter recognition.

## Algorithm

Backpropagation with tanh activation, one output neuron per letter:

1. **Forward pass:** Z = tanh(V·x + V0), Y = tanh(W·Z + W0)
2. **Backward pass:** compute δ_k at output, propagate δ_j to hidden layer
3. **Weight update:** online (per letter per cycle)

## Architecture

- **Inputs:** 35 (5×7 pixel grid, bipolar: −1 or +1)
- **Hidden layer:** 15 neurons (tanh activation)
- **Output layer:** 26 neurons — one per letter A–Z (tanh activation)
- **Learning rate (α):** 0.01
- **Max cycles:** 50000
- **Error target:** 0.5 (sum of squared errors × 0.5 across all outputs)

## Dataset

26 hand-crafted 5×7 bipolar letter patterns (A–Z).

One-vs-rest targets: +1 for the correct letter, −1 for all others.

Weights are initialized randomly with seed 42 for reproducibility.

## Source

**Aula 05 — MLP Backpropagation**
