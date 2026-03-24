# Package `imgreg`

Implements **Image Regression with MLP** — a demonstration of the Universal Approximation Theorem.

## Algorithm

The network learns to map pixel coordinates (x, y) → (R, G, B), reconstructing a 16×16 image pixel by pixel using SGD with pixel shuffling per epoch.

- **Hidden layers:** ReLU activation
- **Output layer:** Sigmoid activation (compresses output to [0, 1] for RGB)
- **Weight initialization:** He initialization — W ~ N(0, √(2/fan\_in)), optimal for ReLU
- **Optimization:** SGD with pixel shuffle per epoch to avoid memorizing presentation order
- **Loss:** MSE — L = 0.5 · Σ(target − output)²

## Architecture (configurable)

- **Inputs:** 2 (normalized x, y coordinates in [−1, 1])
- **Hidden layers:** 2–5 layers, 16–128 neurons each (user-configurable)
- **Output:** 3 (R, G, B channels in [0, 1])
- **Learning rate:** 0.001–0.05 (user-configurable)
- **Max epochs:** up to 2000 (user-configurable)

## Available target images

| Name | Description |
|------|-------------|
| `coracao` | Red heart shape on dark background (heart curve equation) |
| `smiley`  | Yellow smiley face on dark blue background |
| `radial`  | Concentric colored waves (sin/cos pattern) |
| `brasil`  | Simplified Brazilian flag (green, yellow, blue) |

## Source

**Aula 05 — MLP — Aproximação Universal**
