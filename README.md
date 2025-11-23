
-----

# `bitwise-polynomials` (mc-network)

> **Break the Memory Wall.** Train billion-parameter Polynomial Networks on your laptop in milliseconds.

**bitwise-polynomials** is a high-performance engine for high-dimensional polynomial expansion. By replacing explicit feature maps with implicit **Combinadic Algebra**, it allows you to scale from $D=100$ to $D=10,000$ without consuming RAM.

It serves two primary use cases:

1.  **Regression:** A drop-in replacement for `sklearn.preprocessing.PolynomialFeatures` that runs 1000x faster.
2.  **Logic Synthesis:** A "Differentiable FPGA" that can reverse-engineer digital logic circuits (XOR, Adders) purely from data.

-----

## ⚠️ Architectures & Limits

The engine operates in two distinct modes depending on the problem scale:

| Mode | Method | Scale Limit | Use Case |
| :--- | :--- | :--- | :--- |
| **Trainer** | Gradient Descent (Hogwild) | **$M \approx 3 \text{ Billion}$** | High-Dim Regression, MNIST |
| **Solver** | Algebraic (Ridge Regression) | **$M \approx 45,000$** | Logic Synthesis, Exact Circuits |

*Note: The Solver is mathematically exact but requires inverting an $M \times M$ matrix, which becomes memory-bound at $M > 45,000$ (approx 6GB RAM).*

-----

## 🚀 Performance Benchmark (Apple M1 Max)

| Dataset | Dimensions | Degree | Features | Method | Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logic (Small) | $D=16$ | 4 | 1,820 | **Solver** | **0.001s** |
| Housing (Med) | $D=100$ | 3 | 176,851 | **Trainer** | **0.005s** |
| **Massive** | **$D=500$** | **4** | **2.6 Billion** | **Trainer** | **0.14s** |

-----

## 💻 Usage

### 1\. Deep Regression

Use the `PolynomialStack` to train a Gradient Boosted Polynomial Network on massive datasets.

```python
from bitwise_polynomials import PolynomialStack

# M is calculated automatically based on D=500, N=4
# Uses the "Trainer" architecture (Gradient Descent)
model = PolynomialStack(M=2600000000, D=500, degree=4, layers=10)

# Train (Runs on Metal/CUDA)
model.fit(X, y)
```

### 2\. The "Learned FPGA" (Logic Synthesis)

Use `solve_ridge` to find exact boolean coefficients for digital circuits.

```python
from bitwise_polynomials import solve_ridge

# Truth Table for a Full Adder
# ... (Load X, y) ...

# Uses the "Solver" architecture (Algebraic)
w = solve_ridge(H, y, alpha=1e-5)

# Result: Exact coefficients for the logic gate (e.g. 1.0 * A*B*C)
```

-----

## 🧠 How It Works

### The Core: Combinadics

Instead of storing a lookup table mapping `Index -> (exp1, exp2, ...)`, we use the **Combinadic Number System**. Any integer $k$ has a unique representation as a sum of binomial coefficients. We implement a binary-search inversion algorithm in **Metal Shading Language** (and CUDA) to decode these indices in real-time inside the compute kernel.

### The Solver: Closed-Form Algebra

For logic tasks, we utilize the fact that logic gates are just polynomials over the field $\{-1, 1\}$. We generate the feature matrix $H$ on the GPU and solve for weights $w$ using Ridge Regression:
$$w = (H^T H + \alpha I)^{-1} H^T y$$
This allows for **One-Shot Learning** of complex circuits.

-----

## 📜 Citation

If you use the Bitwise Engine in your research, please cite:

```bibtex
@misc{exactus2025,
  author = {Couch, James},
  title = {Exactus: The Bitwise Engine for Memory-Oblivious Polynomial Networks},
  year = {2025},
  url = {https://github.com/tjamescouch/exactus}
}
```

-----
