# Damped Harmonic Oscillator — Solved with PINNs

> **Physics-Informed Neural Networks** (Soft & Hard formulations) conditioned on the damping ratio ξ ∈ [0.1, 0.4], trained on the domain z ∈ [0, 20].

---

![Training Evolution](training_evolution.gif)

*The animation above shows both PINNs converging toward the exact analytical solution across all four damping ratios as training progresses.*

---

## The Problem

We solve

$$\frac{d^2x}{dz^2} + 2\xi\,\frac{dx}{dz} + x = 0, \quad z\in[0,20]$$

with fixed initial conditions $x(0)=0.7$, $x'(0)=1.2$ and damping ratio $\xi\in[0.1,\,0.4]$ (all underdamped).

Rather than training a separate model for each ξ, we train **one network** that takes $(z,\,\xi)$ as joint input and learns the full solution manifold — so you can query any ξ in the training range at inference time with a single forward pass.

The closed-form reference solution is

$$x(z;\xi)=e^{-\xi z}\!\left[A\cos(\omega_d z) + B\sin(\omega_d z)\right]$$

where $\omega_d=\sqrt{1-\xi^2}$, $A=x_0$, $B=\dfrac{v_0+\xi x_0}{\omega_d}$.

---

## Two PINN Formulations

### Soft PINN

The network $\mathcal{N}_s(z,\xi;\theta)$ maps inputs directly to $x$ and the loss is

$$\mathcal{L}_{\text{soft}} = \lambda_{\text{pde}}\underbrace{\frac{1}{N_c}\sum_{i=1}^{N_c}\!\bigl(x''+2\xi x'+x\bigr)^2}_{\text{PDE residual}} + \lambda_{IC}\Bigl[\bigl(x(0,\xi)-x_0\bigr)^2 + \bigl(x'(0,\xi)-v_0\bigr)^2\Bigr]$$

The weight $\lambda_{IC}=50$ was chosen by a short sweep. The IC terms shrink during training but never reach exact zero.

### Hard PINN

We embed the ICs directly into the output layer via the **ansatz**

$$x(z,\xi) = x_0 + v_0\,z + z^2\cdot\mathcal{N}_h(z,\xi;\theta)$$

Quick check:

| Condition | Value |
|---|---|
| $x(0,\xi)$ | $x_0$ ✓ |
| $x'(0,\xi)$ | $v_0+0+0=v_0$ ✓ |

Because both ICs are guaranteed for **any** network weights, the hard-PINN loss contains only the PDE residual — no λ_IC hyperparameter needed.

---

## Shared Architecture

Both variants share the same backbone:

```
Input (z, ξ) → [Linear(2→64) → Tanh] × 5 → Linear(64→1)
```

| Property | Value |
|---|---|
| Hidden layers | 5 |
| Hidden units | 64 |
| Activation | Tanh |
| Initialisation | Xavier normal |
| Optimiser | Adam + Cosine Annealing |
| Learning rate | 1 × 10⁻³ → 5 × 10⁻⁶ |
| Epochs | 12 000 |
| Collocation points / batch | 2 500 |
| Parameters | ~25 k |

---

## Results

![Main Comparison](pinn_main_comparison.png)

*Top row: solution traces for four ξ values. Middle row: pointwise absolute error (log scale). Bottom row: training loss curves and RMSE bar chart.*

### Quantitative Error Summary

| ξ | Soft RMSE | Hard RMSE | Soft Max \|e\| | Hard Max \|e\| |
|---|---|---|---|---|
| 0.1 | ≈ 3 × 10⁻³ | ≈ 2 × 10⁻³ | ≈ 8 × 10⁻³ | ≈ 5 × 10⁻³ |
| 0.2 | ≈ 2 × 10⁻³ | ≈ 1 × 10⁻³ | ≈ 6 × 10⁻³ | ≈ 3 × 10⁻³ |
| 0.3 | ≈ 1 × 10⁻³ | ≈ 7 × 10⁻⁴ | ≈ 3 × 10⁻³ | ≈ 2 × 10⁻³ |
| 0.4 | ≈ 7 × 10⁻⁴ | ≈ 5 × 10⁻⁴ | ≈ 2 × 10⁻³ | ≈ 1 × 10⁻³ |

*(Exact numbers depend on training run; Hard PINN consistently outperforms Soft PINN in the majority of runs.)*

---

## Phase Portraits

![Phase Portraits](phase_portraits.png)

*Each spiral starts at $(x_0, v_0)=(0.7,\,1.2)$ (red dot) and converges to the origin as the oscillation damps out. Both PINNs track the exact spiral closely.*

---

## Solution Manifold (3-D)

![3D Surface](3d_solution_manifold.png)

*The same solution plotted over the full $(z,\xi)$ space. The PINN surfaces closely match the exact analytical surface.*

---

## Loss Convergence

![Loss Convergence](loss_convergence.gif)

*The Hard PINN's loss (green) drops more steeply than the Soft PINN's (red) in the early phases because it does not have to "learn" the initial conditions — they are already satisfied by the ansatz.*

---

## Generalisation

Both models are evaluated at ξ = 0.15, 0.25, 0.35 — values that were never targeted during training. The predictions remain accurate, confirming the networks have learned the underlying solution family rather than a collection of memorised trajectories.

---

## Repo Structure

```
.
├── damped_oscillator_pinn.ipynb   # Google Colab notebook (main deliverable)
├── README.md
├── training_evolution.gif
├── loss_convergence.gif
├── exact_solutions.png
├── pinn_main_comparison.png
├── phase_portraits.png
├── 3d_solution_manifold.png
├── error_heatmap.png
└── generalisation_check.png
```

---

## Getting Started

### Option A — Google Colab (recommended)

1. Open `damped_oscillator_pinn.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Switch the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`).
3. Run all cells (`Runtime → Run all`). Training takes ~8 minutes on T4.

### Option B — Local

```bash
git clone <this-repo>
cd <this-repo>
pip install torch numpy matplotlib scipy pandas pillow
jupyter notebook damped_oscillator_pinn.ipynb
```

No additional dependencies are needed; all GIF generation uses Pillow (bundled with matplotlib's animation writer).

---

## Design Choices and Observations

**Why Tanh and not ReLU?**  
PINNs compute second-order derivatives via autograd. ReLU has zero second derivative almost everywhere, making it a poor choice for second-order ODEs. Tanh is smooth, bounded, and its derivatives are easy to compute — it's been the standard activation for PINNs since the original Raissi et al. (2019) paper.

**Why cosine annealing?**  
The loss landscape for PINNs often has wide, flat regions after initial rapid descent. Cosine annealing keeps the learning rate high enough to escape these plateaus, then cools it down toward the end to avoid overshooting sharp minima.

**Why 2 500 collocation points per batch?**  
This is a balance between gradient quality (more points → better estimate of the true PDE residual) and speed. On a GPU, 2 500 points adds very little overhead over 500, so we err on the side of more points.

**Hard PINN stability:**  
Because the Hard PINN ansatz enforces ICs exactly from epoch 0, the optimizer sees a smaller effective loss from the start. This often leads to a smoother loss curve and fewer "jumps" compared to the Soft PINN, where high λ_IC can create competing gradients early in training.

---

## Future SOTA Ideas

1. **Fourier Feature Embeddings** — map the raw inputs through random Fourier features before the first linear layer. This removes the spectral bias and would help for ξ = 0.1 where the signal oscillates for the entire 20-unit window.

2. **hp-Adaptive Collocation (residual-based refinement)** — instead of sampling collocation points uniformly, use the current residual to identify regions with high error and concentrate points there. The DeepXDE library supports this out of the box.

3. **Meta-PINN / MAML** — treat each ξ as a "task" and use Model-Agnostic Meta-Learning. After meta-training, the network adapts to a completely new ξ in a handful of gradient steps.

4. **Physics-Informed Neural Operator (PINO)** — lift the approach from parameter-to-solution ($\xi\to x$) to function-to-function ($c(z)\to x(z)$), allowing spatially-varying damping profiles that the current formulation cannot handle.

5. **Causal Training Curriculum (Wang et al. 2022)** — weight the PDE residual by a causality factor $w(z)\propto e^{-\epsilon\,\mathcal{R}(<z)}$ so the network first gets $z\approx 0$ right before worrying about $z=20$. This is known to cut training time roughly in half for stiff equations.

6. **Bayesian PINNs** — replace point-estimate weights with distributions (via variational inference or MCMC). The output becomes a distribution over solutions, giving calibrated uncertainty bounds — useful for engineering applications where you need to know not just the prediction but how much to trust it.

7. **Neural ODE + Adjoint Hybrid** — propagate ICs forward with a Neural ODE solver (exact gradients via adjoint method) and use PINN residual losses only as a physics-informed regulariser. This could dramatically reduce the number of collocation points needed.

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). Respecting causality is all you need for training physics-informed neural networks. *arXiv:2203.07404*.
- Lu, L., Meng, X., Mao, Z., & Karniadakis, G.E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.
- Lagaris, I.E., Likas, A., & Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.

---

## License

MIT — do whatever you like with it, but a reference back here is appreciated.
