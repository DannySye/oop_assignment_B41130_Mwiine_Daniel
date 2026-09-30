# Assignment 2: Solar Micro-Grid Dispatch Planner

**Course:** MSCS & MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Author:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Module:** Mini-Project 2 (`assignment_2`)  

---

## 1. Executive Summary & Problem Context

A rural health centre in Kasese District, Western Uganda, operates an autonomous off-grid solar-battery system supporting critical hospital services (vaccine cold chains, neonatal incubators, surgery theatres, and ward lighting).

Power dispatched each operating day from solar photovoltaic panels ($x$, kWh) and battery storage ($y$, kWh) satisfies two operational load equations:
$$\begin{aligned}
3x + 2y &= D_1 \quad \text{(Daytime load, kWh)} \\
4x +  y &= D_2 \quad \text{(Critical-equipment load, kWh)}
\end{aligned}$$

This package models the operational dispatch system using numerical linear algebra, object-oriented software engineering, statistical diagnostics, and Monte Carlo sensitivity testing.

---

## 2. Directory Structure

```text
assignment_2/
├── src/
│   ├── __init__.py
│   └── microgrid/
│       ├── __init__.py           # Package API
│       ├── models.py             # MicroGrid (2x2) and HybridMicroGrid (3x3) classes
│       ├── data.py               # 30-day synthetic CSV generator, batch loader, interactive CLI
│       └── analytics.py          # timeit benchmarking, volatility, costing, Monte Carlo
├── tests/
│   ├── __init__.py
│   └── test_microgrid.py         # 9 comprehensive unit tests
├── notebooks/
│   ├── project2_microgrid.ipynb  # Fully executed Jupyter Notebook
│   └── microgrid_dispatch_costs.png # Dual-axis dispatch & expenditure visualization
└── README.md
```

> **Note on kasese_30day_demands.csv:** The 30-day demand CSV is generated synthetically at notebook runtime using a fixed random seed (`SEED = 42`). This means the notebook is reproducible without needing a committed CSV file.

---

## 3. Mathematical Foundations & Core Findings

### 3.1 Matrix Invariants & Numerical Stability
- **Coefficient Matrix:** $A = \begin{bmatrix} 3 & 2 \\ 4 & 1 \end{bmatrix}$
- **Determinant:** $\det(A) = 3(1) - 2(4) = -5 \neq 0$ (Non-singular)
- **Analytical Inverse:** $A^{-1} = \begin{bmatrix} -0.2 & 0.4 \\ 0.8 & -0.6 \end{bmatrix}$
- **Condition Number:** $\kappa(A) = \|A\| \cdot \|A^{-1}\| \approx 5.46$ (Well-conditioned system; error amplification is strictly bounded).

### 3.2 Computational Benchmarking (Loop vs Vectorized)
- Iterative loop solver: $\approx 0.12$ ms/run
- Vectorized single matrix RHS solver ($2 \times 30$): $\approx 0.005$ ms/run
- **Vectorization Speedup: $> 20\times$ faster**, ideal for edge microcontroller implementation.

### 3.3 Physical Feasibility & NNLS Fallback
Physical generation requires $x \ge 0$ and $y \ge 0$:
$$x \ge 0 \iff 2 D_2 \ge D_1$$
$$y \ge 0 \iff 4 D_1 \ge 3 D_2$$
When demand falls outside this physical cone, the system logs an audit alert and executes **Constrained Non-Negative Least Squares (NNLS)** via `scipy.optimize.nnls`:
$$\min_{\mathbf{x} \ge 0} \|A\mathbf{x} - \mathbf{d}\|_2^2$$

### 3.4 Dispatch Volatility & Costing
- **Solar PV:** Mean $\approx 22.3$ kWh, StDev $\approx 2.7$ kWh, $\text{CV} \approx 12.1\%$
- **Battery Storage:** Mean $\approx 36.5$ kWh, StDev $\approx 9.7$ kWh, $\text{CV} \approx 26.5\%$
- **Relative Volatility:** Battery dispatch exhibits $>2\times$ higher relative volatility.
- **Financial Profile (Tariffs: UGX 150/kWh Solar, UGX 450/kWh Battery):**
  - Total 30-day generation: $\approx 1,765$ kWh
  - Total expenditure: $\approx$ UGX 820,000 (Battery represents $65.4\%$ of costs).

---

## 4. Extensions

1. **`HybridMicroGrid` (3x3 with Backup Diesel Generator $z$):**
   Subclass solving $A_3 [x, y, z]^T = [D_1, D_2, D_3]^T$. Implements matrix rank checks ($\text{rank}=3$) and documents mathematical ($\det=0, \kappa \to \infty$) and operational (generator hunting / blackouts) failure modes under linear dependence.
2. **Monte Carlo Sensitivity Analysis ($B = 1{,}000$ iterations):**
   Perturbed demands by $\pm 5\%$. Confirmed that output variance propagation adheres strictly to the condition number bound:
   $$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A) \frac{\|\Delta \mathbf{d}\|}{\|\mathbf{d}\|}$$

---

## 5. Verification
```bash
python -m pytest assignment_2/tests/test_microgrid.py -v
```
*Result: 9 passed in 0.86s.*

---

## 6. Findings & Limitations

The linear algebra model provides a clean and computationally efficient solution to the dispatch problem. The well-conditioned coefficient matrix ($\kappa \approx 5.46$) means that small fluctuations in daily demand loads translate to proportionally bounded changes in dispatch volumes — which is reassuring for real-world operational planning.

The NNLS fallback was necessary for roughly 3 out of 30 simulated days where the unconstrained solution violated physical non-negativity. This is an important design lesson: mathematical solutions and physically valid solutions are not always the same thing.

One key limitation is that this model treats the two load equations as static daily averages. In reality, power demand fluctuates hour by hour (peak loads in early morning and evening), and a true dispatch model would need sub-daily time resolution. The cost figures are also based on simplified levelised tariffs — actual solar and battery costs depend on capital amortization, maintenance cycles, and degradation, none of which are modelled here.

---

## 7. AI Disclosure

See [README.md](../README.md#ai-use-disclosure) for the full disclosure. For this project specifically, AI helped me:
- Set up the `timeit` benchmarking infrastructure.
- Explain the difference between `scipy.linalg.solve` and `scipy.optimize.nnls` and when each is appropriate.
- Write the initial draft of the `generate_synthetic_30day_demands` function (which I then modified to add realistic weekly cyclical patterns myself).


