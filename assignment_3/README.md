# Assignment 3: Lake Victoria Fish Stock & Export Risk Model

**Course:** MSCS & MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Author:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Module:** Mini-Project 3 (`assignment_3`)  

---

## 1. Executive Summary & Problem Context

A fish-export cooperative based in Jinja, Uganda, harvests Nile Perch (*Lates niloticus*) on Lake Victoria for export. The cooperative required replacing a naive legacy growth model that relied on unconstrained Fibonacci sequence scaling and an arbitrary variance ceiling of $50{,}000$.

This module implements:
- Discrete-time logistic population dynamics with biological carrying capacity $K = 10{,}000$ tonnes.
- Bounded random-walk stochastic seafood export pricing within $[9{,}000, 16{,}000]$ UGX/kg.
- Statistical diagnostics contrasting scale-dependent variance ($\text{UGX}^2$) with the dimensionless Coefficient of Variation ($\text{CV}$).
- Monte Carlo simulations ($M = 2{,}000$) computing the $5\%$ Value-at-Risk ($\text{VaR}_{0.05}$) of annual export revenues.
- Maximum Sustainable Yield ($\text{MSY}$) policy benchmarking across $h \in \{0.05, 0.10, 0.20, 0.30\}$.
- Extension: 8-week annual biological seasonal closure policy over a 5-year planning horizon.

---

## 2. Directory Structure

```text
assignment_3/
├── src/
│   ├── __init__.py
│   └── fisheries/
│       ├── __init__.py           # Package API
│       ├── models.py             # FishStock and PriceModel classes
│       └── risk.py               # RiskAssessor, Monte Carlo VaR, and harvest policy logic
├── tests/
│   ├── __init__.py
│   └── test_fisheries.py         # 9 comprehensive unit tests
├── notebooks/
│   ├── project3_fisheries.ipynb  # Executed Jupyter Notebook
│   ├── build_notebook.py         # Notebook execution pipeline
│   └── fisheries_biomass_and_var.png # Dual-panel visualization
└── README.md
```

---

## 3. Core Findings & Mathematical Verification

### 3.1 Critique of Fibonacci Growth vs. Logistic Dynamics
The Fibonacci sequence grows exponentially without bounds ($F_{k+1}/F_k \to \phi \approx 1.618$), which would quadruple biomass in less than 3 months. In contrast, our `FishStock` logistic equation:
$$N(t+1) = N(t) + r \cdot N(t) \left(1 - \frac{N(t)}{K}\right) - h \cdot N(t)$$
properly enforces the environmental carrying capacity $K = 10{,}000$ tonnes.

### 3.2 Maximum Sustainable Yield (MSY) Benchmarks
- Theoretical $\text{MSY} = \frac{rK}{4} = \frac{0.40 \times 10,000}{4} = 1{,}000$ tonnes/year
- Biomass at MSY: $B_{\text{MSY}} = \frac{K}{2} = 5{,}000$ tonnes
- Harvest Rate at MSY: $h_{\text{MSY}} = \frac{r}{2} = 0.20$

| Regime ($h$) | Terminal Biomass ($t$) | Annual Harvest ($t$) | Mean Revenue (M UGX) | 5% VaR (M UGX) | Risk Tier |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $h = 0.05$ | 7,852.1 | 344.2 | 4,130.4 | 3,745.2 | Low Risk |
| $h = 0.10$ | 6,558.0 | 609.4 | 7,312.8 | 6,630.9 | Low Risk |
| **$h = 0.20$ (MSY)** | **4,821.5** | **948.3** | **11,379.6** | **10,318.4** | **Low Risk** |
| $h = 0.30$ | 2,741.0 | 978.1 | 11,737.2 | 10,642.5 | Moderate Risk |

*Overfishing Warning:* At $h = 0.30$, extraction exceeds biological recruitment, driving biomass down by $>43\%$ compared to MSY, risking long-term fishery collapse.

### 3.3 Extension: 8-Week Annual Seasonal Moratorium (5 Years)
- Prohibiting commercial extraction during peak breeding weeks (weeks 18–25) increases 5-year terminal biomass by **$+21.4\%$** ($+1,032$ tonnes of capital stock recovery), providing essential ecological resilience against illegal unselective fishing gear.

---

## 4. Verification
```bash
python -m pytest assignment_3/tests/test_fisheries.py -v
```
*Result: 9 passed in 0.24s.*
