# Object-Oriented Programming with Python – Portfolio of Applied Mini-Projects

**Student Name:** Daniel Mwiine  
**Student ID:** B41130  
**Registration Number:** S26M25/001  
**Program:** Master of Science in Computer Science & Master of Science in Data Science  
**Course:** Advanced Object-Oriented Programming & Computational Modelling  
**Term:** Advent 2026  
**Corpus / Repository:** `DannySye/oop_assignment_B41130_Mwiine_Daniel`

---

## 1. Executive Summary

This repository presents the complete computational and object-oriented implementations for all **five applied domain mini-projects** specified in the coursework curriculum. Grounded in real-world socioeconomic, infrastructural, ecological, and economic challenges across Uganda and East Africa, each project combines:
- **Strict Object-Oriented Design:** Custom classes, type annotations, descriptive docstrings, inheritance hierarchies, abstract base classes, defensive validation, and operator overloading (`__repr__`, `__len__`, `__lt__`, `__eq__`, `__getitem__`).
- **Reproducible Scientific Computing:** Vectorized operations via `numpy`, scientific algorithms via `scipy` (`scipy.linalg`, `scipy.optimize`, `scipy.signal`), and reproducible pseudo-random generation with pinned seeds (`np.random.default_rng(seed)`).
- **Automated Verification:** Fully automated test suites with **57 passed tests** under `pytest` covering both nominal execution paths and rigorous boundary/edge cases.
- **Headless Executed Notebooks:** Five self-contained Jupyter notebooks executed top-to-bottom via `nbclient`, producing rich outputs, summary tables, and publication-ready 300 DPI visualizations.
- **Rigorous Extensions:** Dedicated computational extensions for every project extending baseline specifications into advanced analytical regimes.

---

## 2. Portfolio Index & Status Summary

| # | Mini-Project Title | Domain & Geographic Context | Directory | Jupyter Notebook | Pytest Suite | Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **UBOS District Population Forecaster & School Infrastructure Planner** | Demographics & Education (Kampala, Wakiso, Gulu, Mukono, Mbarara) | [`assignment_1/`](assignment_1/) | [`project1_population.ipynb`](assignment_1/notebooks/project1_population.ipynb) | [`test_population.py`](assignment_1/tests/test_population.py) | **23/23 Passed** |
| **2** | **Solar Micro-Grid Dispatch Planner** | Renewable Energy Engineering (Kasese Rural Microgrid) | [`assignment_2/`](assignment_2/) | [`project2_microgrid.ipynb`](assignment_2/notebooks/project2_microgrid.ipynb) | [`test_microgrid.py`](assignment_2/tests/test_microgrid.py) | **9/9 Passed** |
| **3** | **Lake Victoria Fish Stock & Export Risk Model** | Fisheries Ecology & Financial Risk (Lake Victoria Nile Perch & Tilapia) | [`assignment_3/`](assignment_3/) | [`project3_fisheries.ipynb`](assignment_3/notebooks/project3_fisheries.ipynb) | [`test_fisheries.py`](assignment_3/tests/test_fisheries.py) | **9/9 Passed** |
| **4** | **Rainfall Pattern & Crop Suitability Analyser** | Agro-Climatology & Agronomy (Gulu, Kampala, Mbarara) | [`assignment_4/`](assignment_4/) | [`project4_climate.ipynb`](assignment_4/notebooks/project4_climate.ipynb) | [`test_climate.py`](assignment_4/tests/test_climate.py) | **8/8 Passed** |
| **5** | **Taxi Route Revenue, Pricing & Fleet Planner** | Transport Economics & Fleet Operations (Ntinda, Entebbe, Mukono) | [`assignment_5/`](assignment_5/) | [`project5_transport.ipynb`](assignment_5/notebooks/project5_transport.ipynb) | [`test_transport.py`](assignment_5/tests/test_transport.py) | **8/8 Passed** |
| **ALL** | **Consolidated Portfolio Suite** | **Entire Project Scope** | **Root** | **All 5 Executed** | **Unified Pytest** | **57/57 Passed** |

---

## 3. Mini-Project Overviews

### Mini-Project 1: UBOS District Population Forecaster & School Infrastructure Planner
- **Directory:** [`assignment_1/`](assignment_1/)
- **Core Concepts:** Demographic cohort forecasting, compound growth models ($P_t = P_0 (1+r)^t$), descriptive statistics from first principles without external statistical libraries, educational infrastructure planning (pupil-to-teacher ratio $PTR=40$, classroom cap $PCR=50$).
- **OOP Architecture:** `District` model with defensive validation, polymorphic forecasting hierarchy (`PopulationForecaster`, `ExponentialGrowthForecaster`, `LinearGrowthForecaster`), `SchoolInfrastructurePlanner`, and bootstrap resampling engine (`BootstrapConfidenceInterval`).
- **Extension Task:** Non-parametric empirical bootstrap prediction intervals ($B=1,000$) quantifying demographic uncertainty up to year 2040.
- **Visual Artifact:** [`assignment_1/notebooks/district_population_forecasts.png`](assignment_1/notebooks/district_population_forecasts.png)

### Mini-Project 2: Solar Micro-Grid Dispatch Planner
- **Directory:** [`assignment_2/`](assignment_2/)
- **Core Concepts:** 24-hour generation dispatch across Solar PV, Battery Storage (BESS), and Diesel Backup; multi-tier economic cost optimization ($0.08, $0.15, $0.45/kWh); 2-bus matrix system dispatch ($A \mathbf{x} = \mathbf{D}$) with Non-Negative Least Squares (`scipy.optimize.nnls`) physical feasibility fallback.
- **OOP Architecture:** `MicroGrid` base class encapsulated with state-of-charge (SoC) tracking ($20\% - 90\%$), inheritance via `HybridMicroGrid` adding biomass gasification dispatch, 30-day CSV automated parsing, and comprehensive energy balance diagnostics.
- **Extension Task:** 30-day continuous simulation under stochastically generated rainy/cloudy weather anomalies evaluating diesel reliance surges and economic sensitivity.
- **Visual Artifact:** [`assignment_2/notebooks/microgrid_dispatch_costs.png`](assignment_2/notebooks/microgrid_dispatch_costs.png)

### Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model
- **Directory:** [`assignment_3/`](assignment_3/)
- **Core Concepts:** Bioeconomic modelling via discrete Schaefer surplus production dynamics:
  $$B_{t+1} = B_t + r B_t \left(1 - \frac{B_t}{K}\right) - Y_t$$
  Maximum Sustainable Yield ($MSY = \frac{rK}{4} = 62,500\text{ tons}$), lognormal export price volatility ($S_{t+1} = S_t \exp\left((\mu - 0.5\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)$), Monte Carlo Value-at-Risk (VaR at $95\%$ and $99\%$), and Expected Shortfall (CVaR).
- **OOP Architecture:** `FishStock` class managing carrying capacity constraints and population collapse alarms, `PriceModel` stochastic generator, and `RiskAssessor` portfolio analytics.
- **Extension Task:** Seasonal moratorium policy simulation (3-month fishing closure) demonstrating biological stock recovery and $+18.2\%$ long-run revenue enhancement.
- **Visual Artifact:** [`assignment_3/notebooks/fisheries_biomass_and_var.png`](assignment_3/notebooks/fisheries_biomass_and_var.png)

### Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser
- **Directory:** [`assignment_4/`](assignment_4/)
- **Core Concepts:** Agro-meteorological time-series analysis; circular boundary peak detection (`scipy.signal.find_peaks`) classifying unimodal (Gulu) vs. bimodal (Kampala, Mbarara) rainfall regimes; multi-dimensional vector cosine similarity:
  $$\cos(\theta) = \frac{\mathbf{r}_{\text{region}} \cdot \mathbf{r}_{\text{crop}}}{\|\mathbf{r}_{\text{region}}\| \|\mathbf{r}_{\text{crop}}\|}$$
  Agronomic rule-based crop scoring matrix (Coffee, Maize, Cassava).
- **OOP Architecture:** `Region` domain entity with circular rainfall wrapping and operator overloading (`__eq__`, `__len__`, `__getitem__`), `CropRule` decision entity, and `RainfallAnalyser` vectorized metrics engine.
- **Extension Task:** 10-year synthetic multi-year climate record ($10 \times 12$ matrix) evaluating intra-seasonal drought probabilities and climatic stability.
- **Visual Artifacts:** [`assignment_4/notebooks/rainfall_and_crop_suitability.png`](assignment_4/notebooks/rainfall_and_crop_suitability.png), [`assignment_4/notebooks/gulu_multiyear_boxplots.png`](assignment_4/notebooks/gulu_multiyear_boxplots.png)

### Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner
- **Directory:** [`assignment_5/`](assignment_5/)
- **Core Concepts:** Urban transit economics (Kampala Minibus Taxi Association); microeconomic market equilibrium solving via matrix inversion (`scipy.linalg.solve`):
  $$Q_d = 120 - 0.02P, \quad Q_s = 10 + 0.03P \implies P^* = \text{UGX } 2{,}200, \quad Q^* = 76$$
  Rolling-origin walk-forward backtesting (Days 4–10); fleet deployment optimization ($8\text{ trips/day} \times 14\text{ seats} = 112\text{ seats/day}$, $+15\%$ safety buffer, ceiling integer rounding).
- **OOP Architecture:** `Route` entity with sorting by gross revenue (`__lt__`, `__repr__`, `__len__`), abstract base `TransportForecaster` with subclasses `MovingAverageForecaster`, `SimpleExponentialSmoothingForecaster` (with optimal $\alpha^*$ grid search), `LinearTrendForecaster`, and `SeasonalNaiveForecaster`.
- **Extension Task:** 60-day transit demand synthesis showing that Seasonal-Naive models ($m=7$) outperform Moving Averages by $>40\%$ error reduction on periodic transit series.
- **Visual Artifacts:** [`assignment_5/transport_forecasts.png`](assignment_5/transport_forecasts.png), [`assignment_5/seasonal_transport_benchmark.png`](assignment_5/seasonal_transport_benchmark.png)

---

## 4. Repository Structure

```text
oop_assignment_B41130_Mwiine_Daniel/
├── README.md                                  <- Root master portfolio documentation
├── pyproject.toml                             <- Pytest configuration for all 5 suites
├── requirements.txt                           <- Unified Python dependencies
│
├── assignment_1/                              <- UBOS Population & Infrastructure
│   ├── README.md                              <- Project 1 technical documentation
│   ├── requirements.txt
│   ├── src/population/                        <- Models, stats, forecasters, bootstrap
│   ├── tests/test_population.py               <- 23 passing tests
│   └── notebooks/
│       ├── build_notebook.py
│       ├── project1_population.ipynb          <- Fully executed notebook
│       └── district_population_forecasts.png
│
├── assignment_2/                              <- Solar Micro-Grid Dispatch Planner
│   ├── README.md                              <- Project 2 technical documentation
│   ├── requirements.txt
│   ├── kasese_30day_demands.csv               <- 30-day simulation dataset
│   ├── microgrid_dispatch_costs.png
│   ├── src/microgrid/                         <- Models, data loader, analytics
│   ├── tests/test_microgrid.py                <- 9 passing tests
│   └── notebooks/
│       ├── build_notebook.py
│       ├── project2_microgrid.ipynb           <- Fully executed notebook
│       └── microgrid_dispatch_costs.png
│
├── assignment_3/                              <- Lake Victoria Fisheries Risk Model
│   ├── README.md                              <- Project 3 technical documentation
│   ├── requirements.txt
│   ├── fisheries_biomass_and_var.png
│   ├── src/fisheries/                         <- Stock models, price models, VaR
│   ├── tests/test_fisheries.py                <- 9 passing tests
│   └── notebooks/
│       ├── build_notebook.py
│       ├── project3_fisheries.ipynb           <- Fully executed notebook
│       └── fisheries_biomass_and_var.png
│
├── assignment_4/                              <- Rainfall & Crop Suitability Analyser
│   ├── README.md                              <- Project 4 technical documentation
│   ├── requirements.txt
│   ├── rainfall_and_crop_suitability.png
│   ├── gulu_multiyear_boxplots.png
│   ├── src/climate/                           <- Region models, peak detection, cosine sim
│   ├── tests/test_climate.py                  <- 8 passing tests
│   └── notebooks/
│       ├── build_notebook.py
│       ├── project4_climate.ipynb             <- Fully executed notebook
│       ├── rainfall_and_crop_suitability.png
│       └── gulu_multiyear_boxplots.png
│
└── assignment_5/                              <- Taxi Revenue & Fleet Dispatch
    ├── README.md                              <- Project 5 technical documentation
    ├── requirements.txt
    ├── transport_forecasts.png
    ├── seasonal_transport_benchmark.png
    ├── src/transport/                         <- Route models, forecasters, fleet sizing
    ├── tests/test_transport.py                <- 8 passing tests
    └── notebooks/
        ├── build_notebook.py
        ├── project5_transport.ipynb           <- Fully executed notebook
        ├── transport_forecasts.png
        └── seasonal_transport_benchmark.png
```

---

## 5. Quickstart & Verification Guide

### 5.1 Environment Installation
Ensure Python 3.10+ (tested on Python 3.12) is installed:
```powershell
pip install -r requirements.txt
```

### 5.2 Running the Unified Automated Test Suite
Execute all 57 unit and integration tests across the 5 projects:
```powershell
python -m pytest -v
```

Expected output:
```text
============================= 57 passed in 4.85s ==============================
```

To run an individual project's test suite:
```powershell
python -m pytest assignment_1/tests/test_population.py -v
python -m pytest assignment_2/tests/test_microgrid.py -v
python -m pytest assignment_3/tests/test_fisheries.py -v
python -m pytest assignment_4/tests/test_climate.py -v
python -m pytest assignment_5/tests/test_transport.py -v
```

### 5.3 Headless Notebook Execution & Regeneration
Each notebook can be viewed interactively or rebuilt and executed headlessly via its dedicated `build_notebook.py` orchestrator:
```powershell
# Rebuild and execute any notebook headlessly
python assignment_1/notebooks/build_notebook.py
python assignment_2/notebooks/build_notebook.py
python assignment_3/notebooks/build_notebook.py
python assignment_4/notebooks/build_notebook.py
python assignment_5/notebooks/build_notebook.py
```

---

## 6. Academic Integrity & Technical Verification Statement

- **Independence & Rigor:** All architectural designs, mathematical models, boundary validations, and analytical discussions were developed strictly according to the course specifications.
- **AI Tooling Disclosure:** AI coding assistance (Google Gemini) was employed in an interactive pair-programming capacity for rapid boilerplate generation, cross-platform terminal orchestration, and formatting; all algorithmic logic, numerical solvers, and domain conclusions were audited, tested, and verified for mathematical correctness and programmatic reliability.
