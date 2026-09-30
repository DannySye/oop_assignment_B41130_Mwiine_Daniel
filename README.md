# OOP with Python – Mini-Projects Assignment

**Student:** Daniel Mwiine  
**Registration No.:** B41130 / S26M25/001  
**Programme:** MSCS & MSDS  
**Course:** Object-Oriented Programming with Python  
**Term:** Advent 2026  
**Institution:** Uganda Christian University / Makerere University  

---

## What This Repository Contains

This repo is my submission for the five mini-projects assignment. Each project is in its own folder (`assignment_1/` through `assignment_5/`) with:
- A `src/` folder holding all the Python source modules I wrote.
- A `tests/` folder with pytest unit tests.
- A `notebooks/` folder with the executed Jupyter notebook and any generated figures.
- A `README.md` explaining the project.

The projects cover five different real-world problems set in Uganda, using OOP design, NumPy/SciPy, and matplotlib. No ML libraries were used — everything is implemented from mathematical first principles.

---

## Project Summary

| # | Project | Core Techniques | Tests |
|:-:|:--------|:----------------|:-----:|
| 1 | [UBOS District Population Forecaster](assignment_1/README.md) | Compound growth, linear regression, Fibonacci forecasting, bootstrap CI | 23 |
| 2 | [Solar Micro-Grid Dispatch Planner](assignment_2/README.md) | Linear systems (`scipy.linalg`), NNLS, stacked bar visualisation | 9 |
| 3 | [Lake Victoria Fish Stock & Export Risk](assignment_3/README.md) | Logistic growth (Schaefer model), Monte Carlo VaR, seasonal closure | 9 |
| 4 | [Rainfall Pattern & Crop Suitability](assignment_4/README.md) | Circular peak detection (`scipy.signal`), cosine similarity, crop rules | 8 |
| 5 | [Taxi Route Revenue & Fleet Planner](assignment_5/README.md) | Market equilibrium (`scipy.linalg`), SMA/SES/Linear forecasting, fleet sizing | 8 |

**All 57 tests pass** (run `pytest -v` from the repo root to verify).

---

## Repository Structure

The spec asked for a flat structure with all modules under one `src/`. I chose instead to keep each mini-project isolated in its own folder — both for clarity during development and to make it easier to run and test each project independently. All five projects still satisfy the structural requirements (modular `.py` source files, separate test files, one notebook per project).

```
oop_assignment_B41130_Mwiine_Daniel/
├── README.md               ← this file (includes AI disclosure)
├── pyproject.toml          ← pytest config covering all 5 test suites
├── requirements.txt        ← all Python dependencies
├── .gitignore
│
├── assignment_1/           ← Population Forecaster
│   ├── src/population/     ← models, stats, forecasters, evaluation, bootstrap, planning
│   ├── tests/              ← 23 tests
│   ├── notebooks/          ← project1_population.ipynb + figure
│   └── README.md
│
├── assignment_2/           ← Solar Micro-Grid
│   ├── src/microgrid/      ← models, data loader, analytics
│   ├── tests/              ← 9 tests
│   ├── notebooks/          ← project2_microgrid.ipynb + figure
│   └── README.md
│
├── assignment_3/           ← Fisheries Risk Model
│   ├── src/fisheries/      ← FishStock, PriceModel, RiskAssessor
│   ├── tests/              ← 9 tests
│   ├── notebooks/          ← project3_fisheries.ipynb + figure
│   └── README.md
│
├── assignment_4/           ← Rainfall & Crop Suitability
│   ├── src/climate/        ← Region, CropRule, peak detection, metrics
│   ├── tests/              ← 8 tests
│   ├── notebooks/          ← project4_climate.ipynb + figures
│   └── README.md
│
└── assignment_5/           ← Taxi Route Planner
    ├── src/transport/      ← Route, forecasters, fleet sizing
    ├── tests/              ← 8 tests
    ├── notebooks/          ← project5_transport.ipynb + figures
    └── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ is required (tested on Python 3.12).

### Run all tests

```bash
pytest -v
```

This runs all 57 tests across all five projects. To run a single project's tests:

```bash
pytest assignment_1/tests/ -v
pytest assignment_3/tests/ -v
# etc.
```

### Open a notebook

```bash
jupyter notebook assignment_1/notebooks/project1_population.ipynb
```

Each notebook can be run with **Restart & Run All** from the Jupyter interface and will execute top-to-bottom without errors.

---

## Notes and Known Limitations

- The rainfall data in Project 4 is illustrative/synthetic (given in the spec), not real UNMA station data. An extension with real data would strengthen the analysis.
- The fish stock model in Project 3 uses annual discrete time steps for simplicity. A continuous-time PDE model (e.g., age-structured) would be more biologically accurate.
- The transport demand data in Project 5 is a 10-day snapshot. Longer time series would make the backtesting results more statistically meaningful.
- All forecasting in Project 1 uses univariate methods — no spatial autocorrelation between neighbouring districts (e.g., Kampala and Wakiso) is modelled.

---

## AI Use Disclosure

As required by the academic integrity guidelines in the assignment specification, this section transparently documents how AI coding assistants were used. I used **Google Gemini** as a pair-programming aid throughout the development process.

### What I Used AI For

1. **Boilerplate scaffolding:** Setting up the initial project folder structure, `pyproject.toml`, and `.gitignore`. I gave the AI the directory layout from the spec and asked it to create the skeleton.

2. **Syntax and API lookups:** I regularly asked things like "how do I make a NumPy array read-only?" or "what is the signature for `scipy.linalg.solve`?" — things I could find in the docs but faster through conversation.

3. **Debugging test failures:** When `pytest` tests failed (e.g., the circular peak detection in Assignment 4 was initially wrong), I pasted the error and the relevant code and asked the AI to explain what was happening. I then understood the fix (tiling the array for wrap-around) and implemented it myself after verifying the reasoning.

4. **Notebook execution infrastructure:** The scripts that programmatically assemble and execute notebooks via `nbclient` were largely AI-scaffolded, since this was a workflow I had not used before.

5. **LaTeX/markdown formatting:** Some of the mathematical notation in the READMEs and notebook markdown cells was AI-assisted (mostly getting LaTeX syntax right for things like the MSY formula and the equilibrium matrix system).

6. **Writing first drafts of docstrings:** I asked the AI to generate initial docstrings for methods I had already written, then edited them for accuracy.

### What I Did NOT Use AI For

- **Algorithmic design decisions** — choosing which OOP patterns to use, what validation rules to enforce, how to structure the inheritance hierarchy.
- **Mathematical derivations** — the MSY formula, the Schaefer logistic model, the matrix equilibrium setup for Assignment 5, and the bootstrap resampling procedure were worked out from course materials and textbooks.
- **Data interpretation and written analysis** — the Findings & Limitations sections in all notebooks were written by me personally.
- **Test case design** — I designed all pytest test cases myself, deciding which boundary conditions to cover (zero division, negative values, mismatched array lengths, etc.).

### Sample Prompts Used (Paraphrased)

- *"I have a `DistrictPopulation` class that stores population arrays. How should I implement `__getitem__` to support both integer indexing (return year, value tuple) and slice indexing (return a new DistrictPopulation)?"*
- *"My `scipy.signal.find_peaks` is not detecting December rainfall peaks in Kampala because the array is circular. What strategy would fix this?"*
- *"The spec says to critique why Fibonacci sequence growth is biologically unrealistic for fish stocks. Can you help me outline 3–4 points, then I'll write them in my own words?"*
- *"In `scipy.optimize.nnls`, what does the second return value (residual norm) mean and when should I use it over `scipy.linalg.solve`?"*
- *"Help me write an `nbclient` script to programmatically build a notebook from code strings, execute it headlessly, and write the output back to an `.ipynb` file."*

### Viva Readiness

I can explain every line of code in this repository. Where AI suggestions were incorporated, I reviewed, tested, and modified them until I understood exactly what they did and why. I would not submit code I cannot explain.
