# AI Usage Transparency Log

**Student:** Daniel Mwiine (B41130)  
**Course:** MSCS/MSDS – Object-Oriented Programming with Python  
**Term:** Advent 2026

---

## Overview

As required by the academic integrity guidelines in the assignment specification, this document transparently discloses how AI coding assistants were used in the completion of this assignment. I used **Google Gemini** (through the Antigravity IDE) as a pair-programming aid throughout the development process.

---

## Scope of AI Assistance

### What I Used AI For

1. **Boilerplate scaffolding:** Setting up the initial project folder structure, `pyproject.toml`, and `.gitignore`. I gave the AI the directory layout from the spec and asked it to create the skeleton structure.

2. **Syntax and API lookups:** I regularly asked things like "how do I make a NumPy array read-only?" or "what is the signature for `scipy.linalg.solve`?" — things I could find in docs but faster through conversation.

3. **Debugging test failures:** When `pytest` tests failed (e.g., the circular peak detection in Assignment 4 was initially wrong), I pasted the error and the relevant code and asked the AI to explain what was happening. I then understood the fix (tiling the array for wrap-around) and implemented it myself after verifying the reasoning.

4. **Notebook execution infrastructure:** The `build_notebook.py` scripts that programmatically assemble and execute notebooks via `nbclient` were largely AI-scaffolded, since this was a workflow I had not used before.

5. **LaTeX/markdown formatting:** Some of the mathematical notation in the READMEs and notebook markdown cells was AI-assisted (mostly getting LaTeX syntax right, e.g., for MSY formulas, equilibrium system matrices).

6. **Writing first drafts of docstrings:** I asked the AI to generate initial docstrings for methods I had already written, then edited them for accuracy and tone.

### What I Did NOT Use AI For (core intellectual work)

- **All algorithmic design decisions** — choosing which OOP pattern to use, what validation rules to enforce, how to structure the inheritance hierarchy.
- **Mathematical derivations** — the MSY formula derivation, the Schaefer logistic model, the matrix equilibrium setup for Assignment 5, and the bootstrap resampling procedure were all worked out from scratch using course materials and referenced textbooks.
- **Data interpretation and written analysis** — the Findings & Limitations sections in all five notebooks were written by me personally, reflecting my own understanding of what the numbers mean in the Ugandan context.
- **Test case design** — I designed all pytest test cases myself, deciding which boundary conditions to check (e.g., zero division, negative values, mismatched array lengths).

---

## Sample Prompts Used

Below are representative prompts I gave to the AI during development (paraphrased):

- *"I have a `DistrictPopulation` class that stores population arrays. How should I implement `__getitem__` to support both integer indexing (return year, value tuple) and slice indexing (return a new DistrictPopulation)?"*

- *"My `scipy.signal.find_peaks` is not detecting December rainfall peaks in Kampala because the array is circular (December wraps into January). What strategy would fix this?"*

- *"The spec says to critique why Fibonacci sequence growth is biologically unrealistic for fish stocks. Can you help me outline 3–4 points, then I'll write them in my own words?"*

- *"In `scipy.optimize.nnls`, what does the second return value (residual norm) mean and when should I use it over `scipy.linalg.solve`?"*

- *"Help me write an `nbclient` script to programmatically build a notebook from code strings, execute it headlessly, and write the output back to an `.ipynb` file."*

---

## Viva Readiness

I can explain every line of code in this repository. Where AI suggestions were incorporated, I reviewed, tested, and modified them until I understood exactly what they did and why. I would not submit code I cannot explain.

---

*This log was written by me personally, not by the AI. I am aware that I may be called for a short viva to explain any part of my submission.*
