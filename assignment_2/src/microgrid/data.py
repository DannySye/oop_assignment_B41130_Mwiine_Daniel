"""
Data ingestion and synthesis module for the Kasese Health Centre Micro-Grid.
Provides CSV synthetic generation, batch loading, and validated interactive input.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


def generate_synthetic_30day_demands(
    output_path: Optional[Path | str] = None,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Generates a 30-day synthetic demand profile for Kasese Health Centre.
    Encodes weekly cyclical shifts (operating weekdays vs weekend maintenance)
    and realistic Gaussian stochastic noise.

    Formula:
        D1(t) = 140 + 25 * sin(2*pi*t / 7) + noise_1
        D2(t) = 110 + 15 * cos(2*pi*t / 7) + noise_2

    Returns:
        NumPy array of shape (30, 2) containing [D1, D2] for days 1 to 30.
    """
    rng = np.random.default_rng(random_seed)
    days = np.arange(1, 31)

    # Base cyclical pattern (7-day week cycle)
    d1_base = 140.0 + 25.0 * np.sin(2.0 * np.pi * days / 7.0)
    d2_base = 110.0 + 15.0 * np.cos(2.0 * np.pi * days / 7.0)

    # Realistic stochastic noise (standard deviation 6 kWh)
    d1_noise = rng.normal(0.0, 6.0, size=30)
    d2_noise = rng.normal(0.0, 5.0, size=30)

    # Combined positive loads
    d1 = np.maximum(d1_base + d1_noise, 40.0)
    d2 = np.maximum(d2_base + d2_noise, 30.0)

    demands = np.column_stack([d1, d2])

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day", "d1_daytime_kwh", "d2_critical_kwh"])
            for day_idx, (val1, val2) in enumerate(demands, 1):
                writer.writerow([day_idx, f"{val1:.2f}", f"{val2:.2f}"])

    return demands


def load_demand_csv(filepath: Path | str) -> np.ndarray:
    """
    Reads a 30-day demand CSV file with input format validation.

    Args:
        filepath: Path to CSV file containing columns for day, d1, and d2.

    Returns:
        NumPy array of shape (N, 2).
    """
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Demand CSV file not found at: {filepath}")

    records: List[Tuple[float, float]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            try:
                # Find column keys flexibly
                d1_key = next(k for k in row.keys() if "d1" in k.lower())
                d2_key = next(k for k in row.keys() if "d2" in k.lower())
                val1 = float(row[d1_key])
                val2 = float(row[d2_key])
                if val1 <= 0 or val2 <= 0:
                    raise ValueError(f"Demands must be positive, found d1={val1}, d2={val2} on row {row_idx}.")
                records.append((val1, val2))
            except Exception as exc:
                raise ValueError(f"Error parsing CSV row {row_idx}: {exc}") from exc

    if len(records) == 0:
        raise ValueError("CSV file contains no valid demand records.")

    return np.array(records, dtype=np.float64)


def prompt_valid_float(prompt_text: str) -> float:
    """
    Prompts user via input() with robust validation:
    rejects non-numeric, negative, or blank entries with dynamic re-prompting.
    """
    while True:
        raw_val = input(prompt_text).strip()
        if not raw_val:
            print("Error: Input cannot be blank. Please enter a positive number.")
            continue
        try:
            val = float(raw_val)
            if val <= 0.0:
                print(f"Error: Energy demand must be strictly positive (> 0), received {val}.")
                continue
            return val
        except ValueError:
            print(f"Error: '{raw_val}' is not a valid numeric float. Please enter a valid number.")


def interactive_input_demands() -> Tuple[float, float]:
    """
    Interactive terminal interface to input daytime (D1) and critical (D2) loads.
    """
    print("\n--- Kasese Health Centre Micro-Grid Input Interface ---")
    d1 = prompt_valid_float("Enter Daytime Load D1 (kWh): ")
    d2 = prompt_valid_float("Enter Critical-Equipment Load D2 (kWh): ")
    return d1, d2
