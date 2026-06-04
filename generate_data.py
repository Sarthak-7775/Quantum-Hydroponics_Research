"""
1_generate_data.py
──────────────────
Hydroponic Digital Twin — Synthetic Dataset Generator

Simulates a multi-week hydroponic growth cycle for Lactuca sativa by:
  • Generating realistic, correlated time-series sensor streams
  • Injecting controlled sensor dropouts and noise artefacts
  • Deriving the composite Plant Health Index (H) from chlorophyll
    and leaf-area sub-indices
  • Cleaning, scaling, and exporting the final dataset to hydro_data.csv

Author  : Hybrid Quantum-Classical CPS Research Group
License : MIT
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Simulation hyper-parameters
# ─────────────────────────────────────────────
RANDOM_SEED          = 42
N_DAYS               = 28          # Growth cycle length
SAMPLES_PER_HOUR     = 1           # 1 reading per hour
DROPOUT_PROBABILITY  = 0.03        # 3 % sensor dropout rate
OUTPUT_FILE          = Path("hydro_data.csv")

# Hardware power envelopes (Watts)
PUMP_POWER_RANGE     = (25.0,  60.0)
LED_POWER_RANGE      = (80.0, 200.0)
COOLER_POWER_RANGE   = (40.0, 120.0)

# Environmental operating ranges
PH_RANGE             = (5.5,  7.0)
EC_RANGE             = (1.0,  3.5)   # mS/cm
AMBIENT_TEMP_RANGE   = (18.0, 28.0)  # °C
WATER_TEMP_RANGE     = (18.0, 24.0)  # °C
HUMIDITY_RANGE       = (55.0, 85.0)  # %

# Biological thresholds
CHLOROPHYLL_RANGE    = (1.5,  4.5)   # mg/g
LEAF_AREA_RANGE      = (20.0, 220.0) # cm²

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Helper: smooth correlated signal
# ─────────────────────────────────────────────
def _smooth_signal(
    n: int,
    low: float,
    high: float,
    noise_std: float = 0.02,
    trend_strength: float = 0.0,
) -> np.ndarray:
    """
    Generate a realistic sensor signal using cumulative Gaussian noise
    clipped to [low, high].

    Parameters
    ----------
    n              : number of time steps
    low, high      : physical bounds
    noise_std      : fractional standard deviation of the innovation
    trend_strength : optional linear drift across the series
    """
    mid   = (low + high) / 2.0
    scale = (high - low) / 2.0

    innovations = np.random.normal(0, noise_std * scale, size=n)
    trend       = np.linspace(0, trend_strength * scale, n)
    raw         = np.cumsum(innovations) + mid + trend

    # Ornstein–Uhlenbeck mean reversion to keep signal in bounds
    ou = np.empty(n)
    ou[0] = mid
    theta, sigma = 0.05, noise_std * scale
    for t in range(1, n):
        ou[t] = ou[t - 1] + theta * (mid - ou[t - 1]) + sigma * np.random.randn()

    return np.clip(ou, low, high)


# ─────────────────────────────────────────────
# Digital Twin: signal synthesis
# ─────────────────────────────────────────────
def simulate_growth_cycle(n_steps: int) -> pd.DataFrame:
    """
    Simulate a complete growth cycle and return a raw DataFrame with
    timestamps, hardware states, sensor readings, and biological metrics.
    """
    log.info("Simulating %d time steps (%d days)…", n_steps, N_DAYS)

    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n_steps,
        freq="h",
    )

    # ── Hardware (control) signals ──────────────────────────────────────
    pump_power  = _smooth_signal(n_steps, *PUMP_POWER_RANGE,  noise_std=0.015)
    led_power   = _smooth_signal(n_steps, *LED_POWER_RANGE,   noise_std=0.020)
    cooler_power= _smooth_signal(n_steps, *COOLER_POWER_RANGE,noise_std=0.018)

    # Diurnal LED cycle: reduce power during night hours (22:00 – 06:00)
    hours       = timestamps.hour.to_numpy()
    night_mask  = (hours >= 22) | (hours < 6)
    led_power   = np.where(night_mask, led_power * 0.15, led_power)

    # ── Environmental (sensor) signals ──────────────────────────────────
    # pH drifts slightly upward over the cycle (nutrient depletion)
    ph          = _smooth_signal(n_steps, *PH_RANGE,          noise_std=0.010,
                                 trend_strength=0.1)
    ec          = _smooth_signal(n_steps, *EC_RANGE,           noise_std=0.015,
                                 trend_strength=-0.05)
    ambient_temp= _smooth_signal(n_steps, *AMBIENT_TEMP_RANGE, noise_std=0.012)
    water_temp  = _smooth_signal(n_steps, *WATER_TEMP_RANGE,   noise_std=0.008)
    humidity    = _smooth_signal(n_steps, *HUMIDITY_RANGE,     noise_std=0.020)

    # ── Biological metrics ───────────────────────────────────────────────
    # Chlorophyll increases with LED exposure; declines under heat stress
    led_norm    = (led_power  - LED_POWER_RANGE[0])  / (LED_POWER_RANGE[1]  - LED_POWER_RANGE[0])
    temp_stress = np.clip((ambient_temp - 25.0) / 5.0, 0.0, 1.0)   # 0 below 25 °C, 1 at 30 °C

    growth_days = np.linspace(0, 1, n_steps) ** 0.7   # sigmoid-like growth curve

    chlorophyll = (
        CHLOROPHYLL_RANGE[0]
        + (CHLOROPHYLL_RANGE[1] - CHLOROPHYLL_RANGE[0])
        * (0.5 * growth_days + 0.3 * led_norm - 0.2 * temp_stress)
        + np.random.normal(0, 0.05, n_steps)
    )
    chlorophyll = np.clip(chlorophyll, *CHLOROPHYLL_RANGE)

    leaf_area   = (
        LEAF_AREA_RANGE[0]
        + (LEAF_AREA_RANGE[1] - LEAF_AREA_RANGE[0])
        * (0.6 * growth_days + 0.2 * (ec - EC_RANGE[0]) / (EC_RANGE[1] - EC_RANGE[0]))
        + np.random.normal(0, 2.5, n_steps)
    )
    leaf_area   = np.clip(leaf_area, *LEAF_AREA_RANGE)

    df = pd.DataFrame({
        "timestamp"              : timestamps,
        "pump_power_w"           : np.round(pump_power,   3),
        "led_power_w"            : np.round(led_power,    3),
        "cooler_power_w"         : np.round(cooler_power, 3),
        "water_ph"               : np.round(ph,           4),
        "water_ec"               : np.round(ec,           4),
        "ambient_temp"           : np.round(ambient_temp, 3),
        "water_temp"             : np.round(water_temp,   3),
        "humidity"               : np.round(humidity,     3),
        "maceration_chlorophyll" : np.round(chlorophyll,  4),
        "leaf_area_cm2"          : np.round(leaf_area,    3),
    })

    return df


# ─────────────────────────────────────────────
# Sensor dropout injection
# ─────────────────────────────────────────────
SENSOR_COLS = [
    "water_ph", "water_ec", "ambient_temp",
    "water_temp", "humidity",
    "maceration_chlorophyll", "leaf_area_cm2",
]

def inject_dropouts(df: pd.DataFrame, p: float = DROPOUT_PROBABILITY) -> pd.DataFrame:
    """
    Randomly null-out sensor readings to simulate real hardware dropouts.
    """
    log.info("Injecting sensor dropouts  (p = %.2f)…", p)
    df = df.copy()
    for col in SENSOR_COLS:
        mask = np.random.rand(len(df)) < p
        df.loc[mask, col] = np.nan
    dropped = df[SENSOR_COLS].isna().sum().sum()
    log.info("  %d dropout cells introduced.", dropped)
    return df


# ─────────────────────────────────────────────
# Data cleaning
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair missing values:
      • Forward-fill then backward-fill for continuity
      • Column median as final fallback
    """
    log.info("Cleaning sensor dropouts…")
    df = df.copy()
    df[SENSOR_COLS] = (
        df[SENSOR_COLS]
        .ffill()
        .bfill()
        .fillna(df[SENSOR_COLS].median())
    )
    remaining_nulls = df.isna().sum().sum()
    assert remaining_nulls == 0, f"Unexpected nulls after cleaning: {remaining_nulls}"
    log.info("  All NaN values resolved.")
    return df


# ─────────────────────────────────────────────
# Health Index calculation
# ─────────────────────────────────────────────
def compute_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the master Plant Health Index H ∈ [0, 1] as a weighted geometric
    mean of the normalised chlorophyll and leaf-area sub-indices.

        H = (χ_norm)^0.5 × (A_norm)^0.5

    where χ_norm and A_norm are MinMax-scaled to [0, 1] using known
    physiological bounds.
    """
    log.info("Computing Plant Health Index (H)…")
    chi_norm = (df["maceration_chlorophyll"] - CHLOROPHYLL_RANGE[0]) / (
        CHLOROPHYLL_RANGE[1] - CHLOROPHYLL_RANGE[0]
    )
    area_norm = (df["leaf_area_cm2"] - LEAF_AREA_RANGE[0]) / (
        LEAF_AREA_RANGE[1] - LEAF_AREA_RANGE[0]
    )

    # Clip to guard against out-of-range synthetic artefacts
    chi_norm  = chi_norm.clip(0.0, 1.0)
    area_norm = area_norm.clip(0.0, 1.0)

    # Geometric mean → balanced sensitivity to both sub-indices
    health_index = np.sqrt(chi_norm * area_norm)

    df = df.copy()
    df["health_index"] = np.round(health_index, 6)
    log.info(
        "  H stats → min: %.4f | mean: %.4f | max: %.4f",
        df["health_index"].min(),
        df["health_index"].mean(),
        df["health_index"].max(),
    )
    return df


# ─────────────────────────────────────────────
# Feature scaling
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "pump_power_w", "led_power_w", "cooler_power_w",
    "water_ph", "water_ec", "ambient_temp",
    "water_temp", "humidity",
    "maceration_chlorophyll", "leaf_area_cm2",
]

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MinMax [0, 1] scaling to all continuous feature columns.
    The health_index target is already in [0, 1] and is left untouched.
    """
    log.info("Scaling features to [0, 1]…")
    scaler = MinMaxScaler()
    df     = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    n_steps = N_DAYS * 24 * SAMPLES_PER_HOUR

    df = simulate_growth_cycle(n_steps)
    df = inject_dropouts(df)
    df = clean_data(df)
    df = compute_health_index(df)
    df = scale_features(df)

    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Dataset saved → %s  (%d rows × %d cols)", OUTPUT_FILE, *df.shape)
    log.info("Column summary:\n%s", df.describe().to_string())


if __name__ == "__main__":
    main()