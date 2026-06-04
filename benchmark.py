"""
4_benchmark.py
───────────────
Classical vs Quantum Benchmark — Optimization Algorithm Comparison

Rigorously benchmarks the VQE framework against two classical heuristics:
  • Simulated Annealing  (SA)
  • Genetic Algorithm    (GA)

All three algorithms solve the same 3-qubit Ising Hamiltonian for a range of
health-index inputs and are compared on:
  ──────────────────────────────────────────────────────────────────
  Metric               Description
  ──────────────────────────────────────────────────────────────────
  best_energy          Minimum cost (ground-state energy) found
  convergence_iters    Iterations until |ΔE| < ε tolerance
  wall_time_s          Wallclock seconds for optimisation
  energy_gap_to_vqe    Best_energy − VQE_energy  (positive = VQE wins)
  ──────────────────────────────────────────────────────────────────

Results are written to benchmark_results.json and a matplotlib summary
figure is saved to benchmark_summary.png.

Author  : Hybrid Quantum-Classical CPS Research Group
License : MIT
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import numpy as np

# Import Hamiltonian builder from the quantum module
from run_quantum import build_hamiltonian   # type: ignore[import]
from qiskit.quantum_info import SparsePauliOp

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
# Configuration
# ─────────────────────────────────────────────
RESULTS_FILE     = Path("benchmark_results.json")
FIGURE_FILE      = Path("benchmark_summary.png")

HEALTH_TEST_VALS = [0.30, 0.50, 0.70, 0.90]    # range of H to benchmark
CONVERGENCE_TOL  = 1e-5                          # |ΔE| threshold
N_RUNS           = 5                             # runs per algorithm per H (variance analysis)
RANDOM_SEED      = 42

# SA hyper-parameters
SA_T_INIT        = 5.0
SA_T_MIN         = 1e-5
SA_ALPHA         = 0.97
SA_MAX_ITER      = 2_000

# GA hyper-parameters
GA_POP_SIZE      = 32
GA_MAX_GEN       = 200
GA_CROSSOVER_P   = 0.80
GA_MUTATION_P    = 0.10
GA_ELITE_K       = 4

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# Energy evaluation: Ising Hamiltonian on ±1
# ─────────────────────────────────────────────
def hamiltonian_energy(spins: np.ndarray, hamiltonian: SparsePauliOp) -> float:
    """
    Evaluate ⟨H⟩ for a classical spin configuration spins ∈ {+1, −1}^N.

    Parameters
    ----------
    spins       : array of shape [N_QUBITS] with values ±1
    hamiltonian : SparsePauliOp (Qiskit)

    Returns
    -------
    Float scalar — the classical energy of this configuration.
    """
    energy = 0.0
    for pauli_str, coeff in zip(
        hamiltonian.paulis.to_labels(), hamiltonian.coeffs
    ):
        term = float(coeff.real)
        for i, char in enumerate(reversed(pauli_str)):   # rightmost = qubit 0
            if char == "Z":
                term *= spins[i]
        energy += term
    return energy


def all_configurations(n: int = 3) -> list[np.ndarray]:
    """Enumerate all 2^n spin configurations ∈ {+1, -1}^n."""
    configs = []
    for mask in range(2 ** n):
        spins = np.array(
            [1 if (mask >> i) & 1 == 0 else -1 for i in range(n)],
            dtype=np.float64,
        )
        configs.append(spins)
    return configs


def brute_force_minimum(hamiltonian: SparsePauliOp) -> tuple[float, np.ndarray]:
    """
    Exact ground-state via exhaustive enumeration (feasible for 3 qubits).
    Used as the reference optimum in benchmarking.
    """
    configs = all_configurations(N_QUBITS := 3)
    energies = [hamiltonian_energy(c, hamiltonian) for c in configs]
    idx = int(np.argmin(energies))
    return energies[idx], configs[idx]


# ─────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────
def simulated_annealing(
    hamiltonian: SparsePauliOp,
    t_init: float = SA_T_INIT,
    t_min:  float = SA_T_MIN,
    alpha:  float = SA_ALPHA,
    max_iter: int = SA_MAX_ITER,
) -> dict[str, Any]:
    """
    Metropolis–Hastings Simulated Annealing over the Ising spin space.

    Perturbation: single random spin-flip at each step.
    Acceptance  : exp(−ΔE / T) ≥ U[0,1]
    """
    n          = 3
    spins      = np.random.choice([-1, 1], size=n).astype(np.float64)
    energy     = hamiltonian_energy(spins, hamiltonian)
    best_spins = spins.copy()
    best_e     = energy
    T          = t_init

    curve      = [energy]
    conv_iter  = max_iter
    prev_best  = float("inf")

    for it in range(1, max_iter + 1):
        flip_idx            = np.random.randint(n)
        candidate           = spins.copy()
        candidate[flip_idx] *= -1
        cand_e              = hamiltonian_energy(candidate, hamiltonian)
        delta               = cand_e - energy

        if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-300)):
            spins  = candidate
            energy = cand_e

        if energy < best_e:
            best_e     = energy
            best_spins = spins.copy()

        curve.append(best_e)

        if abs(best_e - prev_best) < CONVERGENCE_TOL and conv_iter == max_iter:
            conv_iter = it
        prev_best = best_e

        T = max(T * alpha, t_min)

    return {
        "best_energy"       : best_e,
        "best_spins"        : best_spins.tolist(),
        "convergence_iters" : conv_iter,
        "energy_curve"      : curve,
    }


# ─────────────────────────────────────────────
# Genetic Algorithm
# ─────────────────────────────────────────────
def genetic_algorithm(
    hamiltonian: SparsePauliOp,
    pop_size:   int   = GA_POP_SIZE,
    max_gen:    int   = GA_MAX_GEN,
    cx_prob:    float = GA_CROSSOVER_P,
    mut_prob:   float = GA_MUTATION_P,
    elite_k:    int   = GA_ELITE_K,
) -> dict[str, Any]:
    """
    Binary-encoded Genetic Algorithm with elitism, single-point crossover,
    and bit-flip mutation.

    Encoding: bit 0 → +1 (OFF), bit 1 → −1 (ON).
    Fitness  : −energy (maximise fitness = minimise energy).
    """
    n = 3

    def decode(chromosome: np.ndarray) -> np.ndarray:
        return np.where(chromosome == 0, 1.0, -1.0).astype(np.float64)

    def fitness(chromosome: np.ndarray) -> float:
        return -hamiltonian_energy(decode(chromosome), hamiltonian)

    # Initialise population
    population = np.random.randint(0, 2, size=(pop_size, n))
    curve      = []
    conv_iter  = max_gen
    prev_best  = float("inf")

    for gen in range(1, max_gen + 1):
        fitnesses  = np.array([fitness(ind) for ind in population])
        best_idx   = int(np.argmax(fitnesses))
        best_e     = -fitnesses[best_idx]
        curve.append(best_e)

        if abs(best_e - prev_best) < CONVERGENCE_TOL and conv_iter == max_gen:
            conv_iter = gen
        prev_best = best_e

        # Elitism — carry top-k unchanged
        elite_idx   = np.argsort(fitnesses)[-elite_k:]
        elite       = population[elite_idx].copy()

        # Tournament selection (size 3)
        def select() -> np.ndarray:
            competitors = np.random.choice(pop_size, size=3, replace=False)
            winner      = competitors[np.argmax(fitnesses[competitors])]
            return population[winner].copy()

        new_pop = list(elite)
        while len(new_pop) < pop_size:
            parent_a, parent_b = select(), select()

            # Single-point crossover
            if np.random.rand() < cx_prob:
                point  = np.random.randint(1, n)
                child  = np.concatenate([parent_a[:point], parent_b[point:]])
            else:
                child  = parent_a.copy()

            # Bit-flip mutation
            for locus in range(n):
                if np.random.rand() < mut_prob:
                    child[locus] ^= 1

            new_pop.append(child)

        population = np.array(new_pop[:pop_size])

    # Final evaluation
    fitnesses  = np.array([fitness(ind) for ind in population])
    best_idx   = int(np.argmax(fitnesses))
    best_e     = -fitnesses[best_idx]
    best_spins = decode(population[best_idx])

    return {
        "best_energy"       : best_e,
        "best_spins"        : best_spins.tolist(),
        "convergence_iters" : conv_iter,
        "energy_curve"      : curve,
    }


# ─────────────────────────────────────────────
# VQE stub (imports live result or re-runs)
# ─────────────────────────────────────────────
def vqe_energy_from_results(health_index: float) -> float | None:
    """
    Attempt to load a cached VQE result from quantum_results.json.
    Returns None if not available (triggers re-run notice).
    """
    results_path = Path("quantum_results.json")
    if not results_path.exists():
        return None
    with open(results_path) as fh:
        data = json.load(fh)
    if abs(data.get("input_health_index", -1) - health_index) < 0.01:
        return data.get("eigenvalue")
    return None


def run_vqe_direct(health_index: float, hamiltonian: SparsePauliOp) -> dict[str, Any]:
    """
    Lightweight VQE invocation for benchmark comparison.
    Wraps the simulation path from 3_run_quantum.py.
    """
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SPSA

    ansatz    = RealAmplitudes(num_qubits=3, entanglement="linear", reps=3)
    optimizer = SPSA(maxiter=300)
    estimator = StatevectorEstimator()
    vqe       = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result    = vqe.compute_minimum_eigenvalue(hamiltonian)
    return {"best_energy": float(result.eigenvalue.real)}


# ─────────────────────────────────────────────
# Single benchmark run
# ─────────────────────────────────────────────
def _timed_run(fn: Callable, *args, **kwargs) -> tuple[dict[str, Any], float]:
    t0     = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def benchmark_one(
    health_index: float,
    hamiltonian:  SparsePauliOp,
) -> dict[str, Any]:
    """
    Run all three algorithms on a single Hamiltonian and return comparative stats.
    """
    exact_energy, _ = brute_force_minimum(hamiltonian)
    log.info("  Exact ground-state energy: %.6f", exact_energy)

    results = {}

    # ── Simulated Annealing ───────────────────────────────────────────────
    sa_energies, sa_times, sa_conv = [], [], []
    for run in range(N_RUNS):
        np.random.seed(RANDOM_SEED + run)
        res, t  = _timed_run(simulated_annealing, hamiltonian)
        sa_energies.append(res["best_energy"])
        sa_times.append(t)
        sa_conv.append(res["convergence_iters"])

    results["simulated_annealing"] = {
        "best_energy"       : float(np.min(sa_energies)),
        "energy_mean"       : float(np.mean(sa_energies)),
        "energy_std"        : float(np.std(sa_energies)),
        "wall_time_s_mean"  : float(np.mean(sa_times)),
        "convergence_mean"  : float(np.mean(sa_conv)),
        "energy_gap_to_exact": float(np.min(sa_energies) - exact_energy),
    }
    log.info("  SA  best=%.6f  gap=%.6f", results["simulated_annealing"]["best_energy"],
             results["simulated_annealing"]["energy_gap_to_exact"])

    # ── Genetic Algorithm ─────────────────────────────────────────────────
    ga_energies, ga_times, ga_conv = [], [], []
    for run in range(N_RUNS):
        np.random.seed(RANDOM_SEED + run)
        res, t  = _timed_run(genetic_algorithm, hamiltonian)
        ga_energies.append(res["best_energy"])
        ga_times.append(t)
        ga_conv.append(res["convergence_iters"])

    results["genetic_algorithm"] = {
        "best_energy"       : float(np.min(ga_energies)),
        "energy_mean"       : float(np.mean(ga_energies)),
        "energy_std"        : float(np.std(ga_energies)),
        "wall_time_s_mean"  : float(np.mean(ga_times)),
        "convergence_mean"  : float(np.mean(ga_conv)),
        "energy_gap_to_exact": float(np.min(ga_energies) - exact_energy),
    }
    log.info("  GA  best=%.6f  gap=%.6f", results["genetic_algorithm"]["best_energy"],
             results["genetic_algorithm"]["energy_gap_to_exact"])

    # ── VQE ───────────────────────────────────────────────────────────────
    cached_vqe = vqe_energy_from_results(health_index)
    if cached_vqe is not None:
        vqe_energy = cached_vqe
        vqe_time   = 0.0
        log.info("  VQE  energy=%.6f  (loaded from cache)", vqe_energy)
    else:
        log.info("  Running VQE live for H=%.2f…", health_index)
        res, vqe_time = _timed_run(run_vqe_direct, health_index, hamiltonian)
        vqe_energy    = res["best_energy"]

    results["vqe"] = {
        "best_energy"        : vqe_energy,
        "wall_time_s"        : vqe_time,
        "energy_gap_to_exact": vqe_energy - exact_energy,
    }
    log.info("  VQE best=%.6f  gap=%.6f", vqe_energy, results["vqe"]["energy_gap_to_exact"])

    results["exact_energy"]    = exact_energy
    results["health_index"]    = health_index

    # ── Comparative deltas ────────────────────────────────────────────────
    results["vqe_vs_sa_delta"] = float(
        vqe_energy - results["simulated_annealing"]["best_energy"]
    )
    results["vqe_vs_ga_delta"] = float(
        vqe_energy - results["genetic_algorithm"]["best_energy"]
    )
    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_benchmark(all_results: list[dict[str, Any]]) -> None:
    health_vals = [r["health_index"]    for r in all_results]
    vqe_gaps    = [r["vqe"]["energy_gap_to_exact"]                          for r in all_results]
    sa_gaps     = [r["simulated_annealing"]["energy_gap_to_exact"]          for r in all_results]
    ga_gaps     = [r["genetic_algorithm"]["energy_gap_to_exact"]            for r in all_results]
    sa_times    = [r["simulated_annealing"]["wall_time_s_mean"]             for r in all_results]
    ga_times    = [r["genetic_algorithm"]["wall_time_s_mean"]               for r in all_results]
    vqe_times   = [r["vqe"]["wall_time_s"]                                  for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Quantum vs Classical Benchmark — Hydroponic Ising Optimizer",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Energy gap to exact ground state ─────────────────────────
    ax = axes[0]
    x  = np.arange(len(health_vals))
    w  = 0.25
    ax.bar(x - w, vqe_gaps, width=w, label="VQE",                color="#4C72B0", alpha=0.9)
    ax.bar(x,     sa_gaps,  width=w, label="Simulated Annealing", color="#DD8452", alpha=0.9)
    ax.bar(x + w, ga_gaps,  width=w, label="Genetic Algorithm",   color="#55A868", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in health_vals])
    ax.set_ylabel("Energy Gap to Exact Minimum  (lower = better)")
    ax.set_title("Solution Quality")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Wall-clock time ───────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - w, vqe_times, width=w, label="VQE",                color="#4C72B0", alpha=0.9)
    ax.bar(x,     sa_times,  width=w, label="Simulated Annealing", color="#DD8452", alpha=0.9)
    ax.bar(x + w, ga_times,  width=w, label="Genetic Algorithm",   color="#55A868", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in health_vals])
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_title("Computational Overhead")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    log.info("Benchmark figure saved → %s", FIGURE_FILE)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    log.info("Starting benchmark over %d health-index values…", len(HEALTH_TEST_VALS))
    all_results = []

    for h in HEALTH_TEST_VALS:
        log.info("═══  H = %.2f  ═══════════════════════════════════", h)
        hamiltonian = build_hamiltonian(h)
        result      = benchmark_one(h, hamiltonian)
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    log.info("\n%s", "=" * 72)
    log.info("%-6s  %-10s  %-10s  %-10s  %-12s  %-12s",
             "H", "VQE_gap", "SA_gap", "GA_gap", "VQE_vs_SA", "VQE_vs_GA")
    log.info("-" * 72)
    for r in all_results:
        log.info(
            "%-6.2f  %-10.6f  %-10.6f  %-10.6f  %-12.6f  %-12.6f",
            r["health_index"],
            r["vqe"]["energy_gap_to_exact"],
            r["simulated_annealing"]["energy_gap_to_exact"],
            r["genetic_algorithm"]["energy_gap_to_exact"],
            r["vqe_vs_sa_delta"],
            r["vqe_vs_ga_delta"],
        )
    log.info("=" * 72)

    # ── Persist results ───────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("Benchmark results saved → %s", RESULTS_FILE)

    plot_benchmark(all_results)


if __name__ == "__main__":
    main()