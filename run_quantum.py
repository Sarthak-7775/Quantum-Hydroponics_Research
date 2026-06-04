"""
3_run_quantum.py
─────────────────
Variational Quantum Eigensolver (VQE) — Energy-Optimal Hardware Configurator

Maps the hydroponic multi-objective cost function to an Ising Hamiltonian,
then uses Qiskit's VQE with a hardware-efficient RealAmplitudes Ansatz and
the SPSA classical optimizer to find the ground-state (minimum-energy)
3-qubit hardware configuration:

    q₀  →  Water Pump   (+1 = OFF, −1 = ON)
    q₁  →  LED Array    (+1 = LOW, −1 = HIGH)
    q₂  →  Cooler       (+1 = OFF, −1 = ON)

The Health Index H produced by 2_train_model.py informs the penalty term,
ensuring the solver never drops lettuce health below a configurable threshold.

Backends supported
──────────────────
  • qasm_simulator   (default — local Aer simulation)
  • ibm_*            (IBM Quantum real hardware via IBMQ account)

Author  : Hybrid Quantum-Classical CPS Research Group
License : MIT
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Qiskit imports ────────────────────────────────────────────────────────────
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE, MinimumEigensolverResult
from qiskit_algorithms.optimizers import SPSA

# Optional: IBM Quantum real hardware
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

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
RESULTS_FILE    = Path("quantum_results.json")
N_QUBITS        = 3

# Multi-objective weights
ALPHA           = 0.6        # electricity consumption weight
BETA            = 0.4        # health penalty weight

# Health threshold — VQE penalises any state predicted below this
HEALTH_THRESHOLD = 0.70

# Baseline hardware power draw (Watts) for each qubit when ON
POWER_BASELINE  = {
    "pump_w"   : 45.0,
    "led_w"    : 150.0,
    "cooler_w" : 80.0,
}
TOTAL_BASELINE_W = sum(POWER_BASELINE.values())   # normalisation denominator

# SPSA settings
SPSA_MAX_ITER   = 300
SPSA_LAST_AVG   = 50          # average over final N iterations

# IBM backend  (set env IBMQ_TOKEN and IBMQ_BACKEND to use real hardware)
IBM_TOKEN       = os.getenv("IBMQ_TOKEN",   "")
IBM_BACKEND     = os.getenv("IBMQ_BACKEND", "ibm_sherbrooke")
USE_REAL_HW     = bool(IBM_TOKEN) and IBM_RUNTIME_AVAILABLE


# ─────────────────────────────────────────────
# Qubit → hardware state mapping
# ─────────────────────────────────────────────
@dataclass
class HardwareState:
    """
    Decoded 3-qubit measurement result mapped to physical device states.
    Qiskit bitstring ordering: q₂q₁q₀  (rightmost = q₀).
    """
    pump_on  : bool
    led_high : bool
    cooler_on: bool

    @property
    def label(self) -> str:
        return (
            f"Pump={'ON' if self.pump_on else 'OFF'}  "
            f"LED={'HIGH' if self.led_high else 'LOW'}  "
            f"Cooler={'ON' if self.cooler_on else 'OFF'}"
        )

    @property
    def estimated_power_w(self) -> float:
        return (
            (POWER_BASELINE["pump_w"]   if self.pump_on   else POWER_BASELINE["pump_w"]   * 0.05)
            + (POWER_BASELINE["led_w"]  if self.led_high  else POWER_BASELINE["led_w"]    * 0.15)
            + (POWER_BASELINE["cooler_w"] if self.cooler_on else 0.0)
        )


def bitstring_to_hardware(bitstring: str) -> HardwareState:
    """
    Convert a Qiskit measurement bitstring (e.g. '101') to HardwareState.
    Bit '0' → eigenvalue +1 (OFF/LOW), '1' → eigenvalue -1 (ON/HIGH).
    """
    bits = bitstring.zfill(N_QUBITS)
    return HardwareState(
        pump_on  = bits[-1] == "1",   # q₀ rightmost
        led_high = bits[-2] == "1",   # q₁
        cooler_on= bits[-3] == "1",   # q₂
    )


# ─────────────────────────────────────────────
# Hamiltonian construction
# ─────────────────────────────────────────────
def build_hamiltonian(health_index: float) -> SparsePauliOp:
    """
    Construct the 3-qubit Ising Problem Hamiltonian H:

        H = c₀Z₀ + c₁Z₁ + c₂Z₂
            + c₀₁(Z₀⊗Z₁) + c₁₂(Z₁⊗Z₂) + c₀₂(Z₀⊗Z₂)

    Linear coefficients encode the normalised electricity cost of each device.
    Quadratic coefficients encode correlated health penalties.

    Health penalty logic
    ────────────────────
    If H < HEALTH_THRESHOLD, turning OFF the pump AND cooler simultaneously is
    dangerous (nutrient stagnation + thermal stress).  We elevate c₀₂ sharply.
    Turning off LEDs when H is already low also suppresses photosynthesis, so
    we couple LED–cooler with a moderate penalty.

    Parameters
    ----------
    health_index : float — latest predicted H from the GRU model ∈ [0, 1]

    Returns
    -------
    SparsePauliOp representing H in Qiskit's Pauli string notation.
    Qiskit ordering: rightmost character = qubit 0.
    """
    # Normalised linear energy costs  (ON = eigenvalue −1 reduces H)
    c0 = ALPHA * (POWER_BASELINE["pump_w"]    / TOTAL_BASELINE_W)
    c1 = ALPHA * (POWER_BASELINE["led_w"]     / TOTAL_BASELINE_W)
    c2 = ALPHA * (POWER_BASELINE["cooler_w"]  / TOTAL_BASELINE_W)

    # Base quadratic coupling (mild — prefers co-activation)
    c01 = 0.05
    c12 = 0.05
    c02 = 0.05

    # ── Health-driven penalty escalation ─────────────────────────────────
    deficit = max(0.0, HEALTH_THRESHOLD - health_index)

    # Pump–cooler simultaneous-off penalty (nutrient + heat risk)
    c02 += BETA * deficit * 2.5

    # LED–cooler coupling (low light + high heat is lethal)
    c12 += BETA * deficit * 1.5

    # Pump–LED coupling (dark + stagnant water)
    c01 += BETA * deficit * 1.0

    log.info(
        "Hamiltonian coefficients  →  "
        "c₀=%.4f  c₁=%.4f  c₂=%.4f  "
        "c₀₁=%.4f  c₁₂=%.4f  c₀₂=%.4f",
        c0, c1, c2, c01, c12, c02,
    )

    # Build SparsePauliOp (Qiskit convention: 'IIZ' = Z on qubit 0)
    hamiltonian = SparsePauliOp.from_list([
        ("IIZ", c0),    # Z₀
        ("IZI", c1),    # Z₁
        ("ZII", c2),    # Z₂
        ("IZZ", c01),   # Z₀⊗Z₁
        ("ZZI", c12),   # Z₁⊗Z₂
        ("ZIZ", c02),   # Z₀⊗Z₂
    ])

    return hamiltonian


# ─────────────────────────────────────────────
# VQE execution
# ─────────────────────────────────────────────
def run_vqe(hamiltonian: SparsePauliOp) -> MinimumEigensolverResult:
    """
    Execute the Variational Quantum Eigensolver using:
        Ansatz   : RealAmplitudes (linear entanglement, reps=3)
        Optimizer: SPSA (handles noisy gradient landscapes)
        Backend  : StatevectorEstimator (simulation) or IBM Runtime

    The RealAmplitudes circuit minimises gate depth while providing
    sufficient expressibility for the 3-qubit QUBO landscape.
    """
    # ── Ansatz ────────────────────────────────────────────────────────────
    ansatz = RealAmplitudes(
        num_qubits  = N_QUBITS,
        entanglement= "linear",
        reps        = 3,
    )

    # ── Classical optimizer ────────────────────────────────────────────────
    spsa_optimizer = SPSA(
        maxiter = SPSA_MAX_ITER,
        last_avg= SPSA_LAST_AVG,
        callback= _spsa_callback,
    )

    # ── Estimator primitive ────────────────────────────────────────────────
    if USE_REAL_HW:
        log.info("Connecting to IBM Quantum backend: %s", IBM_BACKEND)
        service  = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
        backend  = service.backend(IBM_BACKEND)
        estimator= RuntimeEstimator(backend=backend)
    else:
        log.info("Using local StatevectorEstimator (noise-free simulation).")
        estimator = StatevectorEstimator()

    # ── VQE ────────────────────────────────────────────────────────────────
    vqe    = VQE(estimator=estimator, ansatz=ansatz, optimizer=spsa_optimizer)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    return result


# Callback to log SPSA convergence
_convergence_log: list[float] = []

def _spsa_callback(
    nfev:    int,
    x:       np.ndarray,
    fx:      float,
    dx:      np.ndarray,
    accept:  bool,
) -> None:
    _convergence_log.append(float(fx))
    if nfev % 50 == 0:
        log.info("  SPSA iter %4d  |  ⟨H⟩ = %.6f", nfev, fx)


# ─────────────────────────────────────────────
# Ground-state decoding
# ─────────────────────────────────────────────
def decode_result(result: MinimumEigensolverResult) -> dict[str, Any]:
    """
    Extract the optimal eigenstate from the VQE result and map it to the
    physical hardware configuration with the lowest energy (cost).
    """
    eigenvalue  = float(result.eigenvalue.real)
    eigenstate  = result.eigenstate          # dict[bitstring, amplitude] or array

    # Determine the most-probable bitstring
    if hasattr(eigenstate, "to_dict"):
        probs       = {k: abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    elif isinstance(eigenstate, dict):
        probs       = {k: abs(v) ** 2 for k, v in eigenstate.items()}
    else:
        # Statevector array
        sv          = np.asarray(eigenstate).flatten()
        probs       = {
            format(i, f"0{N_QUBITS}b"): abs(sv[i]) ** 2
            for i in range(len(sv))
        }

    # Most probable computational basis state
    optimal_bitstring = max(probs, key=probs.get)
    hw_state          = bitstring_to_hardware(optimal_bitstring)

    log.info("──── VQE Result ──────────────────────────────────────")
    log.info("  Minimum eigenvalue (ground-state energy): %.6f", eigenvalue)
    log.info("  Optimal bitstring : %s", optimal_bitstring)
    log.info("  Hardware config   : %s", hw_state.label)
    log.info("  Estimated power   : %.1f W", hw_state.estimated_power_w)
    log.info("─────────────────────────────────────────────────────")

    return {
        "eigenvalue"          : eigenvalue,
        "optimal_bitstring"   : optimal_bitstring,
        "hardware_state"      : {
            "pump_on"  : hw_state.pump_on,
            "led_high" : hw_state.led_high,
            "cooler_on": hw_state.cooler_on,
        },
        "label"               : hw_state.label,
        "estimated_power_w"   : hw_state.estimated_power_w,
        "convergence_curve"   : _convergence_log,
        "state_probabilities" : {k: round(v, 6) for k, v in sorted(
            probs.items(), key=lambda x: -x[1])[:8]
        },
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main(health_index: float = 0.72) -> None:
    """
    Parameters
    ----------
    health_index : float
        The most recent Plant Health Index output from the GRU model.
        Defaults to 0.72 for standalone testing.
        In production, pass the live inference result from 2_train_model.py.
    """
    log.info("Starting VQE optimisation  (H = %.4f)", health_index)

    hamiltonian = build_hamiltonian(health_index)
    log.info("Hamiltonian:\n%s", hamiltonian)

    result = run_vqe(hamiltonian)
    output = decode_result(result)
    output["input_health_index"] = health_index
    output["alpha"] = ALPHA
    output["beta"]  = BETA

    with open(RESULTS_FILE, "w") as fh:
        json.dump(output, fh, indent=2)

    log.info("Quantum results saved → %s", RESULTS_FILE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run VQE to find optimal hydroponic hardware configuration."
    )
    parser.add_argument(
        "--health-index",
        type=float,
        default=0.72,
        help="Plant Health Index from GRU inference (default: 0.72)",
    )
    args = parser.parse_args()
    main(health_index=args.health_index)