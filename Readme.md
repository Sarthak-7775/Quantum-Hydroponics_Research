# Hybrid Quantum-Classical Cyber-Physical System for Hydroponic Lettuce Optimization

## Overview
This repository contains the methodology and codebase for a hybrid quantum-classical optimization framework applied to a controlled-environment hydroponics system cultivating *Lactuca sativa* (Lettuce). The system bridges classical Deep Learning with Quantum Computing to form a continuous closed feedback loop. 

Classical neural networks handle **perception** by parsing noisy time-series sensor data to predict a continuous Plant Health Index. Utilizing this index, a Quantum Computer handles **prescriptive action** by mapping hardware configurations to a quantum optimization landscape, finding the exact operational state that maximizes lettuce health while strictly minimizing electricity consumption.

## Core Architecture: The Hybrid Loop
1. **Classical AI (Perception):** A predictive time-series neural network maps environmental inputs (pH, EC, ambient temperature, humidity) to a decimal Health Index between 0.0 (critical) and 1.0 (optimal peak health). 
2. **Quantum Optimizer (Action):** The Health Index informs a multi-objective cost function within a Quadratic Unconstrained Binary Optimization (QUBO) formulation. 
3. **Variational Quantum Eigensolver (VQE):** A parameterized quantum circuit (Ansatz) interacts with a classical optimizer to iteratively search the mathematical landscape, safely minimizing kilowatt-hours (kWh) without dropping the predicted Health Index below a target threshold.

## Mathematical Formulation
To map the physical hydroponics environment to a quantum processor, we translate the optimization problem into an **Ising Hamiltonian**. We define a **3-qubit system** representing our core hardware variables:
* **q0 (Water Pump State):** Maps to +1 (OFF) or -1 (ON).
* **q1 (LED Array State):** Maps to +1 (LOW) or -1 (HIGH).
* **q2 (Cooler State):** Maps to +1 (OFF) or -1 (ON).

### The Cost Function
The goal is to minimize a multi-objective cost function that balances electricity consumption against a penalty for endangering the predicted Lettuce Health Index. 

### The Problem Hamiltonian
When mapped to Pauli-Z operators, the cost function yields the Problem Hamiltonian, which is composed of:
* **Linear Terms:** Represent the base electricity consumption cost of running individual hardware components.
* **Quadratic Terms:** Represent correlated penalties (e.g., dynamically elevating the "energy" of the system if the algorithm attempts to turn off both the pump and cooler while ambient temperature is dangerously high).

## Quantum Optimization
The framework relies on the **Rayleigh-Ritz variational principle**, which states that the expectation value of the Hamiltonian for any parameterized trial state is always greater than or equal to the true ground state energy (lowest eigenvalue).

The VQE algorithm iteratively updates the parameter vector to drive the expectation value down to its minimum value. This mathematical ground state maps directly to the safest, most energy-efficient hardware configuration.

## Algorithmic Pipeline & Tech Stack
To ensure computational efficiency and address the constraints of the Noisy Intermediate-Scale Quantum (NISQ) era, the following stack is utilized:

| Component | Selected Technology | Academic Justification & Use |
| :--- | :--- | :--- |
| **Classical Perception** | **Gated Recurrent Unit (GRU)** | Detects rapid shifts in time-series environmental sensor data with significantly lower computational overhead than transformer models. |
| **Quantum Algorithm** | **Variational Quantum Eigensolver (VQE)** | The industry standard for near-term hybrid optimization. Highly publishable and tailored perfectly for current NISQ-era constraints. |
| **Circuit Ansatz** | **`RealAmplitudes` (Linear Entanglement)** | A parameterized hardware-efficient quantum circuit that minimizes gate depth, dramatically lowering the risk of decoherence errors on cloud-based quantum processors. |
| **Classical Optimizer** | **SPSA** | Simultaneous Perturbation Stochastic Approximation. Efficiently handles and navigates the inherent computational readout noise of cloud-based IBM quantum backends. |
| **Benchmark Baseline** | **Simulated Annealing (SA)** | Provides a direct, peer-reviewed classical point of comparison to demonstrate the algorithmic accuracy and convergence rate of the quantum framework. |

## Classical vs. Quantum Benchmark
To rigorously evaluate the framework's performance, a benchmarking module pits the Qiskit VQE against classical heuristic optimization algorithms (Simulated Annealing and Genetic Algorithms). The algorithms are compared side-by-side on convergence runtime, computational resource overhead, and the absolute accuracy of finding the global minimum. This establishes a baseline proving the viability of VQE as quantum hardware scales toward supremacy.

## Repository Structure
* `hydro_data.csv`: Time-series metrics (electricity consumption vs. lettuce efficiency indicators like maceration data, leaf area, and wet/dry biomass).
* `1_generate_data.py`: Simulates the biological digital twin to establish baselines and synthesize nominal growth and energy curves.
* `2_train_model.py`: Compiles the PyTorch GRU model to forecast the continuous Health Index.
* `3_run_quantum.py`: Executes the Qiskit VQE optimizer loop on IBM Quantum infrastructure.
* `4_benchmark.py`: Runs Simulated Annealing to benchmark against the VQE performance.