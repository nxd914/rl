# Institutional Polyglot RL Portfolio Optimizer

## Executive Summary
This repository implements a production-grade Reinforcement Learning (RL) agent for dynamic portfolio rebalancing across a basket of 10 volatile equities. The system relies on a **polyglot architecture**: a Proximal Policy Optimization (PPO) agent manages the portfolio through a custom `Farama Gymnasium` environment, while the core environment mechanics—including the critical Net Asset Value (NAV) state transitions and friction math—are handled by a high-performance C++ backend bound to Python via `pybind11`. 

## The Core Math: Friction & Execution Simulation
Traditional mean-variance models fail because they ignore the real-world costs of rebalancing. This engine solves that by directly calculating friction costs *before* returning the step-wise reward to the RL agent.

The C++ engine applies a rigorous two-part penalty function whenever the agent alters its portfolio weights:

1. **Transaction Costs (Taker Fee):** 
   A proportional cost model representing exchange taker fees. The agent pays a fixed basis point (bps) fee (e.g., 10 bps) on the absolute change in portfolio weights.
   $$ \text{Cost}_{\text{taker}} = \text{taker\_fee\_bps} \times \sum | \text{target\_weights} - \text{current\_weights} | \times \text{NAV} $$

2. **Market Impact (Slippage):**
   A non-linear penalty designed to simulate the market impact of placing large orders. We employ a $1.5$ power function to appropriately penalize significant, sudden shifts in allocation.
   $$ \text{Cost}_{\text{impact}} = \text{impact\_coeff} \times \sum \left( | \text{target\_weights} - \text{current\_weights} |^{1.5} \right) \times \text{NAV} $$

The total rebalancing cost is immediately deducted from the portfolio's NAV prior to calculating the period's market return. The step reward returned to the PPO agent is the fractional change in NAV net of these critical trading costs.

## Installation & Build Instructions

To compile the C++ math engine and install the Python bindings:

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install pybind11 gymnasium stable-baselines3 torch numpy

# 3. Compile and install the pybind11 extension module
pip install -e .
```

To run the training script and evaluate the PPO agent:
```bash
python3 train.py
```
