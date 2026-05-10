# 🏛️ Prime Directive: RL Portfolio Optimizer

These directives are the foundational rules for this quantitative research repository. They must be strictly adhered to during all development, refactoring, and feature additions.

> ⚠️ **CRITICAL WARNING:** NO LOOK-AHEAD BIAS. 
> The environment must never expose future data $t+1$ to the agent at time $t$. All observation state features must be constructed strictly from causal, historical data.

---

## 1. Strict Polyglot Architecture

The boundary between the Reinforcement Learning agent and the portfolio simulation is absolute.

*   **C++ Domain (`src/engine.cpp`)**: 
    *   Net Asset Value (NAV) calculations.
    *   Market impact simulations and slippage.
    *   Transaction cost penalties.
    *   Friction math.
*   **Python Domain (`env.py`)**: 
    *   Interaction with the Stable Baselines 3 / PyTorch framework.
    *   Observation space formatting.
    *   Action space translation (e.g., softmax).

> 🚫 **RULE:** The Python wrapper must **never** implement core portfolio simulation logic or internal state transitions.

---

## 2. Mandatory Friction Constraint

Real-world trading is not free. The environment must rigorously simulate execution constraints.

*   **No Idealized Environments:** The portfolio environment must **never** fall back to idealized, frictionless mean-variance optimization.
*   **Continuous Penalties:** The C++ engine must continuously apply both proportional transaction fees (taker fees) and non-linear market impact.
*   **Preservation:** Any modification to the `PortfolioEngine` class must preserve these realistic constraints. A PR removing friction will be immediately rejected.
