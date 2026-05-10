# Prime Directive: RL Portfolio Optimizer

## 1. Strict Polyglot Architecture
All Net Asset Value (NAV) calculations, market impact simulations, and transaction cost penalties **MUST** remain within the high-performance C++ backend engine (`src/engine.cpp`). The Python wrapper is strictly for interacting with the Reinforcement Learning framework and must not implement core portfolio simulation logic.

## 2. Mandatory Friction Constraint
The portfolio environment must **never** fall back to idealized, frictionless mean-variance optimization. The C++ engine must continuously apply both proportional transaction fees (taker fees) and non-linear market impact to accurately simulate trading in volatile equities. Any modification to the engine must preserve these realistic constraints.
