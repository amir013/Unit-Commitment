# Unit Commitment Optimization

Mixed-Integer Linear Programming (MILP) solution for day-ahead unit commitment with renewable energy integration.

## Overview

This project solves the unit commitment problem for a 24-hour day-ahead schedule, optimizing the dispatch of four conventional generators while integrating wind and solar renewable energy sources. The optimization minimizes operational costs while maintaining power balance and respecting generator constraints.

## Features

- **Multi-Generator Optimization**: Coordinates 4 thermal generators with different cost characteristics
- **Renewable Integration**: Incorporates wind and solar generation with probabilistic forecasts
- **Binary On/Off Decisions**: Determines optimal generator commitment (u₁, u₂, u₃, u₄)
- **Constraint Handling**: Enforces min/max power limits with penalty-based soft constraints
- **Imbalance Management**: Handles power imbalance between generation and load
- **Expected Value Approach**: Uses probabilistic wind and solar forecasts

## Problem Formulation

### Decision Variables
- **pᵢ[t]**: Power output of generator i at time t
- **uᵢ[t]**: Binary variable (1 if generator i is on, 0 otherwise)
- **u_wind[t]**, **u_solar[t]**: Renewable generation commitment
- **imbalance[t]**: Power imbalance (positive or negative)

### Objective Function
Minimize total daily cost:
```
Cost = Σₜ Σᵢ (aᵢ·pᵢ²[t] + bᵢ·pᵢ[t]) + imbalance_penalty + violation_penalties
```

### Constraints
- **Power Balance**: Generation = Load - Renewables + Imbalance
- **Generator Limits**: pₘᵢₙ ≤ pᵢ ≤ pₘₐₓ (when online)
- **Binary Logic**: Power output only when generator is committed
- **Renewable Availability**: Wind and solar generation based on forecasts

## Data Requirements

### Input Files
1. **DataC_hourlyload.csv**: Hourly electricity demand (MWh)
2. **DataC_nondispgeneration.csv**: Wind and solar generation scenarios with probabilities
3. **DataC_generatorparams.csv**: Generator cost coefficients and capacity limits

### Generator Parameters
| Generator | a1 ($/MW) | a2 ($/MW²) | Pmin (MW) | Pmax (MW) |
|-----------|-----------|------------|-----------|-----------|
| 1         | 5         | 0.100      | 2000      | 16000     |
| 2         | 6         | 0.200      | 1800      | 23000     |
| 3         | 7         | 0.015      | 100       | 12000     |
| 4         | 4         | 0.018      | 500       | 15000     |

## Requirements

```
python>=3.7
pandas
pyomo
gurobi
```

## Installation

```bash
pip install pandas pyomo
# Gurobi requires separate license and installation
```

## Usage

Run the Jupyter notebook `model_c.ipynb`:

```python
# Load data
load = pd.read_csv('DataC_hourlyload.csv', sep=';')
renewables = pd.read_csv('DataC_nondispgeneration.csv')
gen_params = pd.read_csv('DataC_generatorparams.csv', sep=';')

# Create and solve model
model = pyo.ConcreteModel()
# ... (model setup)
optim = pyo.SolverFactory('gurobi')
result = optim.solve(model)
```

## Renewable Energy Modeling

### Expected Value Calculation
Wind and solar generation are modeled using expected values from probabilistic scenarios:

```
E[Generation] = Σ (Probability × Generation Scenario)
```

### Time Blocks
- **0-5h**: Wind only (no solar)
- **6-10h**: Wind + low solar
- **11-14h**: Wind + peak solar
- **15-19h**: Wind + moderate solar
- **20-23h**: Wind only (no solar)

## Results

The solver outputs:
- 24-hour generator commitment schedule (binary on/off)
- Hourly power output for each generator
- Wind and solar utilization
- Total operational cost
- Power imbalance at each hour

## Solver

Uses **Gurobi** optimizer for MILP solving:
- Fast convergence (~0.5 seconds)
- Handles integer variables efficiently
- Optimal solution guaranteed

## Applications

- Day-ahead electricity market bidding
- Power system planning with renewables
- Grid operator decision support
- Renewable energy integration studies

## License

This project is available for educational and research purposes.
