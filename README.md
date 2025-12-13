# 2D Laplace Solver for Stream Function

A comprehensive Python implementation for solving the 2D Laplace equation using various iterative methods. This solver computes the stream function psi(x,y) for incompressible, irrotational flow in a rectangular domain with Dirichlet boundary conditions.

## Physics Background

### Stream Function Definition
For 2D incompressible flow, the stream function psi(x,y) is defined such that:
- u = d(psi)/dy  (velocity in x-direction)
- v = -d(psi)/dx (velocity in y-direction)

### Laplace Equation
For irrotational flow (zero vorticity), the stream function satisfies the Laplace equation:
```
d2(psi)/dx2 + d2(psi)/dy2 = 0
```

## Problem Setup

### Domain
- Physical domain: [0, 10] x [0, 10]
- Grid spacing: dx = dy = 0.1
- Grid size: 101 x 101 points

### Boundary Conditions (Dirichlet)

| Boundary | Condition |
|----------|-----------|
| Left wall (x=0) | psi = 0 for all y |
| Top wall (y=10) | psi = 0 for all x |
| Bottom (y=0, x in [0,2]) | psi = 0 |
| Bottom inlet (y=0, x in [2,2.4]) | psi = 10*(x-2)/0.4 (linear) |
| Bottom (y=0, x in [2.4,10]) | psi = 10 |
| Right (x=10, y in [0,6]) | psi = 10 |
| Right outlet (x=10, y in [6,6.4]) | psi = 10*(6.4-y)/0.4 (linear) |
| Right (x=10, y in [6.4,10]) | psi = 0 |

### Index-Coordinate Mapping
- x_i = i * dx, where i = 0, 1, ..., 100
- y_j = j * dy, where j = 0, 1, ..., 100
- Bottom inlet: i = 20 to 24 (x = 2.0 to 2.4)
- Right outlet: j = 60 to 64 (y = 6.0 to 6.4)

## Installation

### Requirements
- Python 3.8+
- NumPy
- Matplotlib

### Setup
```bash
cd laplace_solver
pip install -r requirements.txt
```

## Usage

### Interactive Menu (Recommended)

Run the interactive menu system:
```bash
python3 menu.py
```

This provides an easy-to-use interface for:
- Running individual solvers
- Comparing all methods
- Running parameter studies
- Changing settings interactively

### Command Line Interface

Run a single solver method:
```bash
python3 main.py --method sor --omega 1.5 --tolerance 1e-7
```

### Available Solver Methods

| Method | Flag | Description |
|--------|------|-------------|
| Point Jacobi | `--method jacobi` | All updates from previous iteration |
| Point Gauss-Seidel | `--method gauss_seidel` | Immediate updates |
| Line Gauss-Seidel | `--method line_gauss_seidel` | Tridiagonal line solves |
| Point SOR | `--method sor` | Over-relaxation |
| Line SOR | `--method line_sor` | Line solve + relaxation |
| ADI | `--method adi` | Alternating Direction Implicit |
| Red-Black SOR | `--method red_black_sor` | Checkerboard ordering |

### Command Line Options

```
--method METHOD          Solver method (default: sor)
--tolerance TOL          Convergence tolerance (default: 1e-7)
--max-iters N            Maximum iterations (default: 100000)
--omega W                Relaxation parameter for SOR (default: 1.5)
--sweep-direction DIR    forward, reverse, or alternating
--line-direction DIR     x or y (for line methods)
--convergence-metric M   residual or update
--init-value V           Initial interior value (default: 0.0)
--output-dir DIR         Output directory (default: outputs)
--verbose                Print progress
```

### Run All Methods
```bash
python main.py --run-all --tolerance 1e-7 --validate
```

### Parameter Studies

#### Omega Optimization Study
Find optimal relaxation parameter for SOR methods:
```bash
python main.py --omega-study --omega-min 1.0 --omega-max 1.98 --omega-step 0.02
```

#### Initial Condition Sensitivity Study
Compare convergence for different initial guesses:
```bash
python main.py --ic-study --omega 1.85
```

#### Sweep Direction Comparison
Compare forward, reverse, and alternating sweeps:
```bash
python main.py --sweep-study
```

### Validation and Testing
```bash
# Run unit tests
python main.py --test

# Run solver with validation
python main.py --run-all --validate
```

## Output Files

All plots are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `streamlines_*.png` | Contour plot of psi (streamlines) |
| `vertical_cut_comparison.png` | psi(y) at x=5.0 for all methods |
| `convergence_comparison.png` | Residual vs iteration for all methods |
| `omega_optimization_study.png` | Iterations vs omega for SOR |
| `initial_condition_study.png` | Convergence for different ICs |
| `sweep_direction_comparison.png` | Sweep direction effect |

## Project Structure

```
laplace_solver/
|-- main.py                  # CLI entry point
|-- requirements.txt         # Python dependencies
|-- README.md               # This file
|-- src/
|   |-- __init__.py
|   |-- grid.py             # Grid class and boundary conditions
|   |-- solvers.py          # All iterative solvers
|   |-- convergence.py      # Convergence metrics
|   |-- plotting.py         # Visualization functions
|   |-- studies.py          # Parameter study functions
|   |-- validation.py       # Validation tests
|-- scripts/
|   |-- run_all_studies.py  # Run complete analysis
|-- outputs/                # Generated plots
|-- report/
|   |-- truncation_error.md # Truncation error derivation
|-- tests/
    |-- test_solvers.py     # Solver unit tests
```

## Numerical Methods

### 5-Point Laplacian Discretization
```
(psi[i+1,j] - 2*psi[i,j] + psi[i-1,j]) / dx^2
+ (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1]) / dy^2 = 0
```

For dx = dy = h:
```
psi[i,j] = 0.25 * (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1])
```

### Convergence Criteria

Two metrics are available:

1. **Residual-based L2**: RMS of discrete Laplacian residual at interior nodes
2. **Update-based L2**: RMS of (psi_new - psi_old) at interior nodes

Default: residual-based with tolerance 1e-7

### Truncation Error

The 5-point Laplacian has truncation error O(dx^2 + dy^2), i.e., second-order accurate.
See `report/truncation_error.md` for the complete derivation.

## Engineering Decisions

1. **Index convention**: psi[i,j] corresponds to (x_i, y_j) where i is x-index, j is y-index
2. **Boundary enforcement**: Applied after every iteration to ensure fixed values
3. **Line direction**: Default x-direction for line solvers
4. **ADI parameter**: Fixed rho=1.0 for steady-state Laplace equation
5. **Convergence check**: Every iteration (can be modified for efficiency)

## Example Results

Typical iteration counts (tolerance 1e-7):

| Method | Iterations | Notes |
|--------|------------|-------|
| Jacobi | ~15000+ | Slowest |
| Gauss-Seidel | ~8000 | 2x faster than Jacobi |
| Point SOR (optimal) | ~500-800 | Requires tuning omega |
| Line SOR (optimal) | ~300-500 | Fastest |
| ADI | ~600-1000 | Good without tuning |
| Red-Black SOR | ~500-800 | Parallelizable |

## License

MIT License

## Author

Generated for CFD coursework - 2D Laplace solver implementation.
