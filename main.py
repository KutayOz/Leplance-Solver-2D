#!/usr/bin/env python3
"""
2D Laplace Solver - Main Entry Point
====================================

This script provides a command-line interface for solving the 2D Laplace equation
for the stream function psi(x,y) using various iterative methods.

Physics:
--------
2D incompressible flow stream function definition:
    u = d(psi)/dy      (velocity in x-direction)
    v = -d(psi)/dx     (velocity in y-direction)

For irrotational flow (zero vorticity), we have the Laplace equation:
    d2(psi)/dx2 + d2(psi)/dy2 = 0

Usage:
------
    python main.py --method sor --omega 1.5 --tolerance 1e-7
    python main.py --run-all
    python main.py --omega-study
    python main.py --ic-study
    python main.py --validate

See README.md for complete documentation.
"""

import argparse
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.grid import Grid
from src.solvers import (
    jacobi_solver, gauss_seidel_solver, line_gauss_seidel_solver,
    sor_solver, line_sor_solver, adi_solver, red_black_sor_solver,
    get_solver, SOLVER_METHODS
)
from src.convergence import compute_residual_l2, compare_solutions
from src.plotting import (
    plot_vertical_cut_comparison, plot_streamlines,
    plot_omega_study, plot_initial_condition_study,
    plot_convergence_comparison, plot_sweep_direction_comparison
)
from src.studies import (
    run_omega_study, run_initial_condition_study,
    run_sweep_direction_study, run_all_methods
)
from src.validation import run_all_validations, run_unit_tests
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='2D Laplace Solver for Stream Function',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method sor --omega 1.5
  python main.py --run-all --tolerance 1e-6
  python main.py --omega-study --omega-step 0.05
  python main.py --ic-study --omega 1.8
  python main.py --sweep-study --method gauss_seidel
  python main.py --validate
  python main.py --test
        """
    )
    
    # Method selection
    parser.add_argument('--method', type=str, default='sor',
                        choices=list(SOLVER_METHODS.keys()),
                        help='Solver method (default: sor)')
    
    # Solver parameters
    parser.add_argument('--tolerance', type=float, default=1e-7,
                        help='Convergence tolerance (default: 1e-7)')
    parser.add_argument('--max-iters', type=int, default=100000,
                        help='Maximum iterations (default: 100000)')
    parser.add_argument('--omega', type=float, default=1.5,
                        help='Relaxation parameter for SOR methods (default: 1.5)')
    parser.add_argument('--sweep-direction', type=str, default='forward',
                        choices=['forward', 'reverse', 'alternating'],
                        help='Sweep direction for GS/Line methods (default: forward)')
    parser.add_argument('--line-direction', type=str, default='x',
                        choices=['x', 'y'],
                        help='Line direction for Line methods (default: x)')
    parser.add_argument('--convergence-metric', type=str, default='residual',
                        choices=['residual', 'update'],
                        help='Convergence metric type (default: residual)')
    
    # Initial condition
    parser.add_argument('--init-value', type=float, default=0.0,
                        help='Initial value for interior nodes (default: 0.0)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for plots (default: outputs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    # Study modes
    parser.add_argument('--run-all', action='store_true',
                        help='Run all solver methods and compare')
    parser.add_argument('--omega-study', action='store_true',
                        help='Run omega optimization study')
    parser.add_argument('--omega-min', type=float, default=1.0,
                        help='Minimum omega for study (default: 1.0)')
    parser.add_argument('--omega-max', type=float, default=1.98,
                        help='Maximum omega for study (default: 1.98)')
    parser.add_argument('--omega-step', type=float, default=0.02,
                        help='Omega step size for study (default: 0.02)')
    parser.add_argument('--ic-study', action='store_true',
                        help='Run initial condition sensitivity study')
    parser.add_argument('--sweep-study', action='store_true',
                        help='Run sweep direction comparison study')
    
    # Validation and testing
    parser.add_argument('--validate', action='store_true',
                        help='Run validation tests on results')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests')
    
    return parser.parse_args()


def run_single_method(args):
    """Run a single solver method."""
    print(f"\n{'='*60}")
    print(f"Running {args.method} solver")
    print(f"{'='*60}")
    print(f"  Tolerance: {args.tolerance}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Convergence metric: {args.convergence_metric}")
    
    # Create grid
    grid = Grid()
    grid.initialize_interior(value=args.init_value, mode='uniform')
    grid.apply_boundary_conditions()
    
    # Get solver and run
    start_time = time.time()
    
    if args.method == 'jacobi':
        grid, history = jacobi_solver(
            grid, tolerance=args.tolerance, max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric, verbose=args.verbose
        )
    elif args.method == 'gauss_seidel':
        grid, history = gauss_seidel_solver(
            grid, tolerance=args.tolerance, max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric,
            sweep_direction=args.sweep_direction, verbose=args.verbose
        )
    elif args.method == 'line_gauss_seidel':
        grid, history = line_gauss_seidel_solver(
            grid, tolerance=args.tolerance, max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric,
            sweep_direction=args.sweep_direction,
            line_direction=args.line_direction, verbose=args.verbose
        )
    elif args.method == 'sor':
        print(f"  Omega: {args.omega}")
        grid, history = sor_solver(
            grid, omega=args.omega, tolerance=args.tolerance,
            max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric, verbose=args.verbose
        )
    elif args.method == 'line_sor':
        print(f"  Omega: {args.omega}")
        grid, history = line_sor_solver(
            grid, omega=args.omega, tolerance=args.tolerance,
            max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric,
            sweep_direction=args.sweep_direction,
            line_direction=args.line_direction, verbose=args.verbose
        )
    elif args.method == 'adi':
        grid, history = adi_solver(
            grid, tolerance=args.tolerance, max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric, verbose=args.verbose
        )
    elif args.method == 'red_black_sor':
        print(f"  Omega: {args.omega}")
        grid, history = red_black_sor_solver(
            grid, omega=args.omega, tolerance=args.tolerance,
            max_iterations=args.max_iters,
            convergence_metric=args.convergence_metric, verbose=args.verbose
        )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    converged_str = "Yes" if history.converged else "No (max iterations reached)"
    print(f"  Converged: {converged_str}")
    print(f"  Iterations: {history.iterations}")
    print(f"  Final residual: {compute_residual_l2(grid.psi, grid.dx, grid.dy):.2e}")
    print(f"  Time: {elapsed_time:.2f} seconds")
    
    # Generate plots
    print(f"\nGenerating plots...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_streamlines(grid, output_dir=args.output_dir, method_name=args.method)
    
    return {args.method: (grid, history)}


def run_all_methods_mode(args):
    """Run all solver methods and generate comparison plots."""
    print(f"\n{'='*60}")
    print("Running ALL solver methods")
    print(f"{'='*60}")
    
    # Run all methods
    results = run_all_methods(
        tolerance=args.tolerance,
        max_iterations=args.max_iters,
        omega_sor=args.omega,
        verbose=args.verbose
    )
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_vertical_cut_comparison(results, x_cut=5.0, output_dir=args.output_dir)
    
    # Plot streamlines for one method (Point SOR)
    if 'Point SOR' in results:
        plot_streamlines(results['Point SOR'][0], output_dir=args.output_dir,
                        method_name='Point SOR')
    
    # Plot convergence comparison
    histories_dict = {name: hist for name, (grid, hist) in results.items()}
    plot_convergence_comparison(histories_dict, output_dir=args.output_dir)
    
    return results


def run_omega_study_mode(args):
    """Run omega optimization study for SOR methods."""
    print(f"\n{'='*60}")
    print("Running OMEGA optimization study")
    print(f"{'='*60}")
    
    omega_range = np.arange(args.omega_min, args.omega_max + args.omega_step/2, args.omega_step)
    
    # Point SOR study
    print("\n--- Point SOR ---")
    point_sor_results = run_omega_study(
        omega_range, solver_type='point_sor',
        tolerance=args.tolerance, max_iterations=args.max_iters,
        verbose=args.verbose
    )
    
    # Line SOR study
    print("\n--- Line SOR ---")
    line_sor_results = run_omega_study(
        omega_range, solver_type='line_sor',
        tolerance=args.tolerance, max_iterations=args.max_iters,
        line_direction=args.line_direction, verbose=args.verbose
    )
    
    # Generate plot
    print("\nGenerating omega study plot...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_omega_study(
        omega_range,
        point_sor_results['iterations'],
        line_sor_results['iterations'],
        output_dir=args.output_dir
    )
    
    return point_sor_results, line_sor_results


def run_ic_study_mode(args):
    """Run initial condition sensitivity study."""
    print(f"\n{'='*60}")
    print("Running INITIAL CONDITION sensitivity study")
    print(f"{'='*60}")
    
    # First find optimal omega if not specified
    omega_opt = args.omega
    if omega_opt == 1.5:  # Default value, let's find optimal
        print("\nFinding optimal omega first...")
        omega_range = np.arange(1.7, 1.95, 0.05)
        study_results = run_omega_study(
            omega_range, solver_type='point_sor',
            tolerance=args.tolerance, max_iterations=args.max_iters,
            verbose=False
        )
        omega_opt = study_results['omega_opt']
        print(f"Using optimal omega = {omega_opt:.3f}")
    
    # Run IC study
    ic_results = run_initial_condition_study(
        omega_opt=omega_opt,
        initial_values=[0.0, 2.5, 5.0, 10.0],
        tolerance=args.tolerance,
        max_iterations=args.max_iters,
        verbose=args.verbose
    )
    
    # Generate plot
    print("\nGenerating initial condition study plot...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    histories = {name: hist for name, (grid, hist) in ic_results.items()}
    plot_initial_condition_study(histories, output_dir=args.output_dir)
    
    return ic_results


def run_sweep_study_mode(args):
    """Run sweep direction comparison study."""
    print(f"\n{'='*60}")
    print("Running SWEEP DIRECTION comparison study")
    print(f"{'='*60}")
    
    # Gauss-Seidel sweep comparison
    print("\n--- Gauss-Seidel ---")
    gs_results = run_sweep_direction_study(
        method='gauss_seidel',
        tolerance=args.tolerance,
        max_iterations=args.max_iters,
        verbose=args.verbose
    )
    
    # Line SOR sweep comparison
    print("\n--- Line SOR ---")
    line_sor_results = run_sweep_direction_study(
        method='line_sor',
        tolerance=args.tolerance,
        max_iterations=args.max_iters,
        verbose=args.verbose
    )
    
    # Combine results
    all_results = {**gs_results, **line_sor_results}
    
    # Generate plot
    print("\nGenerating sweep direction comparison plot...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_sweep_direction_comparison(all_results, output_dir=args.output_dir)
    
    return all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("2D LAPLACE SOLVER FOR STREAM FUNCTION")
    print("="*60)
    print("\nDomain: [0, 10] x [0, 10]")
    print("Grid: 101 x 101 (dx = dy = 0.1)")
    print("Equation: d2(psi)/dx2 + d2(psi)/dy2 = 0")
    print("Boundary conditions: Dirichlet (inlet/outlet flow)")
    
    # Run unit tests if requested
    if args.test:
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    # Determine which mode to run
    results = None
    
    if args.omega_study:
        run_omega_study_mode(args)
    elif args.ic_study:
        run_ic_study_mode(args)
    elif args.sweep_study:
        run_sweep_study_mode(args)
    elif args.run_all:
        results = run_all_methods_mode(args)
    else:
        results = run_single_method(args)
    
    # Run validation if requested and we have results
    if args.validate and results is not None:
        run_all_validations(results, verbose=args.verbose)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
