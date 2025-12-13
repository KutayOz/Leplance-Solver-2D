#!/usr/bin/env python3
"""
Run All Studies Script
======================

This script runs the complete analysis suite for the 2D Laplace solver:
1. Run all solver methods and compare solutions
2. Omega optimization study for SOR methods
3. Initial condition sensitivity study
4. Sweep direction comparison
5. Generate all required plots
6. Run validation tests

Usage:
    python scripts/run_all_studies.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.grid import Grid
from src.solvers import (
    jacobi_solver, gauss_seidel_solver, line_gauss_seidel_solver,
    sor_solver, line_sor_solver, adi_solver, red_black_sor_solver
)
from src.convergence import compute_residual_l2, compare_solutions
from src.plotting import (
    plot_vertical_cut_comparison, plot_streamlines,
    plot_omega_study, plot_initial_condition_study,
    plot_convergence_comparison, plot_sweep_direction_comparison,
    plot_solution_heatmap
)
from src.studies import (
    run_omega_study, run_initial_condition_study,
    run_sweep_direction_study, run_all_methods
)
from src.validation import run_all_validations, run_unit_tests


def main():
    """Run complete analysis suite."""
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    tolerance = 1e-7
    max_iterations = 100000
    
    print("\n" + "="*70)
    print("2D LAPLACE SOLVER - COMPLETE ANALYSIS SUITE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Tolerance: {tolerance}")
    print(f"Max iterations: {max_iterations}")
    
    total_start = time.time()
    
    # =========================================================================
    # 1. Run Unit Tests
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: UNIT TESTS")
    print("="*70)
    
    unit_tests_passed = run_unit_tests()
    if not unit_tests_passed:
        print("WARNING: Some unit tests failed. Continuing anyway...")
    
    # =========================================================================
    # 2. Run All Solver Methods
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: RUN ALL SOLVER METHODS")
    print("="*70)
    
    # Use a moderate omega for initial comparison
    omega_default = 1.5
    
    all_results = run_all_methods(
        tolerance=tolerance,
        max_iterations=max_iterations,
        omega_sor=omega_default,
        verbose=True
    )
    
    # Generate comparison plots
    print("\nGenerating method comparison plots...")
    plot_vertical_cut_comparison(all_results, x_cut=5.0, output_dir=output_dir)
    
    histories_dict = {name: hist for name, (grid, hist) in all_results.items()}
    plot_convergence_comparison(histories_dict, output_dir=output_dir)
    
    # Plot streamlines for Point SOR
    if 'Point SOR' in all_results:
        plot_streamlines(all_results['Point SOR'][0], output_dir=output_dir,
                        method_name='Point SOR')
        plot_solution_heatmap(all_results['Point SOR'][0], output_dir=output_dir,
                             method_name='Point SOR')
    
    # =========================================================================
    # 3. Run Validation Tests
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: VALIDATION TESTS")
    print("="*70)
    
    validation_passed = run_all_validations(all_results, verbose=True)
    
    # =========================================================================
    # 4. Omega Optimization Study
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: OMEGA OPTIMIZATION STUDY")
    print("="*70)
    
    omega_range = np.arange(1.0, 1.99, 0.05)
    
    print("\n--- Point SOR ---")
    point_sor_study = run_omega_study(
        omega_range, solver_type='point_sor',
        tolerance=tolerance, max_iterations=max_iterations,
        verbose=False
    )
    
    print("\n--- Line SOR ---")
    line_sor_study = run_omega_study(
        omega_range, solver_type='line_sor',
        tolerance=tolerance, max_iterations=max_iterations,
        verbose=False
    )
    
    # Generate omega study plot
    plot_omega_study(
        omega_range,
        point_sor_study['iterations'],
        line_sor_study['iterations'],
        output_dir=output_dir
    )
    
    print(f"\nOptimal omega (Point SOR): {point_sor_study['omega_opt']:.3f} "
          f"({point_sor_study['min_iterations']} iterations)")
    print(f"Optimal omega (Line SOR): {line_sor_study['omega_opt']:.3f} "
          f"({line_sor_study['min_iterations']} iterations)")
    
    # =========================================================================
    # 5. Initial Condition Sensitivity Study
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: INITIAL CONDITION SENSITIVITY STUDY")
    print("="*70)
    
    omega_opt = point_sor_study['omega_opt']
    
    ic_results = run_initial_condition_study(
        omega_opt=omega_opt,
        initial_values=[0.0, 2.5, 5.0, 10.0],
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=True
    )
    
    # Generate IC study plot
    ic_histories = {name: hist for name, (grid, hist) in ic_results.items()}
    plot_initial_condition_study(ic_histories, output_dir=output_dir)
    
    # =========================================================================
    # 6. Sweep Direction Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: SWEEP DIRECTION COMPARISON")
    print("="*70)
    
    print("\n--- Gauss-Seidel ---")
    gs_sweep_results = run_sweep_direction_study(
        method='gauss_seidel',
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=True
    )
    
    print("\n--- Line SOR ---")
    line_sor_sweep_results = run_sweep_direction_study(
        method='line_sor',
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=True
    )
    
    # Combine and plot
    all_sweep_results = {**gs_sweep_results, **line_sor_sweep_results}
    plot_sweep_direction_comparison(all_sweep_results, output_dir=output_dir)
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal runtime: {total_time:.1f} seconds")
    print(f"\nGenerated plots in {output_dir}:")
    
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith('.png'):
            print(f"  - {fname}")
    
    print("\nKey findings:")
    print(f"  - Optimal omega (Point SOR): {point_sor_study['omega_opt']:.3f}")
    print(f"  - Optimal omega (Line SOR): {line_sor_study['omega_opt']:.3f}")
    print(f"  - Unit tests: {'PASSED' if unit_tests_passed else 'FAILED'}")
    print(f"  - Validation: {'PASSED' if validation_passed else 'FAILED'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
