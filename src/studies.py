"""
Parameter Studies Module for 2D Laplace Solver
==============================================

This module provides functions for conducting parameter studies:
1. Omega optimization for SOR methods
2. Initial condition sensitivity analysis
3. Sweep direction comparison
"""

import numpy as np
import copy
from .grid import Grid
from .solvers import sor_solver, line_sor_solver, gauss_seidel_solver
from .convergence import compare_solutions


def run_omega_study(omega_range, solver_type='point_sor', tolerance=1e-7,
                    max_iterations=100000, line_direction='x', verbose=False):
    """
    Run omega optimization study for SOR methods.
    
    For each omega value, solve the Laplace equation and record the number
    of iterations required to reach convergence.
    
    Parameters:
        omega_range (array-like): Array of omega values to test (e.g., [1.0, 1.02, ..., 1.98])
        solver_type (str): 'point_sor' or 'line_sor'
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations per solve
        line_direction (str): Line direction for Line SOR ('x' or 'y')
        verbose (bool): Print progress
        
    Returns:
        dict: Results containing:
            - 'omega_values': Array of omega values
            - 'iterations': Array of iterations to convergence
            - 'omega_opt': Optimal omega value
            - 'min_iterations': Minimum iterations achieved
    """
    omega_values = np.array(omega_range)
    iterations = np.zeros(len(omega_values), dtype=int)
    
    print(f"Running omega study for {solver_type}...")
    print(f"Testing {len(omega_values)} omega values from {omega_values[0]:.2f} to {omega_values[-1]:.2f}")
    
    for idx, omega in enumerate(omega_values):
        # Create fresh grid for each run
        grid = Grid()
        grid.initialize_interior(value=0.0, mode='uniform')
        grid.apply_boundary_conditions()
        
        # Select solver
        if solver_type == 'point_sor':
            _, history = sor_solver(grid, omega=omega, tolerance=tolerance,
                                   max_iterations=max_iterations,
                                   convergence_metric='residual', verbose=False)
        elif solver_type == 'line_sor':
            _, history = line_sor_solver(grid, omega=omega, tolerance=tolerance,
                                        max_iterations=max_iterations,
                                        line_direction=line_direction,
                                        convergence_metric='residual', verbose=False)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        iterations[idx] = history.iterations
        
        if verbose or idx % 10 == 0:
            converged_str = "converged" if history.converged else "max_iters"
            print(f"  omega = {omega:.3f}: {history.iterations:6d} iterations ({converged_str})")
    
    # Find optimal omega
    idx_opt = np.argmin(iterations)
    omega_opt = omega_values[idx_opt]
    min_iterations = iterations[idx_opt]
    
    print(f"\nOptimal omega for {solver_type}: {omega_opt:.3f} ({min_iterations} iterations)")
    
    return {
        'omega_values': omega_values,
        'iterations': iterations,
        'omega_opt': omega_opt,
        'min_iterations': min_iterations
    }


def run_initial_condition_study(omega_opt, initial_values=[0.0, 2.5, 5.0, 10.0],
                                tolerance=1e-7, max_iterations=100000, verbose=False):
    """
    Run initial condition sensitivity study for Point SOR.
    
    Tests different uniform initial values for interior nodes and compares
    convergence behavior.
    
    Parameters:
        omega_opt (float): Optimal omega value to use
        initial_values (list): List of initial values to test
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        verbose (bool): Print progress
        
    Returns:
        dict: Dictionary mapping IC description to (grid, history) tuples
    """
    results = {}
    
    print(f"\nRunning initial condition study with omega = {omega_opt:.3f}...")
    
    for init_val in initial_values:
        ic_name = f"psi_init = {init_val}"
        
        # Create grid with specified initial condition
        grid = Grid()
        grid.initialize_interior(value=init_val, mode='uniform')
        grid.apply_boundary_conditions()
        
        # Run Point SOR
        grid, history = sor_solver(grid, omega=omega_opt, tolerance=tolerance,
                                   max_iterations=max_iterations,
                                   convergence_metric='residual', verbose=False)
        
        results[ic_name] = (grid, history)
        
        if verbose:
            print(f"  {ic_name}: {history.iterations} iterations")
    
    # Also test non-uniform initial conditions
    print("  Testing non-uniform initial conditions...")
    
    # Linear x gradient
    grid = Grid()
    grid.initialize_interior(value=0.0, mode='linear_x')
    grid.apply_boundary_conditions()
    grid, history = sor_solver(grid, omega=omega_opt, tolerance=tolerance,
                               max_iterations=max_iterations,
                               convergence_metric='residual', verbose=False)
    results["Linear x gradient"] = (grid, history)
    
    # Linear y gradient
    grid = Grid()
    grid.initialize_interior(value=0.0, mode='linear_y')
    grid.apply_boundary_conditions()
    grid, history = sor_solver(grid, omega=omega_opt, tolerance=tolerance,
                               max_iterations=max_iterations,
                               convergence_metric='residual', verbose=False)
    results["Linear y gradient"] = (grid, history)
    
    print("\nInitial condition study results:")
    for ic_name, (grid, history) in results.items():
        print(f"  {ic_name}: {history.iterations} iterations")
    
    return results


def run_sweep_direction_study(method='gauss_seidel', tolerance=1e-7,
                              max_iterations=100000, verbose=False):
    """
    Compare convergence rates for different sweep directions.
    
    Tests forward, reverse, and alternating sweep directions for
    Gauss-Seidel and Line SOR methods.
    
    Parameters:
        method (str): 'gauss_seidel' or 'line_sor'
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        verbose (bool): Print progress
        
    Returns:
        dict: Dictionary mapping sweep direction to (grid, history) tuples
    """
    sweep_directions = ['forward', 'reverse', 'alternating']
    results = {}
    
    print(f"\nRunning sweep direction study for {method}...")
    
    for direction in sweep_directions:
        # Create fresh grid
        grid = Grid()
        grid.initialize_interior(value=0.0, mode='uniform')
        grid.apply_boundary_conditions()
        
        # Run solver with specified sweep direction
        if method == 'gauss_seidel':
            grid, history = gauss_seidel_solver(
                grid, tolerance=tolerance, max_iterations=max_iterations,
                sweep_direction=direction, convergence_metric='residual',
                verbose=False
            )
        elif method == 'line_sor':
            grid, history = line_sor_solver(
                grid, omega=1.5, tolerance=tolerance, max_iterations=max_iterations,
                sweep_direction=direction, line_direction='x',
                convergence_metric='residual', verbose=False
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results[f"{method} ({direction})"] = (grid, history)
        
        if verbose:
            print(f"  {direction}: {history.iterations} iterations")
    
    print("\nSweep direction study results:")
    for name, (grid, history) in results.items():
        print(f"  {name}: {history.iterations} iterations")
    
    return results


def run_all_methods(tolerance=1e-7, max_iterations=100000, omega_sor=1.5,
                    verbose=False):
    """
    Run all solver methods and return results for comparison.
    
    Parameters:
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        omega_sor (float): Omega value for SOR methods
        verbose (bool): Print progress
        
    Returns:
        dict: Dictionary mapping method names to (grid, history) tuples
    """
    from .solvers import (
        jacobi_solver, gauss_seidel_solver, line_gauss_seidel_solver,
        sor_solver, line_sor_solver, adi_solver, red_black_sor_solver
    )
    
    results = {}
    
    print(f"\nRunning all solver methods (tol={tolerance}, omega={omega_sor})...")
    
    # Define methods to run
    methods = [
        ('Jacobi', lambda g: jacobi_solver(g, tolerance=tolerance, 
                                           max_iterations=max_iterations,
                                           convergence_metric='residual', verbose=False)),
        ('Gauss-Seidel', lambda g: gauss_seidel_solver(g, tolerance=tolerance,
                                                       max_iterations=max_iterations,
                                                       convergence_metric='residual', verbose=False)),
        ('Line Gauss-Seidel', lambda g: line_gauss_seidel_solver(g, tolerance=tolerance,
                                                                  max_iterations=max_iterations,
                                                                  convergence_metric='residual', verbose=False)),
        ('Point SOR', lambda g: sor_solver(g, omega=omega_sor, tolerance=tolerance,
                                           max_iterations=max_iterations,
                                           convergence_metric='residual', verbose=False)),
        ('Line SOR', lambda g: line_sor_solver(g, omega=omega_sor, tolerance=tolerance,
                                               max_iterations=max_iterations,
                                               convergence_metric='residual', verbose=False)),
        ('ADI', lambda g: adi_solver(g, tolerance=tolerance, max_iterations=max_iterations,
                                     convergence_metric='residual', verbose=False)),
        ('Red-Black SOR', lambda g: red_black_sor_solver(g, omega=omega_sor, tolerance=tolerance,
                                                          max_iterations=max_iterations,
                                                          convergence_metric='residual', verbose=False)),
    ]
    
    for method_name, solver_func in methods:
        print(f"  Running {method_name}...", end=' ', flush=True)
        
        # Create fresh grid
        grid = Grid()
        grid.initialize_interior(value=0.0, mode='uniform')
        grid.apply_boundary_conditions()
        
        # Run solver
        grid, history = solver_func(grid)
        results[method_name] = (grid, history)
        
        converged_str = "converged" if history.converged else "max_iters"
        print(f"{history.iterations:6d} iterations ({converged_str})")
    
    # Verify solutions are similar
    print("\nVerifying solution consistency...")
    method_names = list(results.keys())
    ref_grid = results[method_names[0]][0]
    
    for method_name in method_names[1:]:
        grid = results[method_name][0]
        diff = compare_solutions(ref_grid.psi, grid.psi)
        print(f"  L2 diff ({method_names[0]} vs {method_name}): {diff:.2e}")
    
    return results
