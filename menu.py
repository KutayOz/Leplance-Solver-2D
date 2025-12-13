#!/usr/bin/env python3
"""
2D Laplace Solver - Interactive Menu System
============================================

This script provides an interactive menu-driven interface for solving the 2D 
Laplace equation for the stream function psi(x,y) using various iterative methods.

Run with: python3 menu.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.grid import Grid
from src.solvers import (
    jacobi_solver, gauss_seidel_solver, line_gauss_seidel_solver,
    sor_solver, line_sor_solver, adi_solver, red_black_sor_solver,
    SOLVER_METHODS
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
from src.output_manager import OutputManager, get_output_manager
import numpy as np


# =============================================================================
# GLOBAL SETTINGS (can be modified through menu)
# =============================================================================
class Settings:
    """Global settings for the solver."""
    tolerance = 1e-7
    max_iterations = 100000
    omega = 1.5
    sweep_direction = 'forward'
    line_direction = 'x'
    convergence_metric = 'residual'
    init_value = 0.0
    output_dir = 'outputs'
    verbose = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the program header."""
    print("\n" + "=" * 60)
    print("     2D LAPLACE SOLVER FOR STREAM FUNCTION")
    print("=" * 60)
    print("  Domain: [0, 10] x [0, 10]")
    print("  Grid: 101 x 101 (dx = dy = 0.1)")
    print("  Equation: d2(psi)/dx2 + d2(psi)/dy2 = 0")
    print("=" * 60)


def print_current_settings():
    """Print current solver settings."""
    print("\n--- Current Settings ---")
    print(f"  Tolerance:          {Settings.tolerance}")
    print(f"  Max iterations:     {Settings.max_iterations}")
    print(f"  Omega (SOR):        {Settings.omega}")
    print(f"  Sweep direction:    {Settings.sweep_direction}")
    print(f"  Line direction:     {Settings.line_direction}")
    print(f"  Convergence metric: {Settings.convergence_metric}")
    print(f"  Initial value:      {Settings.init_value}")
    print(f"  Output directory:   {Settings.output_dir}")
    print(f"  Verbose:            {Settings.verbose}")
    print("-" * 26)


def get_user_choice(prompt, valid_choices):
    """Get a valid choice from the user."""
    while True:
        try:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            print(f"  Invalid choice. Please enter one of: {', '.join(valid_choices)}")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return None


def get_float_input(prompt, default=None, min_val=None, max_val=None):
    """Get a float input from the user."""
    while True:
        try:
            user_input = input(prompt).strip()
            if user_input == '' and default is not None:
                return default
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return default


def get_int_input(prompt, default=None, min_val=None, max_val=None):
    """Get an integer input from the user."""
    while True:
        try:
            user_input = input(prompt).strip()
            if user_input == '' and default is not None:
                return default
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("  Please enter a valid integer")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return default


def pause():
    """Pause and wait for user to press Enter."""
    input("\nPress Enter to continue...")


# =============================================================================
# MENU FUNCTIONS
# =============================================================================

def main_menu():
    """Display the main menu and get user choice."""
    print("\n" + "=" * 40)
    print("          MAIN MENU")
    print("=" * 40)
    print("  1. Run a single solver method")
    print("  2. Run all methods (compare)")
    print("  3. Omega optimization study")
    print("  4. Initial condition study")
    print("  5. Sweep direction study")
    print("  6. Run unit tests")
    print("  7. Change settings")
    print("  8. View current settings")
    print("  9. Help / About")
    print("  0. Exit")
    print("-" * 40)
    
    return get_user_choice("Enter your choice [0-9]: ", 
                          [str(i) for i in range(10)])


def solver_selection_menu():
    """Display solver selection menu."""
    print("\n" + "=" * 40)
    print("      SELECT SOLVER METHOD")
    print("=" * 40)
    print("  1. Point Jacobi")
    print("  2. Point Gauss-Seidel")
    print("  3. Line Gauss-Seidel (Thomas algorithm)")
    print("  4. Point SOR")
    print("  5. Line SOR")
    print("  6. ADI (Alternating Direction Implicit)")
    print("  7. Red-Black SOR")
    print("  0. Back to main menu")
    print("-" * 40)
    
    choice = get_user_choice("Enter your choice [0-7]: ",
                            [str(i) for i in range(8)])
    
    methods = {
        '1': 'jacobi',
        '2': 'gauss_seidel',
        '3': 'line_gauss_seidel',
        '4': 'sor',
        '5': 'line_sor',
        '6': 'adi',
        '7': 'red_black_sor',
        '0': None
    }
    
    return methods.get(choice)


def settings_menu():
    """Display and modify settings."""
    while True:
        print("\n" + "=" * 40)
        print("          SETTINGS")
        print("=" * 40)
        print(f"  1. Tolerance          [{Settings.tolerance}]")
        print(f"  2. Max iterations     [{Settings.max_iterations}]")
        print(f"  3. Omega (SOR)        [{Settings.omega}]")
        print(f"  4. Sweep direction    [{Settings.sweep_direction}]")
        print(f"  5. Line direction     [{Settings.line_direction}]")
        print(f"  6. Convergence metric [{Settings.convergence_metric}]")
        print(f"  7. Initial value      [{Settings.init_value}]")
        print(f"  8. Output directory   [{Settings.output_dir}]")
        print(f"  9. Verbose            [{Settings.verbose}]")
        print("  0. Back to main menu")
        print("-" * 40)
        
        choice = get_user_choice("Enter setting to change [0-9]: ",
                                [str(i) for i in range(10)])
        
        if choice == '0' or choice is None:
            break
        elif choice == '1':
            val = get_float_input(f"  Enter tolerance (current: {Settings.tolerance}): ",
                                 default=Settings.tolerance, min_val=1e-15, max_val=1.0)
            if val is not None:
                Settings.tolerance = val
        elif choice == '2':
            val = get_int_input(f"  Enter max iterations (current: {Settings.max_iterations}): ",
                               default=Settings.max_iterations, min_val=100, max_val=1000000)
            if val is not None:
                Settings.max_iterations = val
        elif choice == '3':
            val = get_float_input(f"  Enter omega (current: {Settings.omega}, range 1.0-1.99): ",
                                 default=Settings.omega, min_val=0.1, max_val=1.99)
            if val is not None:
                Settings.omega = val
        elif choice == '4':
            print("  Sweep directions: forward, reverse, alternating")
            val = get_user_choice("  Enter sweep direction: ", 
                                 ['forward', 'reverse', 'alternating'])
            if val is not None:
                Settings.sweep_direction = val
        elif choice == '5':
            val = get_user_choice("  Enter line direction (x/y): ", ['x', 'y'])
            if val is not None:
                Settings.line_direction = val
        elif choice == '6':
            val = get_user_choice("  Enter convergence metric (residual/update): ",
                                 ['residual', 'update'])
            if val is not None:
                Settings.convergence_metric = val
        elif choice == '7':
            val = get_float_input(f"  Enter initial value (current: {Settings.init_value}): ",
                                 default=Settings.init_value)
            if val is not None:
                Settings.init_value = val
        elif choice == '8':
            val = input(f"  Enter output directory (current: {Settings.output_dir}): ").strip()
            if val:
                Settings.output_dir = val
        elif choice == '9':
            Settings.verbose = not Settings.verbose
            print(f"  Verbose set to: {Settings.verbose}")


# =============================================================================
# SOLVER EXECUTION FUNCTIONS
# =============================================================================

def run_single_solver():
    """Run a single solver method."""
    method = solver_selection_menu()
    if method is None:
        return
    
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} solver")
    print(f"{'='*60}")
    print_current_settings()
    
    # Create grid
    grid = Grid()
    grid.initialize_interior(value=Settings.init_value, mode='uniform')
    grid.apply_boundary_conditions()
    
    # Create organized output directory
    output_mgr = get_output_manager(Settings.output_dir)
    run_dir = output_mgr.create_single_solver_dir(method)
    
    # Run solver
    start_time = time.time()
    
    if method == 'jacobi':
        grid, history = jacobi_solver(
            grid, tolerance=Settings.tolerance, 
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric, 
            verbose=Settings.verbose
        )
    elif method == 'gauss_seidel':
        grid, history = gauss_seidel_solver(
            grid, tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            sweep_direction=Settings.sweep_direction,
            verbose=Settings.verbose
        )
    elif method == 'line_gauss_seidel':
        grid, history = line_gauss_seidel_solver(
            grid, tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            sweep_direction=Settings.sweep_direction,
            line_direction=Settings.line_direction,
            verbose=Settings.verbose
        )
    elif method == 'sor':
        grid, history = sor_solver(
            grid, omega=Settings.omega,
            tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            verbose=Settings.verbose
        )
    elif method == 'line_sor':
        grid, history = line_sor_solver(
            grid, omega=Settings.omega,
            tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            sweep_direction=Settings.sweep_direction,
            line_direction=Settings.line_direction,
            verbose=Settings.verbose
        )
    elif method == 'adi':
        grid, history = adi_solver(
            grid, tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            verbose=Settings.verbose
        )
    elif method == 'red_black_sor':
        grid, history = red_black_sor_solver(
            grid, omega=Settings.omega,
            tolerance=Settings.tolerance,
            max_iterations=Settings.max_iterations,
            convergence_metric=Settings.convergence_metric,
            verbose=Settings.verbose
        )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    converged_str = "Yes" if history.converged else "No (max iterations reached)"
    print(f"  Converged:      {converged_str}")
    print(f"  Iterations:     {history.iterations}")
    print(f"  Final residual: {compute_residual_l2(grid.psi, grid.dx, grid.dy):.2e}")
    print(f"  Time:           {elapsed_time:.2f} seconds")
    
    # Save run info
    run_info = {
        'Method': method,
        'Tolerance': Settings.tolerance,
        'Max iterations': Settings.max_iterations,
        'Omega': Settings.omega if method in ['sor', 'line_sor', 'red_black_sor'] else 'N/A',
        'Converged': converged_str,
        'Iterations': history.iterations,
        'Final residual': f"{compute_residual_l2(grid.psi, grid.dx, grid.dy):.2e}",
        'Time (seconds)': f"{elapsed_time:.2f}"
    }
    output_mgr.save_run_info(run_dir, run_info)
    
    # Ask about generating plots
    print("\n--- Generate Plots ---")
    print(f"  Output folder: {run_dir}")
    choice = get_user_choice("  Generate streamlines plot? (y/n): ", ['y', 'n', 'Y', 'N'])
    if choice and choice.lower() == 'y':
        plot_streamlines(grid, output_dir=run_dir, method_name=method)
    
    choice = get_user_choice("  Generate heatmap plot? (y/n): ", ['y', 'n', 'Y', 'N'])
    if choice and choice.lower() == 'y':
        plot_solution_heatmap(grid, output_dir=run_dir, method_name=method)
    
    pause()


def run_all_methods_comparison():
    """Run all solver methods and compare."""
    print(f"\n{'='*60}")
    print("Running ALL solver methods for comparison")
    print(f"{'='*60}")
    print_current_settings()
    
    confirm = get_user_choice("\nThis may take several minutes. Continue? (y/n): ",
                             ['y', 'n', 'Y', 'N'])
    if confirm is None or confirm.lower() != 'y':
        return
    
    # Create organized output directory
    output_mgr = get_output_manager(Settings.output_dir)
    run_dir = output_mgr.create_all_methods_dir()
    print(f"\nOutput folder: {run_dir}")
    
    # Run all methods
    start_time = time.time()
    results = run_all_methods(
        tolerance=Settings.tolerance,
        max_iterations=Settings.max_iterations,
        omega_sor=Settings.omega,
        verbose=Settings.verbose
    )
    elapsed_time = time.time() - start_time
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_vertical_cut_comparison(results, x_cut=5.0, output_dir=run_dir)
    
    histories_dict = {name: hist for name, (grid, hist) in results.items()}
    plot_convergence_comparison(histories_dict, output_dir=run_dir)
    
    if 'Point SOR' in results:
        plot_streamlines(results['Point SOR'][0], output_dir=run_dir,
                        method_name='Point SOR')
    
    # Save run info
    run_info = {
        'Operation': 'All Methods Comparison',
        'Tolerance': Settings.tolerance,
        'Max iterations': Settings.max_iterations,
        'Omega (SOR)': Settings.omega,
        'Total time (seconds)': f"{elapsed_time:.2f}",
        'Methods run': ', '.join(results.keys())
    }
    for name, (grid, hist) in results.items():
        run_info[f'{name} iterations'] = hist.iterations
    output_mgr.save_run_info(run_dir, run_info)
    
    # Run validation
    print("\n--- Validation ---")
    choice = get_user_choice("  Run validation tests? (y/n): ", ['y', 'n', 'Y', 'N'])
    if choice and choice.lower() == 'y':
        run_all_validations(results, verbose=True)
    
    pause()


def run_omega_optimization():
    """Run omega optimization study for SOR methods."""
    print(f"\n{'='*60}")
    print("OMEGA OPTIMIZATION STUDY")
    print(f"{'='*60}")
    
    # Get omega range from user
    omega_min = get_float_input("  Enter minimum omega [1.0]: ", default=1.0, 
                                min_val=0.1, max_val=1.9)
    omega_max = get_float_input("  Enter maximum omega [1.98]: ", default=1.98,
                                min_val=omega_min + 0.01, max_val=1.99)
    omega_step = get_float_input("  Enter omega step [0.05]: ", default=0.05,
                                 min_val=0.01, max_val=0.2)
    
    if omega_min is None or omega_max is None or omega_step is None:
        return
    
    omega_range = np.arange(omega_min, omega_max + omega_step/2, omega_step)
    
    print(f"\nTesting {len(omega_range)} omega values from {omega_min:.2f} to {omega_max:.2f}")
    
    confirm = get_user_choice("Continue? (y/n): ", ['y', 'n', 'Y', 'N'])
    if confirm is None or confirm.lower() != 'y':
        return
    
    # Create organized output directory
    output_mgr = get_output_manager(Settings.output_dir)
    run_dir = output_mgr.create_omega_study_dir()
    print(f"\nOutput folder: {run_dir}")
    
    # Point SOR study
    print("\n--- Point SOR ---")
    start_time = time.time()
    point_sor_results = run_omega_study(
        omega_range, solver_type='point_sor',
        tolerance=Settings.tolerance, max_iterations=Settings.max_iterations,
        verbose=False
    )
    
    # Line SOR study
    print("\n--- Line SOR ---")
    line_sor_results = run_omega_study(
        omega_range, solver_type='line_sor',
        tolerance=Settings.tolerance, max_iterations=Settings.max_iterations,
        line_direction=Settings.line_direction, verbose=False
    )
    elapsed_time = time.time() - start_time
    
    # Generate plot
    print("\nGenerating omega study plot...")
    plot_omega_study(
        omega_range,
        point_sor_results['iterations'],
        line_sor_results['iterations'],
        output_dir=run_dir
    )
    
    # Save run info
    run_info = {
        'Operation': 'Omega Optimization Study',
        'Omega range': f"{omega_min:.2f} to {omega_max:.2f} (step {omega_step:.2f})",
        'Tolerance': Settings.tolerance,
        'Max iterations': Settings.max_iterations,
        'Total time (seconds)': f"{elapsed_time:.2f}",
        'Point SOR optimal omega': f"{point_sor_results['omega_opt']:.3f}",
        'Point SOR min iterations': point_sor_results['min_iterations'],
        'Line SOR optimal omega': f"{line_sor_results['omega_opt']:.3f}",
        'Line SOR min iterations': line_sor_results['min_iterations']
    }
    output_mgr.save_run_info(run_dir, run_info)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Point SOR optimal omega: {point_sor_results['omega_opt']:.3f} "
          f"({point_sor_results['min_iterations']} iterations)")
    print(f"  Line SOR optimal omega:  {line_sor_results['omega_opt']:.3f} "
          f"({line_sor_results['min_iterations']} iterations)")
    
    # Offer to update settings
    choice = get_user_choice("\n  Update omega setting to Point SOR optimal? (y/n): ",
                            ['y', 'n', 'Y', 'N'])
    if choice and choice.lower() == 'y':
        Settings.omega = point_sor_results['omega_opt']
        print(f"  Omega updated to {Settings.omega}")
    
    pause()


def run_ic_sensitivity():
    """Run initial condition sensitivity study."""
    print(f"\n{'='*60}")
    print("INITIAL CONDITION SENSITIVITY STUDY")
    print(f"{'='*60}")
    print(f"  Using omega = {Settings.omega}")
    print("  Testing initial values: 0.0, 2.5, 5.0, 10.0")
    print("  Plus linear gradients in x and y")
    
    confirm = get_user_choice("\nContinue? (y/n): ", ['y', 'n', 'Y', 'N'])
    if confirm is None or confirm.lower() != 'y':
        return
    
    # Create organized output directory
    output_mgr = get_output_manager(Settings.output_dir)
    run_dir = output_mgr.create_ic_study_dir()
    print(f"\nOutput folder: {run_dir}")
    
    # Run study
    start_time = time.time()
    ic_results = run_initial_condition_study(
        omega_opt=Settings.omega,
        initial_values=[0.0, 2.5, 5.0, 10.0],
        tolerance=Settings.tolerance,
        max_iterations=Settings.max_iterations,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    
    # Generate plot
    print("\nGenerating initial condition study plot...")
    ic_histories = {name: hist for name, (grid, hist) in ic_results.items()}
    plot_initial_condition_study(ic_histories, output_dir=run_dir)
    
    # Save run info
    run_info = {
        'Operation': 'Initial Condition Sensitivity Study',
        'Omega': Settings.omega,
        'Tolerance': Settings.tolerance,
        'Max iterations': Settings.max_iterations,
        'Total time (seconds)': f"{elapsed_time:.2f}"
    }
    for name, (grid, hist) in ic_results.items():
        run_info[f'{name} iterations'] = hist.iterations
    output_mgr.save_run_info(run_dir, run_info)
    
    pause()


def run_sweep_comparison():
    """Run sweep direction comparison study."""
    print(f"\n{'='*60}")
    print("SWEEP DIRECTION COMPARISON STUDY")
    print(f"{'='*60}")
    print("  Comparing: forward, reverse, alternating")
    print("  Methods: Gauss-Seidel and Line SOR")
    
    confirm = get_user_choice("\nContinue? (y/n): ", ['y', 'n', 'Y', 'N'])
    if confirm is None or confirm.lower() != 'y':
        return
    
    # Create organized output directory
    output_mgr = get_output_manager(Settings.output_dir)
    run_dir = output_mgr.create_sweep_study_dir()
    print(f"\nOutput folder: {run_dir}")
    
    # Run studies
    start_time = time.time()
    print("\n--- Gauss-Seidel ---")
    gs_results = run_sweep_direction_study(
        method='gauss_seidel',
        tolerance=Settings.tolerance,
        max_iterations=Settings.max_iterations,
        verbose=True
    )
    
    print("\n--- Line SOR ---")
    line_sor_results = run_sweep_direction_study(
        method='line_sor',
        tolerance=Settings.tolerance,
        max_iterations=Settings.max_iterations,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    
    # Generate plot
    all_results = {**gs_results, **line_sor_results}
    print("\nGenerating sweep direction comparison plot...")
    plot_sweep_direction_comparison(all_results, output_dir=run_dir)
    
    # Save run info
    run_info = {
        'Operation': 'Sweep Direction Comparison Study',
        'Tolerance': Settings.tolerance,
        'Max iterations': Settings.max_iterations,
        'Total time (seconds)': f"{elapsed_time:.2f}"
    }
    for name, (grid, hist) in all_results.items():
        run_info[f'{name} iterations'] = hist.iterations
    output_mgr.save_run_info(run_dir, run_info)
    
    pause()


def run_tests():
    """Run unit tests."""
    print(f"\n{'='*60}")
    print("RUNNING UNIT TESTS")
    print(f"{'='*60}")
    
    success = run_unit_tests()
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check output above.")
    
    pause()


def show_help():
    """Display help information."""
    clear_screen()
    print("""
================================================================================
                    2D LAPLACE SOLVER - HELP
================================================================================

PHYSICS
-------
This solver computes the stream function psi(x,y) for 2D incompressible,
irrotational flow by solving the Laplace equation:

    d2(psi)/dx2 + d2(psi)/dy2 = 0

Stream function definition:
    u = d(psi)/dy  (velocity in x-direction)
    v = -d(psi)/dx (velocity in y-direction)

DOMAIN
------
    Physical domain: [0, 10] x [0, 10]
    Grid: 101 x 101 points (dx = dy = 0.1)
    
BOUNDARY CONDITIONS
-------------------
    Left wall (x=0):     psi = 0
    Top wall (y=10):     psi = 0
    Bottom inlet:        Linear transition from 0 to 10 at x in [2.0, 2.4]
    Right outlet:        Linear transition from 10 to 0 at y in [6.0, 6.4]

SOLVER METHODS
--------------
    1. Jacobi           - Slowest, fully parallel updates
    2. Gauss-Seidel     - Faster, sequential updates
    3. Line Gauss-Seidel- Uses Thomas algorithm for line solves
    4. Point SOR        - Over-relaxation for faster convergence
    5. Line SOR         - Line solve with relaxation
    6. ADI              - Alternating Direction Implicit
    7. Red-Black SOR    - Checkerboard pattern, parallelizable

PARAMETERS
----------
    Tolerance:  Convergence criterion (default 1e-7)
    Omega:      Relaxation parameter for SOR (optimal ~1.8-1.95)
    
OUTPUT
------
    All plots are saved to the outputs/ directory.
    
================================================================================
""")
    pause()


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    """Main program entry point."""
    clear_screen()
    print_header()
    
    while True:
        choice = main_menu()
        
        if choice is None or choice == '0':
            print("\nGoodbye!")
            break
        elif choice == '1':
            run_single_solver()
        elif choice == '2':
            run_all_methods_comparison()
        elif choice == '3':
            run_omega_optimization()
        elif choice == '4':
            run_ic_sensitivity()
        elif choice == '5':
            run_sweep_comparison()
        elif choice == '6':
            run_tests()
        elif choice == '7':
            settings_menu()
        elif choice == '8':
            print_current_settings()
            pause()
        elif choice == '9':
            show_help()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
        sys.exit(0)
