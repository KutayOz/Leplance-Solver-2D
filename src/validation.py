"""
Validation Module for 2D Laplace Solver
=======================================

This module provides validation tests and assertions to verify the correctness
of the solver implementation.

Acceptance Checks:
------------------
1. Boundary nodes remain fixed during iterations
2. Residual decreases over iterations (at least trends down)
3. Different methods converge to similar solutions
"""

import numpy as np
from .grid import Grid
from .convergence import compute_residual_l2, compare_solutions


def validate_boundary_conditions(grid, tolerance=1e-10):
    """
    Verify that boundary conditions are correctly applied.
    
    Checks that all boundary nodes have the expected values as specified
    in the problem definition.
    
    Parameters:
        grid (Grid): Grid object to validate
        tolerance (float): Tolerance for floating point comparison
        
    Returns:
        tuple: (is_valid, error_message)
    """
    psi = grid.psi
    Nx, Ny = psi.shape
    errors = []
    
    # Check left wall (x = 0): psi = 0 for all y
    for j in range(Ny):
        if abs(psi[0, j]) > tolerance:
            errors.append(f"Left wall error at j={j}: psi[0,{j}] = {psi[0,j]:.6e} (expected 0)")
    
    # Check top wall (y = Ly): psi = 0 for all x
    for i in range(Nx):
        if abs(psi[i, Ny-1]) > tolerance:
            errors.append(f"Top wall error at i={i}: psi[{i},{Ny-1}] = {psi[i,Ny-1]:.6e} (expected 0)")
    
    # Check bottom boundary (y = 0)
    # For x in [0, 2.0], i in [0, 20]: psi = 0
    for i in range(0, grid.inlet_i_start + 1):
        if abs(psi[i, 0]) > tolerance:
            errors.append(f"Bottom left error at i={i}: psi[{i},0] = {psi[i,0]:.6e} (expected 0)")
    
    # Inlet opening: x in [2.0, 2.4], i in [20, 24]: linear from 0 to 10
    for i in range(grid.inlet_i_start, grid.inlet_i_end + 1):
        x_val = grid.index_to_x(i)
        expected = 10.0 * (x_val - 2.0) / 0.4
        if abs(psi[i, 0] - expected) > tolerance:
            errors.append(f"Inlet error at i={i}: psi[{i},0] = {psi[i,0]:.6e} (expected {expected:.6e})")
    
    # For x in [2.4, 10.0], i in [24, Nx-1]: psi = 10
    for i in range(grid.inlet_i_end, Nx):
        if abs(psi[i, 0] - 10.0) > tolerance:
            errors.append(f"Bottom right error at i={i}: psi[{i},0] = {psi[i,0]:.6e} (expected 10)")
    
    # Check right boundary (x = Lx)
    # For y in [0, 6.0], j in [0, 60]: psi = 10
    for j in range(0, grid.outlet_j_start + 1):
        if abs(psi[Nx-1, j] - 10.0) > tolerance:
            errors.append(f"Right bottom error at j={j}: psi[{Nx-1},{j}] = {psi[Nx-1,j]:.6e} (expected 10)")
    
    # Outlet opening: y in [6.0, 6.4], j in [60, 64]: linear from 10 to 0
    for j in range(grid.outlet_j_start, grid.outlet_j_end + 1):
        y_val = grid.index_to_y(j)
        expected = 10.0 * (6.4 - y_val) / 0.4
        if abs(psi[Nx-1, j] - expected) > tolerance:
            errors.append(f"Outlet error at j={j}: psi[{Nx-1},{j}] = {psi[Nx-1,j]:.6e} (expected {expected:.6e})")
    
    # For y in [6.4, 10.0], j in [64, Ny-1]: psi = 0
    for j in range(grid.outlet_j_end, Ny):
        if abs(psi[Nx-1, j]) > tolerance:
            errors.append(f"Right top error at j={j}: psi[{Nx-1},{j}] = {psi[Nx-1,j]:.6e} (expected 0)")
    
    if errors:
        return False, "\n".join(errors[:10])  # Limit to first 10 errors
    return True, "All boundary conditions verified"


def validate_residual_trend(history, min_decrease_ratio=0.5):
    """
    Verify that residual generally decreases over iterations.
    
    We don't require strict monotonic decrease, but the final residual
    should be significantly smaller than the initial residual.
    
    Parameters:
        history (ConvergenceHistory): Convergence history from solver
        min_decrease_ratio (float): Maximum allowed final/initial residual ratio
        
    Returns:
        tuple: (is_valid, message)
    """
    if len(history.residuals) < 2:
        return False, "Not enough residual data to validate trend"
    
    residuals = history.get_residuals()
    initial_residual = residuals[0]
    final_residual = residuals[-1]
    
    ratio = final_residual / initial_residual
    
    if ratio > min_decrease_ratio:
        return False, f"Residual did not decrease sufficiently: ratio = {ratio:.2e}"
    
    # Check for general downward trend (allow some local increases)
    # Use rolling average to smooth fluctuations
    window = min(100, len(residuals) // 10)
    if window > 1:
        smoothed = np.convolve(residuals, np.ones(window)/window, mode='valid')
        # Check if smoothed residual generally decreases
        n_segments = 5
        segment_len = len(smoothed) // n_segments
        if segment_len > 0:
            segment_means = [np.mean(smoothed[i*segment_len:(i+1)*segment_len]) 
                           for i in range(n_segments)]
            decreasing_count = sum(1 for i in range(n_segments-1) 
                                  if segment_means[i+1] < segment_means[i])
            if decreasing_count < n_segments - 2:
                return False, "Residual does not show consistent downward trend"
    
    return True, f"Residual decreased from {initial_residual:.2e} to {final_residual:.2e}"


def validate_solution_consistency(results_dict, tolerance=1e-3):
    """
    Verify that different methods converge to similar solutions.
    
    Parameters:
        results_dict (dict): Dictionary mapping method names to (grid, history) tuples
        tolerance (float): Maximum allowed L2 difference between solutions
        
    Returns:
        tuple: (is_valid, message, differences_dict)
    """
    method_names = list(results_dict.keys())
    
    if len(method_names) < 2:
        return True, "Only one method to compare", {}
    
    # Use first method as reference
    ref_name = method_names[0]
    ref_psi = results_dict[ref_name][0].psi
    
    differences = {}
    all_valid = True
    messages = []
    
    for name in method_names[1:]:
        psi = results_dict[name][0].psi
        diff = compare_solutions(ref_psi, psi)
        differences[f"{ref_name} vs {name}"] = diff
        
        if diff > tolerance:
            all_valid = False
            messages.append(f"Large difference: {ref_name} vs {name} = {diff:.2e}")
        else:
            messages.append(f"OK: {ref_name} vs {name} = {diff:.2e}")
    
    return all_valid, "\n".join(messages), differences


def run_all_validations(results_dict, verbose=True):
    """
    Run all validation tests on solver results.
    
    Parameters:
        results_dict (dict): Dictionary mapping method names to (grid, history) tuples
        verbose (bool): Print detailed results
        
    Returns:
        bool: True if all validations pass
    """
    all_passed = True
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    # 1. Boundary condition validation
    print("\n1. Boundary Condition Validation:")
    for method_name, (grid, history) in results_dict.items():
        is_valid, msg = validate_boundary_conditions(grid)
        status = "PASS" if is_valid else "FAIL"
        print(f"   [{status}] {method_name}")
        if not is_valid:
            all_passed = False
            if verbose:
                print(f"       {msg}")
    
    # 2. Residual trend validation
    print("\n2. Residual Trend Validation:")
    for method_name, (grid, history) in results_dict.items():
        is_valid, msg = validate_residual_trend(history)
        status = "PASS" if is_valid else "FAIL"
        print(f"   [{status}] {method_name}: {msg}")
        if not is_valid:
            all_passed = False
    
    # 3. Solution consistency validation
    print("\n3. Solution Consistency Validation:")
    is_valid, msg, diffs = validate_solution_consistency(results_dict)
    status = "PASS" if is_valid else "FAIL"
    print(f"   [{status}]")
    for line in msg.split("\n"):
        print(f"       {line}")
    if not is_valid:
        all_passed = False
    
    print("\n" + "="*60)
    overall_status = "ALL VALIDATIONS PASSED" if all_passed else "SOME VALIDATIONS FAILED"
    print(f"OVERALL: {overall_status}")
    print("="*60 + "\n")
    
    return all_passed


def test_grid_initialization():
    """Test grid initialization and coordinate mapping."""
    print("Testing grid initialization...")
    
    grid = Grid(Lx=10.0, Ly=10.0, dx=0.1, dy=0.1)
    
    # Check dimensions
    assert grid.Nx == 101, f"Expected Nx=101, got {grid.Nx}"
    assert grid.Ny == 101, f"Expected Ny=101, got {grid.Ny}"
    
    # Check coordinate arrays
    assert abs(grid.x[0]) < 1e-10, f"x[0] should be 0, got {grid.x[0]}"
    assert abs(grid.x[-1] - 10.0) < 1e-10, f"x[-1] should be 10, got {grid.x[-1]}"
    assert abs(grid.y[0]) < 1e-10, f"y[0] should be 0, got {grid.y[0]}"
    assert abs(grid.y[-1] - 10.0) < 1e-10, f"y[-1] should be 10, got {grid.y[-1]}"
    
    # Check index mappings
    assert grid.inlet_i_start == 20, f"inlet_i_start should be 20, got {grid.inlet_i_start}"
    assert grid.inlet_i_end == 24, f"inlet_i_end should be 24, got {grid.inlet_i_end}"
    assert grid.outlet_j_start == 60, f"outlet_j_start should be 60, got {grid.outlet_j_start}"
    assert grid.outlet_j_end == 64, f"outlet_j_end should be 64, got {grid.outlet_j_end}"
    
    # Check coordinate-index conversions
    assert grid.x_to_index(2.0) == 20, f"x=2.0 should map to i=20"
    assert grid.x_to_index(2.4) == 24, f"x=2.4 should map to i=24"
    assert grid.y_to_index(6.0) == 60, f"y=6.0 should map to j=60"
    assert grid.y_to_index(6.4) == 64, f"y=6.4 should map to j=64"
    
    print("  Grid initialization tests PASSED")
    return True


def test_boundary_conditions():
    """Test that boundary conditions are correctly applied."""
    print("Testing boundary conditions...")
    
    grid = Grid()
    grid.apply_boundary_conditions()
    
    is_valid, msg = validate_boundary_conditions(grid)
    assert is_valid, f"Boundary conditions failed: {msg}"
    
    print("  Boundary condition tests PASSED")
    return True


def test_thomas_algorithm():
    """Test the Thomas algorithm for tridiagonal systems."""
    print("Testing Thomas algorithm...")
    from .solvers import thomas_algorithm
    
    # Test case: simple tridiagonal system
    # -2*x[0] + x[1] = -1
    # x[0] - 2*x[1] + x[2] = 0
    # x[1] - 2*x[2] = -1
    # Solution: x = [1, 1, 1]
    
    n = 3
    a = np.array([0.0, 1.0, 1.0])
    b = np.array([-2.0, -2.0, -2.0])
    c = np.array([1.0, 1.0, 0.0])
    d = np.array([-1.0, 0.0, -1.0])
    
    x = thomas_algorithm(a, b, c, d)
    expected = np.array([1.0, 1.0, 1.0])
    
    assert np.allclose(x, expected, atol=1e-10), f"Thomas algorithm failed: got {x}, expected {expected}"
    
    # Test larger random system
    n = 50
    a = np.random.rand(n)
    a[0] = 0
    b = -3.0 * np.ones(n)  # Diagonally dominant
    c = np.random.rand(n)
    c[-1] = 0
    d = np.random.rand(n)
    
    x = thomas_algorithm(a, b, c, d)
    
    # Verify solution by computing A*x and comparing to d
    Ax = np.zeros(n)
    Ax[0] = b[0]*x[0] + c[0]*x[1]
    for i in range(1, n-1):
        Ax[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
    Ax[n-1] = a[n-1]*x[n-2] + b[n-1]*x[n-1]
    
    assert np.allclose(Ax, d, atol=1e-10), "Thomas algorithm failed for random system"
    
    print("  Thomas algorithm tests PASSED")
    return True


def run_unit_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("UNIT TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    try:
        test_grid_initialization()
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    try:
        test_boundary_conditions()
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    try:
        test_thomas_algorithm()
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    status = "ALL UNIT TESTS PASSED" if all_passed else "SOME UNIT TESTS FAILED"
    print(f"UNIT TEST RESULT: {status}")
    print("="*60 + "\n")
    
    return all_passed
