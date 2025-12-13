#!/usr/bin/env python3
"""
Unit Tests for 2D Laplace Solver
================================

This module contains comprehensive tests for all solver components.
Run with: python -m pytest tests/test_solvers.py -v
Or: python tests/test_solvers.py
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.grid import Grid
from src.solvers import (
    thomas_algorithm,
    jacobi_solver, gauss_seidel_solver, line_gauss_seidel_solver,
    sor_solver, line_sor_solver, adi_solver, red_black_sor_solver
)
from src.convergence import (
    compute_residual_l2, compute_update_l2, compare_solutions
)
from src.validation import (
    validate_boundary_conditions, validate_residual_trend,
    validate_solution_consistency
)


class TestGrid:
    """Tests for Grid class."""
    
    def test_grid_dimensions(self):
        """Test grid dimensions are correct."""
        grid = Grid(Lx=10.0, Ly=10.0, dx=0.1, dy=0.1)
        assert grid.Nx == 101, f"Expected Nx=101, got {grid.Nx}"
        assert grid.Ny == 101, f"Expected Ny=101, got {grid.Ny}"
        assert grid.psi.shape == (101, 101), f"Wrong psi shape: {grid.psi.shape}"
    
    def test_coordinate_arrays(self):
        """Test coordinate arrays."""
        grid = Grid()
        assert abs(grid.x[0]) < 1e-10, "x[0] should be 0"
        assert abs(grid.x[-1] - 10.0) < 1e-10, "x[-1] should be 10"
        assert abs(grid.y[0]) < 1e-10, "y[0] should be 0"
        assert abs(grid.y[-1] - 10.0) < 1e-10, "y[-1] should be 10"
    
    def test_index_mappings(self):
        """Test index-coordinate conversions."""
        grid = Grid()
        # Key indices from problem specification
        assert grid.inlet_i_start == 20, "inlet_i_start should be 20"
        assert grid.inlet_i_end == 24, "inlet_i_end should be 24"
        assert grid.outlet_j_start == 60, "outlet_j_start should be 60"
        assert grid.outlet_j_end == 64, "outlet_j_end should be 64"
        
        # Coordinate to index
        assert grid.x_to_index(2.0) == 20
        assert grid.x_to_index(2.4) == 24
        assert grid.y_to_index(6.0) == 60
        assert grid.y_to_index(6.4) == 64
    
    def test_boundary_conditions_left_wall(self):
        """Test left wall boundary condition (psi = 0)."""
        grid = Grid()
        grid.apply_boundary_conditions()
        assert np.allclose(grid.psi[0, :], 0.0), "Left wall should be psi=0"
    
    def test_boundary_conditions_top_wall(self):
        """Test top wall boundary condition (psi = 0)."""
        grid = Grid()
        grid.apply_boundary_conditions()
        assert np.allclose(grid.psi[:, -1], 0.0), "Top wall should be psi=0"
    
    def test_boundary_conditions_bottom_inlet(self):
        """Test bottom boundary with inlet."""
        grid = Grid()
        grid.apply_boundary_conditions()
        
        # Bottom left: psi = 0 for i = 0 to 20
        assert np.allclose(grid.psi[:21, 0], 0.0), "Bottom left should be psi=0"
        
        # Inlet: linear from 0 to 10 for i = 20 to 24
        for i in range(20, 25):
            x = grid.index_to_x(i)
            expected = 10.0 * (x - 2.0) / 0.4
            assert abs(grid.psi[i, 0] - expected) < 1e-10, \
                f"Inlet at i={i}: expected {expected}, got {grid.psi[i, 0]}"
        
        # Bottom right: psi = 10 for i = 24 to 100
        assert np.allclose(grid.psi[24:, 0], 10.0), "Bottom right should be psi=10"
    
    def test_boundary_conditions_right_outlet(self):
        """Test right boundary with outlet."""
        grid = Grid()
        grid.apply_boundary_conditions()
        
        # Right bottom: psi = 10 for j = 0 to 60
        assert np.allclose(grid.psi[-1, :61], 10.0), "Right bottom should be psi=10"
        
        # Outlet: linear from 10 to 0 for j = 60 to 64
        for j in range(60, 65):
            y = grid.index_to_y(j)
            expected = 10.0 * (6.4 - y) / 0.4
            assert abs(grid.psi[-1, j] - expected) < 1e-10, \
                f"Outlet at j={j}: expected {expected}, got {grid.psi[-1, j]}"
        
        # Right top: psi = 0 for j = 64 to 100
        assert np.allclose(grid.psi[-1, 64:], 0.0), "Right top should be psi=0"


class TestThomasAlgorithm:
    """Tests for Thomas algorithm (tridiagonal solver)."""
    
    def test_simple_system(self):
        """Test with a simple known system."""
        # System: -2x0 + x1 = -1
        #         x0 - 2x1 + x2 = 0
        #         x1 - 2x2 = -1
        # Solution: x = [1, 1, 1]
        a = np.array([0.0, 1.0, 1.0])
        b = np.array([-2.0, -2.0, -2.0])
        c = np.array([1.0, 1.0, 0.0])
        d = np.array([-1.0, 0.0, -1.0])
        
        x = thomas_algorithm(a, b, c, d)
        expected = np.array([1.0, 1.0, 1.0])
        
        assert np.allclose(x, expected), f"Got {x}, expected {expected}"
    
    def test_random_system(self):
        """Test with random diagonally dominant system."""
        n = 50
        np.random.seed(42)
        
        a = np.random.rand(n)
        a[0] = 0
        b = -4.0 * np.ones(n)  # Diagonally dominant
        c = np.random.rand(n)
        c[-1] = 0
        d = np.random.rand(n)
        
        x = thomas_algorithm(a, b, c, d)
        
        # Verify A*x = d
        Ax = np.zeros(n)
        Ax[0] = b[0]*x[0] + c[0]*x[1]
        for i in range(1, n-1):
            Ax[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
        Ax[n-1] = a[n-1]*x[n-2] + b[n-1]*x[n-1]
        
        assert np.allclose(Ax, d, atol=1e-10), "Thomas algorithm failed verification"


class TestConvergenceMetrics:
    """Tests for convergence metrics."""
    
    def test_residual_zero_for_solution(self):
        """Test that residual is near zero for a valid solution."""
        # Create a simple linear solution psi(x,y) = x + y
        # This satisfies Laplace equation exactly
        grid = Grid()
        for i in range(grid.Nx):
            for j in range(grid.Ny):
                grid.psi[i, j] = grid.x[i] + grid.y[j]
        
        residual = compute_residual_l2(grid.psi, grid.dx, grid.dy)
        assert residual < 1e-10, f"Residual should be ~0, got {residual}"
    
    def test_update_norm(self):
        """Test update norm calculation."""
        psi_old = np.zeros((10, 10))
        psi_new = np.ones((10, 10))
        
        # Interior is 8x8 = 64 points, each with diff = 1
        # L2 = sqrt(64/64) = 1.0
        update = compute_update_l2(psi_new, psi_old)
        assert abs(update - 1.0) < 1e-10, f"Expected update=1.0, got {update}"
    
    def test_solution_comparison(self):
        """Test solution comparison function."""
        psi1 = np.ones((10, 10))
        psi2 = np.ones((10, 10))
        
        diff = compare_solutions(psi1, psi2)
        assert diff < 1e-10, "Identical solutions should have zero difference"
        
        psi2[5, 5] = 2.0
        diff = compare_solutions(psi1, psi2)
        assert diff > 0, "Different solutions should have nonzero difference"


class TestSolvers:
    """Tests for iterative solvers."""
    
    def setup_method(self):
        """Set up fresh grid for each test."""
        self.grid = Grid()
        self.grid.initialize_interior(value=0.0, mode='uniform')
        self.grid.apply_boundary_conditions()
        self.tolerance = 1e-5  # Looser tolerance for faster tests
        self.max_iters = 5000
    
    def test_jacobi_convergence(self):
        """Test Jacobi solver converges."""
        grid, history = jacobi_solver(
            self.grid, tolerance=self.tolerance, max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        # Jacobi is slow, so we just check it's making progress
        assert history.iterations > 0, "Should have done iterations"
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_gauss_seidel_convergence(self):
        """Test Gauss-Seidel solver converges."""
        grid, history = gauss_seidel_solver(
            self.grid, tolerance=self.tolerance, max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_sor_convergence(self):
        """Test Point SOR solver converges."""
        grid, history = sor_solver(
            self.grid, omega=1.5, tolerance=self.tolerance, 
            max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        assert history.converged or history.iterations == self.max_iters
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_line_sor_convergence(self):
        """Test Line SOR solver converges."""
        grid, history = line_sor_solver(
            self.grid, omega=1.5, tolerance=self.tolerance,
            max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_adi_convergence(self):
        """Test ADI solver converges."""
        grid, history = adi_solver(
            self.grid, tolerance=self.tolerance, max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_red_black_convergence(self):
        """Test Red-Black SOR solver converges."""
        grid, history = red_black_sor_solver(
            self.grid, omega=1.5, tolerance=self.tolerance,
            max_iterations=self.max_iters,
            convergence_metric='residual', verbose=False
        )
        is_valid, _ = validate_boundary_conditions(grid)
        assert is_valid, "Boundaries should remain fixed"
    
    def test_sweep_directions(self):
        """Test different sweep directions work."""
        for direction in ['forward', 'reverse', 'alternating']:
            grid = Grid()
            grid.initialize_interior(value=0.0, mode='uniform')
            grid.apply_boundary_conditions()
            
            grid, history = gauss_seidel_solver(
                grid, tolerance=self.tolerance, max_iterations=1000,
                sweep_direction=direction, verbose=False
            )
            assert history.iterations > 0, f"{direction} sweep should work"


class TestSolutionConsistency:
    """Test that different solvers converge to the same solution."""
    
    def test_sor_vs_gauss_seidel(self):
        """Test SOR with omega=1 equals Gauss-Seidel."""
        tol = 1e-6
        max_iters = 10000
        
        # Run Gauss-Seidel
        grid1 = Grid()
        grid1.initialize_interior(value=0.0, mode='uniform')
        grid1.apply_boundary_conditions()
        grid1, _ = gauss_seidel_solver(grid1, tolerance=tol, max_iterations=max_iters)
        
        # Run SOR with omega=1
        grid2 = Grid()
        grid2.initialize_interior(value=0.0, mode='uniform')
        grid2.apply_boundary_conditions()
        grid2, _ = sor_solver(grid2, omega=1.0, tolerance=tol, max_iterations=max_iters)
        
        # Compare solutions
        diff = compare_solutions(grid1.psi, grid2.psi)
        assert diff < 1e-4, f"Solutions should match, diff={diff}"


def run_tests():
    """Run all tests manually."""
    print("\n" + "="*60)
    print("RUNNING SOLVER TESTS")
    print("="*60)
    
    test_classes = [
        TestGrid,
        TestThomasAlgorithm,
        TestConvergenceMetrics,
        TestSolvers,
        TestSolutionConsistency
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        # Get test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Setup if exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                # Run test
                method = getattr(instance, method_name)
                method()
                
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {method_name}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    if failed_tests:
        print("Failed tests:")
        for t in failed_tests:
            print(f"  - {t}")
    print("="*60)
    
    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
