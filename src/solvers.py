"""
Iterative Solvers for 2D Laplace Equation
==========================================

This module implements various iterative methods for solving the 2D Laplace equation:
    d2(psi)/dx2 + d2(psi)/dy2 = 0

with Dirichlet boundary conditions.

Discretization (5-point Laplacian):
-----------------------------------
Using the standard 5-point finite difference stencil at interior nodes:

    (psi[i+1,j] - 2*psi[i,j] + psi[i-1,j]) / dx^2
  + (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1]) / dy^2 = 0

For dx = dy = h, the update formula becomes:
    psi[i,j] = 0.25 * (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1])

Implemented Solvers:
--------------------
a) Point Jacobi
b) Point Gauss-Seidel (with sweep direction options)
c) Line Gauss-Seidel (Thomas algorithm for tridiagonal systems)
d) Point SOR (Successive Over-Relaxation)
e) Line SOR (line solve + relaxation)
f) ADI (Alternating Direction Implicit)
g) Red-Black SOR (checkerboard update pattern)

Sweep Directions:
-----------------
For Point Gauss-Seidel and Line methods, three sweep directions are supported:
- 'forward': Sweep from low to high indices
- 'reverse': Sweep from high to low indices  
- 'alternating': Alternate between forward and reverse each iteration
"""

import numpy as np
from .convergence import (
    compute_residual_l2, 
    compute_update_l2,
    check_convergence,
    ConvergenceHistory
)


# =============================================================================
# THOMAS ALGORITHM (Tridiagonal Solver)
# =============================================================================

def thomas_algorithm(a, b, c, d):
    """
    Solve a tridiagonal system using the Thomas algorithm.
    
    The system is:
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    
    where:
        - a[0] is not used (no x[-1] term for first equation)
        - c[n-1] is not used (no x[n] term for last equation)
    
    Parameters:
        a (ndarray): Sub-diagonal coefficients (length n)
        b (ndarray): Main diagonal coefficients (length n)
        c (ndarray): Super-diagonal coefficients (length n)
        d (ndarray): Right-hand side vector (length n)
        
    Returns:
        x (ndarray): Solution vector (length n)
        
    Notes:
        This is an in-place algorithm that modifies c and d.
        For repeated solves with same matrix, use a different approach.
    """
    n = len(d)
    
    # Make copies to avoid modifying input
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    # Forward elimination
    # First row: c'[0] = c[0]/b[0], d'[0] = d[0]/b[0]
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        # Compute multiplier
        denom = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
    
    # Back substitution
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


# =============================================================================
# POINT JACOBI SOLVER
# =============================================================================

def jacobi_solver(grid, tolerance=1e-7, max_iterations=100000,
                  convergence_metric='residual', verbose=False):
    """
    Solve the Laplace equation using Point Jacobi iteration.
    
    In Jacobi iteration, all updates use values from the previous iteration:
        psi_new[i,j] = 0.25 * (psi_old[i+1,j] + psi_old[i-1,j] 
                              + psi_old[i,j+1] + psi_old[i,j-1])
    
    Parameters:
        grid (Grid): Grid object with boundary conditions applied
        tolerance (float): Convergence tolerance (default 1e-7)
        max_iterations (int): Maximum iterations (default 100000)
        convergence_metric (str): 'residual' or 'update' for L2 norm type
        verbose (bool): Print progress every 1000 iterations
        
    Returns:
        grid (Grid): Grid object with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    
    # For dx = dy, coefficient is 0.25
    # For dx != dy, need to account for different weights
    dx2 = dx * dx
    dy2 = dy * dy
    coeff_x = 1.0 / (2.0 * (dx2 + dy2) / dx2)  # Simplifies to 0.25 when dx=dy
    coeff_y = 1.0 / (2.0 * (dx2 + dy2) / dy2)
    
    # When dx = dy: coeff = 0.25
    if abs(dx - dy) < 1e-12:
        coeff = 0.25
    else:
        # General case: psi[i,j] = (dy2*(psi[i+1,j]+psi[i-1,j]) + dx2*(psi[i,j+1]+psi[i,j-1])) / (2*(dx2+dy2))
        pass
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # Update all interior points using old values (Jacobi)
        # Interior: i in [1, Nx-2], j in [1, Ny-2]
        if abs(dx - dy) < 1e-12:
            # Simplified update for dx = dy
            psi[1:-1, 1:-1] = 0.25 * (
                psi_old[2:, 1:-1] + psi_old[:-2, 1:-1] +  # neighbors in x
                psi_old[1:-1, 2:] + psi_old[1:-1, :-2]    # neighbors in y
            )
        else:
            # General update for dx != dy
            psi[1:-1, 1:-1] = (
                dy2 * (psi_old[2:, 1:-1] + psi_old[:-2, 1:-1]) +
                dx2 * (psi_old[1:-1, 2:] + psi_old[1:-1, :-2])
            ) / (2.0 * (dx2 + dy2))
        
        # Re-apply boundary conditions (they should remain fixed)
        grid.apply_boundary_conditions()
        
        # Compute convergence metric
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"Jacobi iteration {iteration}: {convergence_metric} = {metric:.2e}")
        
        # Check convergence
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Jacobi converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# POINT GAUSS-SEIDEL SOLVER
# =============================================================================

def gauss_seidel_solver(grid, tolerance=1e-7, max_iterations=100000,
                        convergence_metric='residual', sweep_direction='forward',
                        verbose=False):
    """
    Solve the Laplace equation using Point Gauss-Seidel iteration.
    
    In Gauss-Seidel, updates immediately use newly computed values:
        psi[i,j] = 0.25 * (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1])
    
    The order of updates (sweep direction) affects convergence.
    
    Parameters:
        grid (Grid): Grid object with boundary conditions applied
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        convergence_metric (str): 'residual' or 'update'
        sweep_direction (str): 'forward', 'reverse', or 'alternating'
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # Determine sweep direction for this iteration
        if sweep_direction == 'forward':
            i_range = range(1, Nx - 1)
            j_range = range(1, Ny - 1)
        elif sweep_direction == 'reverse':
            i_range = range(Nx - 2, 0, -1)
            j_range = range(Ny - 2, 0, -1)
        elif sweep_direction == 'alternating':
            if iteration % 2 == 1:  # Odd iterations: forward
                i_range = range(1, Nx - 1)
                j_range = range(1, Ny - 1)
            else:  # Even iterations: reverse
                i_range = range(Nx - 2, 0, -1)
                j_range = range(Ny - 2, 0, -1)
        else:
            raise ValueError(f"Unknown sweep direction: {sweep_direction}")
        
        # Update interior points using immediate (Gauss-Seidel) updates
        if abs(dx - dy) < 1e-12:
            for i in i_range:
                for j in j_range:
                    psi[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                        psi[i, j+1] + psi[i, j-1])
        else:
            coeff = 2.0 * (dx2 + dy2)
            for i in i_range:
                for j in j_range:
                    psi[i, j] = (dy2 * (psi[i+1, j] + psi[i-1, j]) +
                                dx2 * (psi[i, j+1] + psi[i, j-1])) / coeff
        
        # Re-apply boundary conditions
        grid.apply_boundary_conditions()
        
        # Compute convergence metric
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"GS ({sweep_direction}) iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Gauss-Seidel converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# LINE GAUSS-SEIDEL SOLVER
# =============================================================================

def line_gauss_seidel_solver(grid, tolerance=1e-7, max_iterations=100000,
                             convergence_metric='residual', sweep_direction='forward',
                             line_direction='x', verbose=False):
    """
    Solve the Laplace equation using Line Gauss-Seidel iteration.
    
    In Line Gauss-Seidel, we solve for an entire line of unknowns simultaneously
    using a tridiagonal system (Thomas algorithm).
    
    For x-line solve (solving along x for fixed j):
        For each j, solve: a*psi[i-1,j] + b*psi[i,j] + c*psi[i+1,j] = d
        where d includes contributions from psi[i,j-1] and psi[i,j+1]
    
    Parameters:
        grid (Grid): Grid object
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        convergence_metric (str): 'residual' or 'update'
        sweep_direction (str): 'forward', 'reverse', or 'alternating'
        line_direction (str): 'x' or 'y' - direction of implicit solve
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # Determine sweep direction
        if sweep_direction == 'forward':
            forward = True
        elif sweep_direction == 'reverse':
            forward = False
        elif sweep_direction == 'alternating':
            forward = (iteration % 2 == 1)
        else:
            raise ValueError(f"Unknown sweep direction: {sweep_direction}")
        
        if line_direction == 'x':
            # Solve along x-direction (lines of constant j)
            # For interior j: j = 1, 2, ..., Ny-2
            n_interior_x = Nx - 2  # Number of interior points in x
            
            j_range = range(1, Ny - 1) if forward else range(Ny - 2, 0, -1)
            
            for j in j_range:
                # Set up tridiagonal system for interior i = 1, ..., Nx-2
                # Equation: (psi[i+1,j] - 2*psi[i,j] + psi[i-1,j])/dx2 
                #         + (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1])/dy2 = 0
                #
                # Rearranging for line solve in x:
                # (1/dx2)*psi[i-1,j] + (-2/dx2 - 2/dy2)*psi[i,j] + (1/dx2)*psi[i+1,j] 
                #     = -(1/dy2)*(psi[i,j+1] + psi[i,j-1])
                
                a = np.ones(n_interior_x) / dx2           # sub-diagonal
                b = np.ones(n_interior_x) * (-2.0/dx2 - 2.0/dy2)  # main diagonal
                c = np.ones(n_interior_x) / dx2           # super-diagonal
                d = np.zeros(n_interior_x)
                
                for idx, i in enumerate(range(1, Nx - 1)):
                    # RHS includes known values from j+1 and j-1
                    d[idx] = -(psi[i, j+1] + psi[i, j-1]) / dy2
                    
                    # Boundary corrections at first and last interior points
                    if i == 1:
                        # psi[0, j] is known (left boundary)
                        d[idx] -= psi[0, j] / dx2
                    if i == Nx - 2:
                        # psi[Nx-1, j] is known (right boundary)
                        d[idx] -= psi[Nx-1, j] / dx2
                
                # Solve tridiagonal system
                solution = thomas_algorithm(a, b, c, d)
                
                # Update psi
                psi[1:Nx-1, j] = solution
        
        else:  # line_direction == 'y'
            # Solve along y-direction (lines of constant i)
            n_interior_y = Ny - 2
            
            i_range = range(1, Nx - 1) if forward else range(Nx - 2, 0, -1)
            
            for i in i_range:
                a = np.ones(n_interior_y) / dy2
                b = np.ones(n_interior_y) * (-2.0/dx2 - 2.0/dy2)
                c = np.ones(n_interior_y) / dy2
                d = np.zeros(n_interior_y)
                
                for idx, j in enumerate(range(1, Ny - 1)):
                    d[idx] = -(psi[i+1, j] + psi[i-1, j]) / dx2
                    
                    if j == 1:
                        d[idx] -= psi[i, 0] / dy2
                    if j == Ny - 2:
                        d[idx] -= psi[i, Ny-1] / dy2
                
                solution = thomas_algorithm(a, b, c, d)
                psi[i, 1:Ny-1] = solution
        
        # Re-apply boundary conditions
        grid.apply_boundary_conditions()
        
        # Compute convergence metric
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"Line GS ({sweep_direction}) iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Line Gauss-Seidel converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# POINT SOR SOLVER
# =============================================================================

def sor_solver(grid, omega=1.5, tolerance=1e-7, max_iterations=100000,
               convergence_metric='residual', verbose=False):
    """
    Solve the Laplace equation using Point SOR (Successive Over-Relaxation).
    
    SOR update formula:
        psi_GS = 0.25 * (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1])
        psi_new[i,j] = (1 - omega) * psi_old[i,j] + omega * psi_GS
    
    where omega is the relaxation parameter:
        - omega = 1: reduces to Gauss-Seidel
        - 1 < omega < 2: over-relaxation (faster convergence)
        - 0 < omega < 1: under-relaxation (more stable)
    
    Parameters:
        grid (Grid): Grid object
        omega (float): Relaxation parameter (1.0 to 1.99)
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        convergence_metric (str): 'residual' or 'update'
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # SOR update for interior points
        if abs(dx - dy) < 1e-12:
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    # Gauss-Seidel update
                    psi_gs = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                     psi[i, j+1] + psi[i, j-1])
                    # SOR update
                    psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        else:
            coeff = 2.0 * (dx2 + dy2)
            for i in range(1, Nx - 1):
                for j in range(1, Ny - 1):
                    psi_gs = (dy2 * (psi[i+1, j] + psi[i-1, j]) +
                             dx2 * (psi[i, j+1] + psi[i, j-1])) / coeff
                    psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        
        grid.apply_boundary_conditions()
        
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"SOR (omega={omega:.2f}) iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Point SOR converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# LINE SOR SOLVER
# =============================================================================

def line_sor_solver(grid, omega=1.5, tolerance=1e-7, max_iterations=100000,
                    convergence_metric='residual', sweep_direction='forward',
                    line_direction='x', verbose=False):
    """
    Solve the Laplace equation using Line SOR.
    
    Line SOR combines line Gauss-Seidel with SOR relaxation:
    1. Solve for entire line using Thomas algorithm (implicit)
    2. Apply SOR relaxation: psi_new = (1-omega)*psi_old + omega*psi_line
    
    Parameters:
        grid (Grid): Grid object
        omega (float): Relaxation parameter
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        convergence_metric (str): 'residual' or 'update'
        sweep_direction (str): 'forward', 'reverse', or 'alternating'
        line_direction (str): 'x' or 'y'
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # Determine sweep direction
        if sweep_direction == 'forward':
            forward = True
        elif sweep_direction == 'reverse':
            forward = False
        elif sweep_direction == 'alternating':
            forward = (iteration % 2 == 1)
        else:
            raise ValueError(f"Unknown sweep direction: {sweep_direction}")
        
        if line_direction == 'x':
            n_interior_x = Nx - 2
            j_range = range(1, Ny - 1) if forward else range(Ny - 2, 0, -1)
            
            for j in j_range:
                a = np.ones(n_interior_x) / dx2
                b = np.ones(n_interior_x) * (-2.0/dx2 - 2.0/dy2)
                c = np.ones(n_interior_x) / dx2
                d = np.zeros(n_interior_x)
                
                for idx, i in enumerate(range(1, Nx - 1)):
                    d[idx] = -(psi[i, j+1] + psi[i, j-1]) / dy2
                    if i == 1:
                        d[idx] -= psi[0, j] / dx2
                    if i == Nx - 2:
                        d[idx] -= psi[Nx-1, j] / dx2
                
                # Line solve
                psi_line = thomas_algorithm(a, b, c, d)
                
                # Apply SOR relaxation
                psi[1:Nx-1, j] = (1.0 - omega) * psi_old[1:Nx-1, j] + omega * psi_line
        
        else:  # y-direction
            n_interior_y = Ny - 2
            i_range = range(1, Nx - 1) if forward else range(Nx - 2, 0, -1)
            
            for i in i_range:
                a = np.ones(n_interior_y) / dy2
                b = np.ones(n_interior_y) * (-2.0/dx2 - 2.0/dy2)
                c = np.ones(n_interior_y) / dy2
                d = np.zeros(n_interior_y)
                
                for idx, j in enumerate(range(1, Ny - 1)):
                    d[idx] = -(psi[i+1, j] + psi[i-1, j]) / dx2
                    if j == 1:
                        d[idx] -= psi[i, 0] / dy2
                    if j == Ny - 2:
                        d[idx] -= psi[i, Ny-1] / dy2
                
                psi_line = thomas_algorithm(a, b, c, d)
                psi[i, 1:Ny-1] = (1.0 - omega) * psi_old[i, 1:Ny-1] + omega * psi_line
        
        grid.apply_boundary_conditions()
        
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"Line SOR (omega={omega:.2f}) iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Line SOR converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# ADI (ALTERNATING DIRECTION IMPLICIT) SOLVER
# =============================================================================

def adi_solver(grid, tolerance=1e-7, max_iterations=100000,
               convergence_metric='residual', verbose=False):
    """
    Solve the Laplace equation using ADI (Alternating Direction Implicit) iteration.
    
    For steady Laplace equation, ADI alternates between:
    1. X-sweep: Implicit in x, explicit in y
    2. Y-sweep: Implicit in y, explicit in x
    
    Each half-step uses the Thomas algorithm for the tridiagonal system.
    
    For the steady Laplace equation, ADI is formulated with a pseudo-time step
    parameter rho. For optimal convergence, rho can be varied (Peaceman-Rachford).
    Here we use a simple fixed-parameter approach.
    
    Parameters:
        grid (Grid): Grid object
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations (each iteration = x-sweep + y-sweep)
        convergence_metric (str): 'residual' or 'update'
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    # ADI parameter (pseudo-time step parameter)
    # For Laplace equation on unit square with uniform grid,
    # optimal single parameter is approximately rho = 1
    rho = 1.0
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # =====================================================================
        # X-SWEEP (implicit in x, explicit in y)
        # =====================================================================
        # Equation: (psi^{n+1/2} - psi^n) / (rho*dx2) = 
        #           d2psi^{n+1/2}/dx2 + d2psi^n/dy2
        #
        # Rearranged:
        # -psi[i-1,j]^{n+1/2}/dx2 + (2/dx2 + 1/(rho*dx2))*psi[i,j]^{n+1/2} 
        #   - psi[i+1,j]^{n+1/2}/dx2 = 
        #   psi[i,j]^n/(rho*dx2) + (psi[i,j+1]^n - 2*psi[i,j]^n + psi[i,j-1]^n)/dy2
        
        psi_half = psi.copy()
        n_interior_x = Nx - 2
        
        for j in range(1, Ny - 1):
            # Tridiagonal coefficients for x-sweep
            a = np.ones(n_interior_x) * (-1.0 / dx2)
            b = np.ones(n_interior_x) * (2.0 / dx2 + 1.0 / (rho * dx2))
            c = np.ones(n_interior_x) * (-1.0 / dx2)
            d = np.zeros(n_interior_x)
            
            for idx, i in enumerate(range(1, Nx - 1)):
                # RHS: explicit y-derivative + psi^n term
                d2psi_dy2 = (psi[i, j+1] - 2.0*psi[i, j] + psi[i, j-1]) / dy2
                d[idx] = psi[i, j] / (rho * dx2) + d2psi_dy2
                
                # Boundary corrections
                if i == 1:
                    d[idx] += psi[0, j] / dx2
                if i == Nx - 2:
                    d[idx] += psi[Nx-1, j] / dx2
            
            solution = thomas_algorithm(a, b, c, d)
            psi_half[1:Nx-1, j] = solution
        
        psi_half[0, :] = psi[0, :]       # Left boundary
        psi_half[-1, :] = psi[-1, :]     # Right boundary
        psi_half[:, 0] = psi[:, 0]       # Bottom boundary
        psi_half[:, -1] = psi[:, -1]     # Top boundary
        
        # =====================================================================
        # Y-SWEEP (implicit in y, explicit in x)
        # =====================================================================
        n_interior_y = Ny - 2
        
        for i in range(1, Nx - 1):
            a = np.ones(n_interior_y) * (-1.0 / dy2)
            b = np.ones(n_interior_y) * (2.0 / dy2 + 1.0 / (rho * dy2))
            c = np.ones(n_interior_y) * (-1.0 / dy2)
            d = np.zeros(n_interior_y)
            
            for idx, j in enumerate(range(1, Ny - 1)):
                d2psi_dx2 = (psi_half[i+1, j] - 2.0*psi_half[i, j] + psi_half[i-1, j]) / dx2
                d[idx] = psi_half[i, j] / (rho * dy2) + d2psi_dx2
                
                if j == 1:
                    d[idx] += psi_half[i, 0] / dy2
                if j == Ny - 2:
                    d[idx] += psi_half[i, Ny-1] / dy2
            
            solution = thomas_algorithm(a, b, c, d)
            psi[i, 1:Ny-1] = solution
        
        # Re-apply boundary conditions
        grid.apply_boundary_conditions()
        
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"ADI iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"ADI converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# RED-BLACK SOR SOLVER
# =============================================================================

def red_black_sor_solver(grid, omega=1.5, tolerance=1e-7, max_iterations=100000,
                         convergence_metric='residual', verbose=False):
    """
    Solve the Laplace equation using Red-Black SOR (checkerboard pattern).
    
    In Red-Black ordering, nodes are colored like a checkerboard:
    - Red nodes: (i + j) is even
    - Black nodes: (i + j) is odd
    
    Update order:
    1. Update all red nodes using current (old) black neighbors
    2. Update all black nodes using newly computed red neighbors
    
    This allows for potential parallelization and often has better convergence
    properties than standard SOR.
    
    Parameters:
        grid (Grid): Grid object
        omega (float): Relaxation parameter
        tolerance (float): Convergence tolerance
        max_iterations (int): Maximum iterations
        convergence_metric (str): 'residual' or 'update'
        verbose (bool): Print progress
        
    Returns:
        grid (Grid): Grid with converged solution
        history (ConvergenceHistory): Convergence history
    """
    history = ConvergenceHistory()
    psi = grid.psi
    Nx, Ny = psi.shape
    dx, dy = grid.dx, grid.dy
    dx2 = dx * dx
    dy2 = dy * dy
    
    # Precompute red and black indices for interior nodes
    # Interior: i in [1, Nx-2], j in [1, Ny-2]
    red_indices = []
    black_indices = []
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if (i + j) % 2 == 0:
                red_indices.append((i, j))
            else:
                black_indices.append((i, j))
    
    for iteration in range(1, max_iterations + 1):
        psi_old = psi.copy()
        
        # Phase 1: Update RED nodes (i + j even)
        if abs(dx - dy) < 1e-12:
            for (i, j) in red_indices:
                psi_gs = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                 psi[i, j+1] + psi[i, j-1])
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        else:
            coeff = 2.0 * (dx2 + dy2)
            for (i, j) in red_indices:
                psi_gs = (dy2 * (psi[i+1, j] + psi[i-1, j]) +
                         dx2 * (psi[i, j+1] + psi[i, j-1])) / coeff
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        
        # Phase 2: Update BLACK nodes (i + j odd)
        if abs(dx - dy) < 1e-12:
            for (i, j) in black_indices:
                psi_gs = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                 psi[i, j+1] + psi[i, j-1])
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        else:
            coeff = 2.0 * (dx2 + dy2)
            for (i, j) in black_indices:
                psi_gs = (dy2 * (psi[i+1, j] + psi[i-1, j]) +
                         dx2 * (psi[i, j+1] + psi[i, j-1])) / coeff
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_gs
        
        grid.apply_boundary_conditions()
        
        if convergence_metric == 'residual':
            metric = compute_residual_l2(psi, dx, dy)
        else:
            metric = compute_update_l2(psi, psi_old)
        
        history.record(residual=metric if convergence_metric == 'residual' else None,
                      update=metric if convergence_metric == 'update' else None)
        
        if verbose and iteration % 1000 == 0:
            print(f"Red-Black SOR (omega={omega:.2f}) iter {iteration}: {convergence_metric} = {metric:.2e}")
        
        converged, reason = check_convergence(metric, tolerance, iteration, max_iterations)
        if converged:
            history.finalize(reason == 'tolerance', reason)
            if verbose:
                print(f"Red-Black SOR converged after {iteration} iterations ({reason})")
            break
    
    return grid, history


# =============================================================================
# SOLVER DISPATCHER
# =============================================================================

SOLVER_METHODS = {
    'jacobi': jacobi_solver,
    'gauss_seidel': gauss_seidel_solver,
    'line_gauss_seidel': line_gauss_seidel_solver,
    'sor': sor_solver,
    'line_sor': line_sor_solver,
    'adi': adi_solver,
    'red_black_sor': red_black_sor_solver
}


def get_solver(method_name):
    """
    Get solver function by name.
    
    Parameters:
        method_name (str): Name of the solver method
        
    Returns:
        function: Solver function
        
    Raises:
        ValueError: If method_name is not recognized
    """
    if method_name not in SOLVER_METHODS:
        raise ValueError(f"Unknown solver method: {method_name}. "
                        f"Available methods: {list(SOLVER_METHODS.keys())}")
    return SOLVER_METHODS[method_name]
