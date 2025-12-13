"""
Convergence Module for 2D Laplace Solver
========================================

This module provides convergence metrics and stopping criteria for the iterative
solvers. Two main convergence metrics are supported:

1. Residual-based L2 norm: Computes the discrete Laplacian residual at interior
   nodes and calculates the RMS (root mean square) or L2 norm.

2. Update-based L2 norm: Computes the RMS/L2 of (psi_new - psi_old) at interior
   nodes.

Discretization (5-point Laplacian):
-----------------------------------
The Laplace equation d2(psi)/dx2 + d2(psi)/dy2 = 0 is discretized using the
standard 5-point finite difference stencil:

    (psi[i+1,j] - 2*psi[i,j] + psi[i-1,j]) / dx^2
  + (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1]) / dy^2 = 0

The residual at node (i,j) is defined as:
    R[i,j] = (psi[i+1,j] + psi[i-1,j] - 2*psi[i,j]) / dx^2
           + (psi[i,j+1] + psi[i,j-1] - 2*psi[i,j]) / dy^2

For dx = dy = h, this simplifies to:
    R[i,j] = (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1] - 4*psi[i,j]) / h^2
"""

import numpy as np


def compute_residual_l2(psi, dx, dy):
    """
    Compute the L2 (RMS) norm of the discrete Laplacian residual at interior nodes.
    
    The residual measures how well the current solution satisfies the Laplace
    equation. A residual of zero means the discrete equation is exactly satisfied.
    
    Parameters:
        psi (ndarray): 2D array of stream function values, shape (Nx, Ny)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        
    Returns:
        float: L2 (RMS) norm of the residual at interior nodes
        
    Notes:
        Interior nodes are i in [1, Nx-2] and j in [1, Ny-2].
        The residual is computed using the 5-point Laplacian stencil.
    """
    Nx, Ny = psi.shape
    
    # Precompute inverse squared spacings
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    # Compute residual using vectorized operations for efficiency
    # Interior points: psi[1:-1, 1:-1]
    # d2psi/dx2 approximation: (psi[i+1,j] - 2*psi[i,j] + psi[i-1,j]) / dx^2
    d2psi_dx2 = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) * dx2_inv
    
    # d2psi/dy2 approximation: (psi[i,j+1] - 2*psi[i,j] + psi[i,j-1]) / dy^2
    d2psi_dy2 = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) * dy2_inv
    
    # Residual = d2psi/dx2 + d2psi/dy2 (should be zero for exact solution)
    residual = d2psi_dx2 + d2psi_dy2
    
    # Compute L2 (RMS) norm
    # RMS = sqrt(sum(R^2) / N) where N is number of interior nodes
    n_interior = (Nx - 2) * (Ny - 2)
    l2_norm = np.sqrt(np.sum(residual ** 2) / n_interior)
    
    return l2_norm


def compute_residual_linf(psi, dx, dy):
    """
    Compute the L-infinity (maximum) norm of the discrete Laplacian residual.
    
    Parameters:
        psi (ndarray): 2D array of stream function values, shape (Nx, Ny)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        
    Returns:
        float: Maximum absolute residual at any interior node
    """
    Nx, Ny = psi.shape
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    d2psi_dx2 = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) * dx2_inv
    d2psi_dy2 = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) * dy2_inv
    
    residual = d2psi_dx2 + d2psi_dy2
    
    return np.max(np.abs(residual))


def compute_update_l2(psi_new, psi_old):
    """
    Compute the L2 (RMS) norm of the update (psi_new - psi_old) at interior nodes.
    
    This metric measures how much the solution changed between iterations.
    When converged, this value approaches zero.
    
    Parameters:
        psi_new (ndarray): New stream function values, shape (Nx, Ny)
        psi_old (ndarray): Previous stream function values, shape (Nx, Ny)
        
    Returns:
        float: L2 (RMS) norm of the update at interior nodes
    """
    # Compute difference at interior nodes only
    # Interior: [1:-1, 1:-1]
    diff = psi_new[1:-1, 1:-1] - psi_old[1:-1, 1:-1]
    
    # Number of interior nodes
    n_interior = diff.size
    
    # L2 (RMS) norm
    l2_norm = np.sqrt(np.sum(diff ** 2) / n_interior)
    
    return l2_norm


def compute_update_linf(psi_new, psi_old):
    """
    Compute the L-infinity (maximum) norm of the update.
    
    Parameters:
        psi_new (ndarray): New stream function values
        psi_old (ndarray): Previous stream function values
        
    Returns:
        float: Maximum absolute change at any interior node
    """
    diff = psi_new[1:-1, 1:-1] - psi_old[1:-1, 1:-1]
    return np.max(np.abs(diff))


def check_convergence(metric_value, tolerance, iteration, max_iterations):
    """
    Check if the solver has converged or should stop.
    
    Parameters:
        metric_value (float): Current convergence metric value (residual or update norm)
        tolerance (float): Convergence tolerance
        iteration (int): Current iteration number
        max_iterations (int): Maximum allowed iterations
        
    Returns:
        tuple: (converged, reason)
            converged (bool): True if solver should stop
            reason (str): Reason for stopping ('tolerance', 'max_iterations', or 'continue')
    """
    if metric_value < tolerance:
        return True, 'tolerance'
    elif iteration >= max_iterations:
        return True, 'max_iterations'
    else:
        return False, 'continue'


def compare_solutions(psi1, psi2):
    """
    Compare two converged solutions by computing their L2 difference.
    
    This is used to verify that different solvers converge to the same solution.
    
    Parameters:
        psi1 (ndarray): First solution array
        psi2 (ndarray): Second solution array
        
    Returns:
        float: L2 (RMS) norm of the difference at interior nodes
    """
    diff = psi1[1:-1, 1:-1] - psi2[1:-1, 1:-1]
    n_interior = diff.size
    return np.sqrt(np.sum(diff ** 2) / n_interior)


class ConvergenceHistory:
    """
    Class to track and store convergence history during iterations.
    
    Attributes:
        residuals (list): List of residual values at each iteration
        updates (list): List of update norms at each iteration
        iterations (int): Total number of iterations performed
        converged (bool): Whether the solver converged
        stop_reason (str): Reason for stopping
    """
    
    def __init__(self):
        self.residuals = []
        self.updates = []
        self.iterations = 0
        self.converged = False
        self.stop_reason = ''
    
    def record(self, residual=None, update=None):
        """Record convergence metrics for current iteration."""
        if residual is not None:
            self.residuals.append(residual)
        if update is not None:
            self.updates.append(update)
        self.iterations += 1
    
    def finalize(self, converged, reason):
        """Finalize the convergence history."""
        self.converged = converged
        self.stop_reason = reason
    
    def get_residuals(self):
        """Return residual history as numpy array."""
        return np.array(self.residuals)
    
    def get_updates(self):
        """Return update norm history as numpy array."""
        return np.array(self.updates)
    
    def __repr__(self):
        return (f"ConvergenceHistory(iterations={self.iterations}, "
                f"converged={self.converged}, reason='{self.stop_reason}')")
