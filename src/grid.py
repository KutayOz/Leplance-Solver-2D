"""
Grid Module for 2D Laplace Solver
=================================

This module defines the computational grid and boundary conditions for solving
the 2D Laplace equation for the stream function psi(x,y).

Physics Background (from code comments):
-----------------------------------------
2D incompressible flow stream function definition:
    u = d(psi)/dy      (velocity in x-direction)
    v = -d(psi)/dx     (velocity in y-direction)

For irrotational flow, vorticity omega = 0, which leads to Laplace equation:
    d2(psi)/dx2 + d2(psi)/dy2 = 0

We solve this PDE for psi on a rectangular domain with Dirichlet boundary conditions.

Coordinate-Index Mapping:
-------------------------
    x_i = i * dx,  i = 0, 1, ..., Nx-1  (Nx = 101 for dx = 0.1, domain [0, 10])
    y_j = j * dy,  j = 0, 1, ..., Ny-1  (Ny = 101 for dy = 0.1, domain [0, 10])

Key index values for dx = dy = 0.1:
    x = 2.0  -> i = 20
    x = 2.4  -> i = 24
    y = 6.0  -> j = 60
    y = 6.4  -> j = 64
"""

import numpy as np


class Grid:
    """
    Represents the computational grid for the 2D Laplace solver.
    
    Attributes:
        Lx (float): Domain length in x-direction (default 10.0)
        Ly (float): Domain length in y-direction (default 10.0)
        dx (float): Grid spacing in x-direction (default 0.1)
        dy (float): Grid spacing in y-direction (default 0.1)
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        x (ndarray): 1D array of x-coordinates
        y (ndarray): 1D array of y-coordinates
        psi (ndarray): 2D array of stream function values, shape (Nx, Ny)
    """
    
    def __init__(self, Lx=10.0, Ly=10.0, dx=0.1, dy=0.1):
        """
        Initialize the grid with specified domain size and spacing.
        
        Parameters:
            Lx (float): Domain length in x-direction
            Ly (float): Domain length in y-direction
            dx (float): Grid spacing in x-direction
            dy (float): Grid spacing in y-direction
        """
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        
        # Compute number of grid points
        # Nx = Lx/dx + 1 to include both endpoints (0 and Lx)
        self.Nx = int(round(Lx / dx)) + 1
        self.Ny = int(round(Ly / dy)) + 1
        
        # Create coordinate arrays
        # x[i] = i * dx for i = 0, 1, ..., Nx-1
        # y[j] = j * dy for j = 0, 1, ..., Ny-1
        self.x = np.linspace(0, Lx, self.Nx)
        self.y = np.linspace(0, Ly, self.Ny)
        
        # Initialize psi array with zeros
        # psi[i, j] corresponds to psi(x_i, y_j)
        # Convention: first index i is x-direction, second index j is y-direction
        self.psi = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        
        # Store key indices for boundary openings
        # These are computed based on dx = dy = 0.1 assumption
        # Bottom inlet: x in [2.0, 2.4] at y = 0
        self.inlet_i_start = int(round(2.0 / dx))   # i = 20
        self.inlet_i_end = int(round(2.4 / dx))     # i = 24
        
        # Right outlet: y in [6.0, 6.4] at x = 10
        self.outlet_j_start = int(round(6.0 / dy))  # j = 60
        self.outlet_j_end = int(round(6.4 / dy))    # j = 64
        
    def index_to_x(self, i):
        """Convert x-index to physical x-coordinate: x = i * dx"""
        return i * self.dx
    
    def index_to_y(self, j):
        """Convert y-index to physical y-coordinate: y = j * dy"""
        return j * self.dy
    
    def x_to_index(self, x):
        """Convert physical x-coordinate to nearest x-index: i = round(x / dx)"""
        return int(round(x / self.dx))
    
    def y_to_index(self, y):
        """Convert physical y-coordinate to nearest y-index: j = round(y / dy)"""
        return int(round(y / self.dy))
    
    def apply_boundary_conditions(self):
        """
        Apply Dirichlet boundary conditions for the stream function psi.
        
        Boundary Conditions:
        --------------------
        1. Left wall (x = 0, i = 0): psi = 0 for all y (j = 0 to Ny-1)
        
        2. Top wall (y = 10, j = Ny-1): psi = 0 for all x (i = 0 to Nx-1)
        
        3. Bottom boundary (y = 0, j = 0):
           - For x in [0, 2.0] (i = 0 to 20): psi = 0
           - For x in [2.4, 10.0] (i = 24 to 100): psi = 10
           - Inlet opening x in [2.0, 2.4] (i = 20 to 24): 
             psi(x, 0) = 10 * (x - 2.0) / 0.4 (linear from 0 to 10)
        
        4. Right boundary (x = 10, i = Nx-1):
           - For y in [0, 6.0] (j = 0 to 60): psi = 10
           - For y in [6.4, 10.0] (j = 64 to 100): psi = 0
           - Outlet opening y in [6.0, 6.4] (j = 60 to 64):
             psi(10, y) = 10 * (6.4 - y) / 0.4 (linear from 10 down to 0)
             equivalently: psi = 10 - 25 * (y - 6.0)
        
        Note: These boundary values must be enforced at every iteration.
        """
        # =====================================================================
        # 1. LEFT WALL: x = 0, i = 0
        #    psi = 0 for all y
        # =====================================================================
        self.psi[0, :] = 0.0
        
        # =====================================================================
        # 2. TOP WALL: y = Ly, j = Ny - 1
        #    psi = 0 for all x
        # =====================================================================
        self.psi[:, self.Ny - 1] = 0.0
        
        # =====================================================================
        # 3. BOTTOM BOUNDARY: y = 0, j = 0
        # =====================================================================
        # 3a. For x in [0, 2.0], i in [0, 20]: psi = 0
        # Using inclusive range: i = 0, 1, ..., inlet_i_start
        for i in range(0, self.inlet_i_start + 1):
            self.psi[i, 0] = 0.0
        
        # 3b. Inlet opening: x in [2.0, 2.4], i in [20, 24]
        # Linear interpolation: psi(x, 0) = 10 * (x - 2.0) / 0.4
        # At i = 20 (x = 2.0): psi = 0
        # At i = 24 (x = 2.4): psi = 10
        for i in range(self.inlet_i_start, self.inlet_i_end + 1):
            x_val = self.index_to_x(i)
            self.psi[i, 0] = 10.0 * (x_val - 2.0) / 0.4
        
        # 3c. For x in [2.4, 10.0], i in [24, Nx-1]: psi = 10
        for i in range(self.inlet_i_end, self.Nx):
            self.psi[i, 0] = 10.0
        
        # =====================================================================
        # 4. RIGHT BOUNDARY: x = Lx, i = Nx - 1
        # =====================================================================
        # 4a. For y in [0, 6.0], j in [0, 60]: psi = 10
        for j in range(0, self.outlet_j_start + 1):
            self.psi[self.Nx - 1, j] = 10.0
        
        # 4b. Outlet opening: y in [6.0, 6.4], j in [60, 64]
        # Linear interpolation: psi(10, y) = 10 * (6.4 - y) / 0.4
        # Equivalently: psi = 10 - 25 * (y - 6.0)
        # At j = 60 (y = 6.0): psi = 10
        # At j = 64 (y = 6.4): psi = 0
        for j in range(self.outlet_j_start, self.outlet_j_end + 1):
            y_val = self.index_to_y(j)
            self.psi[self.Nx - 1, j] = 10.0 * (6.4 - y_val) / 0.4
        
        # 4c. For y in [6.4, 10.0], j in [64, Ny-1]: psi = 0
        for j in range(self.outlet_j_end, self.Ny):
            self.psi[self.Nx - 1, j] = 0.0
    
    def initialize_interior(self, value=0.0, mode='uniform'):
        """
        Initialize interior grid points with specified values.
        
        Parameters:
            value (float): Value to use for uniform initialization
            mode (str): Initialization mode
                - 'uniform': All interior points set to value
                - 'linear_x': Linear interpolation in x from left to right boundary
                - 'linear_y': Linear interpolation in y from bottom to top boundary
                - 'average': Average of boundary values
        """
        if mode == 'uniform':
            # Set all interior points to the specified value
            # Interior points: i in [1, Nx-2], j in [1, Ny-2]
            self.psi[1:self.Nx-1, 1:self.Ny-1] = value
            
        elif mode == 'linear_x':
            # Linear interpolation between left (psi=0) and average right boundary
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    # Simple linear blend from 0 to 5 across domain
                    self.psi[i, j] = 5.0 * i / (self.Nx - 1)
                    
        elif mode == 'linear_y':
            # Linear interpolation in y
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    self.psi[i, j] = 5.0 * (1.0 - j / (self.Ny - 1))
                    
        elif mode == 'average':
            # Set to average boundary value (roughly 5.0)
            self.psi[1:self.Nx-1, 1:self.Ny-1] = 5.0
        
        # Always re-apply boundary conditions after initialization
        self.apply_boundary_conditions()
    
    def get_boundary_mask(self):
        """
        Return a boolean mask indicating boundary nodes.
        True = boundary node, False = interior node.
        """
        mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        mask[0, :] = True    # Left wall
        mask[-1, :] = True   # Right wall
        mask[:, 0] = True    # Bottom wall
        mask[:, -1] = True   # Top wall
        return mask
    
    def copy_psi(self):
        """Return a copy of the current psi array."""
        return self.psi.copy()
    
    def get_psi_at_x(self, x_target):
        """
        Extract psi values along a vertical line at x = x_target.
        
        Parameters:
            x_target (float): x-coordinate for the vertical cut
            
        Returns:
            y_values (ndarray): Array of y-coordinates
            psi_values (ndarray): Array of psi values at (x_target, y)
        """
        i = self.x_to_index(x_target)
        return self.y.copy(), self.psi[i, :].copy()
    
    def __repr__(self):
        return (f"Grid(Lx={self.Lx}, Ly={self.Ly}, dx={self.dx}, dy={self.dy}, "
                f"Nx={self.Nx}, Ny={self.Ny})")
