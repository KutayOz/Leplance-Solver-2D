# Laplace Solver Package
# Solves 2D Laplace equation for stream function using various iterative methods

from .grid import Grid
from .solvers import (
    jacobi_solver,
    gauss_seidel_solver,
    line_gauss_seidel_solver,
    sor_solver,
    line_sor_solver,
    adi_solver,
    red_black_sor_solver
)
from .convergence import compute_residual_l2, compute_update_l2
from .plotting import (
    plot_vertical_cut_comparison,
    plot_streamlines,
    plot_omega_study,
    plot_initial_condition_study
)

__all__ = [
    'Grid',
    'jacobi_solver',
    'gauss_seidel_solver', 
    'line_gauss_seidel_solver',
    'sor_solver',
    'line_sor_solver',
    'adi_solver',
    'red_black_sor_solver',
    'compute_residual_l2',
    'compute_update_l2',
    'plot_vertical_cut_comparison',
    'plot_streamlines',
    'plot_omega_study',
    'plot_initial_condition_study'
]
