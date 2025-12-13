"""
Plotting Module for 2D Laplace Solver
=====================================

This module provides visualization functions for the stream function solution,
convergence analysis, and parameter studies.

Required Plots:
---------------
1. Vertical cut comparison: psi(y) at x = 5.0 for all methods
2. Streamlines: Contour plot of psi over the domain
3. Omega optimization: Iterations vs omega for SOR methods
4. Initial condition sensitivity: Convergence history for different IC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def plot_vertical_cut_comparison(results_dict, x_cut=5.0, output_dir='outputs'):
    """
    Plot psi(y) along a vertical cut at x = x_cut for all methods.
    
    Parameters:
        results_dict (dict): Dictionary mapping method names to (grid, history) tuples
        x_cut (float): x-coordinate for the vertical cut (default 5.0)
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and line styles for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    for idx, (method_name, (grid, history)) in enumerate(results_dict.items()):
        y_values, psi_values = grid.get_psi_at_x(x_cut)
        
        ax.plot(y_values, psi_values, 
                color=colors[idx], 
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2,
                label=f'{method_name} ({history.iterations} iters)',
                marker='o' if len(y_values) < 30 else None,
                markersize=3)
    
    ax.set_xlabel('y', fontsize=12)
    ax.set_ylabel('psi(x={:.1f}, y)'.format(x_cut), fontsize=12)
    ax.set_title('Stream Function Along Vertical Cut at x = {:.1f}'.format(x_cut), fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    
    filepath = os.path.join(output_dir, 'vertical_cut_comparison.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved vertical cut comparison to {filepath}")
    return filepath


def plot_streamlines(grid, output_dir='outputs', method_name='solution', 
                     num_contours=20, show_colorbar=True):
    """
    Plot streamlines (contours of psi) over the full domain.
    
    Streamlines are lines of constant psi. For 2D incompressible flow,
    these represent the paths that fluid particles follow.
    
    Parameters:
        grid (Grid): Grid object with converged solution
        output_dir (str): Directory to save the plot
        method_name (str): Name for the title and filename
        num_contours (int): Number of contour levels
        show_colorbar (bool): Whether to show colorbar
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create meshgrid for plotting
    # Note: psi[i, j] corresponds to (x[i], y[j])
    # For contour plot, need to transpose so that x is horizontal axis
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    
    # Determine contour levels
    psi_min = grid.psi.min()
    psi_max = grid.psi.max()
    levels = np.linspace(psi_min, psi_max, num_contours)
    
    # Filled contours for background
    cf = ax.contourf(X, Y, grid.psi, levels=levels, cmap='RdYlBu_r', alpha=0.8)
    
    # Contour lines (streamlines)
    cs = ax.contour(X, Y, grid.psi, levels=levels, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    if show_colorbar:
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8, label='Stream function psi')
    
    # Mark inlet and outlet regions
    # Bottom inlet: x in [2.0, 2.4] at y = 0
    ax.plot([2.0, 2.4], [0, 0], 'g-', linewidth=4, label='Inlet')
    # Right outlet: y in [6.0, 6.4] at x = 10
    ax.plot([10, 10], [6.0, 6.4], 'r-', linewidth=4, label='Outlet')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Streamlines (Contours of psi) - {method_name}', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    
    filepath = os.path.join(output_dir, f'streamlines_{method_name.replace(" ", "_").lower()}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved streamlines plot to {filepath}")
    return filepath


def plot_omega_study(omega_values, iterations_point_sor, iterations_line_sor=None,
                     output_dir='outputs'):
    """
    Plot iterations to convergence vs omega for SOR methods.
    
    Parameters:
        omega_values (array-like): Array of omega values tested
        iterations_point_sor (array-like): Iterations for Point SOR at each omega
        iterations_line_sor (array-like, optional): Iterations for Line SOR at each omega
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot Point SOR
    ax.plot(omega_values, iterations_point_sor, 'b-o', linewidth=2, 
            markersize=4, label='Point SOR')
    
    # Find optimal omega for Point SOR
    idx_opt_point = np.argmin(iterations_point_sor)
    omega_opt_point = omega_values[idx_opt_point]
    iters_opt_point = iterations_point_sor[idx_opt_point]
    ax.axvline(x=omega_opt_point, color='b', linestyle='--', alpha=0.5)
    ax.annotate(f'omega_opt = {omega_opt_point:.2f}\n({iters_opt_point} iters)',
                xy=(omega_opt_point, iters_opt_point),
                xytext=(omega_opt_point + 0.1, iters_opt_point * 1.5),
                fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    # Plot Line SOR if provided
    if iterations_line_sor is not None:
        ax.plot(omega_values, iterations_line_sor, 'r-s', linewidth=2,
                markersize=4, label='Line SOR')
        
        idx_opt_line = np.argmin(iterations_line_sor)
        omega_opt_line = omega_values[idx_opt_line]
        iters_opt_line = iterations_line_sor[idx_opt_line]
        ax.axvline(x=omega_opt_line, color='r', linestyle='--', alpha=0.5)
        ax.annotate(f'omega_opt = {omega_opt_line:.2f}\n({iters_opt_line} iters)',
                    xy=(omega_opt_line, iters_opt_line),
                    xytext=(omega_opt_line - 0.2, iters_opt_line * 1.5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Relaxation Parameter (omega)', fontsize=12)
    ax.set_ylabel('Iterations to Convergence', fontsize=12)
    ax.set_title('Optimal Omega Study for SOR Methods', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([omega_values[0] - 0.05, omega_values[-1] + 0.05])
    ax.set_yscale('log')
    
    filepath = os.path.join(output_dir, 'omega_optimization_study.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved omega study plot to {filepath}")
    return filepath


def plot_initial_condition_study(histories_dict, output_dir='outputs'):
    """
    Plot convergence history for different initial conditions.
    
    Parameters:
        histories_dict (dict): Dictionary mapping IC description to ConvergenceHistory
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(histories_dict)))
    
    for idx, (ic_name, history) in enumerate(histories_dict.items()):
        # Use residuals if available, otherwise updates
        if len(history.residuals) > 0:
            metric_values = history.get_residuals()
            ylabel = 'Residual L2 Norm'
        else:
            metric_values = history.get_updates()
            ylabel = 'Update L2 Norm'
        
        iterations = np.arange(1, len(metric_values) + 1)
        ax.semilogy(iterations, metric_values, 
                   color=colors[idx], linewidth=1.5,
                   label=f'{ic_name} ({history.iterations} iters)')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Convergence History for Different Initial Conditions (Point SOR)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    filepath = os.path.join(output_dir, 'initial_condition_study.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved initial condition study plot to {filepath}")
    return filepath


def plot_convergence_comparison(histories_dict, output_dir='outputs'):
    """
    Plot convergence history comparison for all methods.
    
    Parameters:
        histories_dict (dict): Dictionary mapping method names to ConvergenceHistory
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    for idx, (method_name, history) in enumerate(histories_dict.items()):
        if len(history.residuals) > 0:
            metric_values = history.get_residuals()
        else:
            metric_values = history.get_updates()
        
        iterations = np.arange(1, len(metric_values) + 1)
        ax.semilogy(iterations, metric_values,
                   color=colors[idx],
                   linestyle=linestyles[idx % len(linestyles)],
                   linewidth=1.5,
                   label=f'{method_name}')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Convergence Metric (L2 Norm)', fontsize=12)
    ax.set_title('Convergence History Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    filepath = os.path.join(output_dir, 'convergence_comparison.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved convergence comparison to {filepath}")
    return filepath


def plot_solution_heatmap(grid, output_dir='outputs', method_name='solution'):
    """
    Plot the stream function as a heatmap.
    
    Parameters:
        grid (Grid): Grid object with solution
        output_dir (str): Directory to save the plot
        method_name (str): Name for the title and filename
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    
    im = ax.pcolormesh(X, Y, grid.psi, cmap='viridis', shading='auto')
    fig.colorbar(im, ax=ax, label='psi')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Stream Function psi - {method_name}', fontsize=14)
    ax.set_aspect('equal')
    
    filepath = os.path.join(output_dir, f'heatmap_{method_name.replace(" ", "_").lower()}.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved heatmap to {filepath}")
    return filepath


def plot_sweep_direction_comparison(results_dict, output_dir='outputs'):
    """
    Plot convergence comparison for different sweep directions.
    
    Parameters:
        results_dict (dict): Dictionary with sweep direction names as keys
                            and (grid, history) tuples as values
        output_dir (str): Directory to save the plot
        
    Returns:
        str: Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (direction_name, (grid, history)) in enumerate(results_dict.items()):
        if len(history.residuals) > 0:
            metric_values = history.get_residuals()
        else:
            metric_values = history.get_updates()
        
        iterations = np.arange(1, len(metric_values) + 1)
        ax.semilogy(iterations, metric_values,
                   color=colors[idx % len(colors)],
                   linewidth=1.5,
                   label=f'{direction_name} ({history.iterations} iters)')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Convergence Metric (L2 Norm)', fontsize=12)
    ax.set_title('Sweep Direction Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    filepath = os.path.join(output_dir, 'sweep_direction_comparison.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved sweep direction comparison to {filepath}")
    return filepath
