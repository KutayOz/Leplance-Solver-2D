"""
Output Manager Module
=====================

Manages output directory structure with timestamped folders for each operation.
Keeps outputs organized by operation type and run time.

Directory Structure:
    outputs/
    |-- single_solver/
    |   |-- 2024-12-13_23-45-30_sor/
    |   |   |-- streamlines.png
    |   |   |-- heatmap.png
    |   |   |-- run_info.txt
    |   |-- 2024-12-13_23-50-15_jacobi/
    |       |-- ...
    |-- all_methods/
    |   |-- 2024-12-14_10-30-00/
    |       |-- vertical_cut_comparison.png
    |       |-- convergence_comparison.png
    |       |-- streamlines_point_sor.png
    |       |-- run_info.txt
    |-- omega_study/
    |   |-- 2024-12-14_11-00-00/
    |       |-- omega_optimization_study.png
    |       |-- run_info.txt
    |-- ic_study/
    |-- sweep_study/
"""

import os
from datetime import datetime


class OutputManager:
    """
    Manages output directories with timestamps and organized structure.
    
    Attributes:
        base_dir (str): Base output directory (default: 'outputs')
    """
    
    # Operation type names for subdirectories
    SINGLE_SOLVER = 'single_solver'
    ALL_METHODS = 'all_methods'
    OMEGA_STUDY = 'omega_study'
    IC_STUDY = 'ic_study'
    SWEEP_STUDY = 'sweep_study'
    VALIDATION = 'validation'
    
    def __init__(self, base_dir='outputs'):
        """
        Initialize the output manager.
        
        Parameters:
            base_dir (str): Base directory for all outputs
        """
        self.base_dir = base_dir
        self._ensure_base_dir()
    
    def _ensure_base_dir(self):
        """Create base output directory if it doesn't exist."""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _get_timestamp(self):
        """Get current timestamp string for folder naming."""
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    def create_run_directory(self, operation_type, suffix=''):
        """
        Create a timestamped directory for a specific operation.
        
        Parameters:
            operation_type (str): Type of operation (e.g., 'single_solver', 'omega_study')
            suffix (str): Optional suffix to add (e.g., method name 'sor')
            
        Returns:
            str: Path to the created directory
        """
        # Create operation type subdirectory
        operation_dir = os.path.join(self.base_dir, operation_type)
        os.makedirs(operation_dir, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = self._get_timestamp()
        if suffix:
            run_name = f"{timestamp}_{suffix}"
        else:
            run_name = timestamp
        
        run_dir = os.path.join(operation_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir
    
    def create_single_solver_dir(self, method_name):
        """
        Create directory for single solver run.
        
        Parameters:
            method_name (str): Name of the solver method
            
        Returns:
            str: Path to the created directory
        """
        return self.create_run_directory(self.SINGLE_SOLVER, suffix=method_name)
    
    def create_all_methods_dir(self):
        """
        Create directory for all methods comparison run.
        
        Returns:
            str: Path to the created directory
        """
        return self.create_run_directory(self.ALL_METHODS)
    
    def create_omega_study_dir(self):
        """
        Create directory for omega optimization study.
        
        Returns:
            str: Path to the created directory
        """
        return self.create_run_directory(self.OMEGA_STUDY)
    
    def create_ic_study_dir(self):
        """
        Create directory for initial condition study.
        
        Returns:
            str: Path to the created directory
        """
        return self.create_run_directory(self.IC_STUDY)
    
    def create_sweep_study_dir(self):
        """
        Create directory for sweep direction study.
        
        Returns:
            str: Path to the created directory
        """
        return self.create_run_directory(self.SWEEP_STUDY)
    
    def save_run_info(self, run_dir, info_dict):
        """
        Save run information to a text file in the run directory.
        
        Parameters:
            run_dir (str): Path to the run directory
            info_dict (dict): Dictionary with run information
        """
        info_path = os.path.join(run_dir, 'run_info.txt')
        
        with open(info_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RUN INFORMATION\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 60 + "\n")
            
            for key, value in info_dict.items():
                f.write(f"{key}: {value}\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"  Run info saved to: {info_path}")
    
    def list_recent_runs(self, operation_type, limit=5):
        """
        List recent runs for a given operation type.
        
        Parameters:
            operation_type (str): Type of operation
            limit (int): Maximum number of runs to list
            
        Returns:
            list: List of (run_name, run_path) tuples, most recent first
        """
        operation_dir = os.path.join(self.base_dir, operation_type)
        
        if not os.path.exists(operation_dir):
            return []
        
        # Get all subdirectories
        runs = []
        for name in os.listdir(operation_dir):
            path = os.path.join(operation_dir, name)
            if os.path.isdir(path):
                runs.append((name, path))
        
        # Sort by name (which includes timestamp) in reverse order
        runs.sort(key=lambda x: x[0], reverse=True)
        
        return runs[:limit]
    
    def get_files_in_run(self, run_dir):
        """
        Get list of files in a run directory.
        
        Parameters:
            run_dir (str): Path to run directory
            
        Returns:
            list: List of filenames
        """
        if not os.path.exists(run_dir):
            return []
        
        return [f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))]


# Global output manager instance
_output_manager = None


def get_output_manager(base_dir='outputs'):
    """
    Get or create the global output manager instance.
    
    Parameters:
        base_dir (str): Base output directory
        
    Returns:
        OutputManager: The output manager instance
    """
    global _output_manager
    if _output_manager is None or _output_manager.base_dir != base_dir:
        _output_manager = OutputManager(base_dir)
    return _output_manager
