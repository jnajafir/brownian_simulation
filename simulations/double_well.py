"""
Double-well potential simulation module.
This module implements the double-well potential simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    KB, T, setup_figure, save_figure, plot_trajectory, set_axis_properties
)

def simulate_double_well(a, b, gamma, dt, n_steps):

    time = np.linspace(0, n_steps*dt, n_steps)
    x = np.zeros(n_steps)
    
    # Initial condition
    x[0] = 500e-9  # Start near +500 nm
    
    # Diffusion coefficient
    D = KB * T / gamma
    
    # Simulation loop (Euler-Maruyama method)
    for i in range(1, n_steps):
        force = -(a * x[i-1]**3 - b * x[i-1])
        noise = np.sqrt(2 * D * dt) * np.random.randn()
        x[i] = x[i-1] + (force / gamma) * dt + noise
    
    x_nm = x * 1e9
    
    return time, x_nm

def run_simulation(params):

    #Parameters
    a = params.get('a', 1.0e7)  # N/m^3
    b = params.get('b', 1.0e-6)  # N/m
    gamma = params.get('gamma', 1e-8)  # kg/s (typical nanoscale drag)
    dt = params.get('dt', 1e-4)  # Time step in seconds
    n_steps = params.get('n_steps', 2000000)  # Number of steps
    
    time, x = simulate_double_well(a, b, gamma, dt, n_steps)
    
    results = {
        'a': a,
        'b': b,
        'gamma': gamma,
        'dt': dt,
        'n_steps': n_steps,
        'time': time,
        'x': x
    }
    
    return results

def visualize_results(results, params, save_path=None):

    time = results['time']
    x = results['x']
    a = results['a']
    b = results['b']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot trajectory
    plot_trajectory(ax, time, x, color='b')#, linewidth=0.5)
    
    # Calculate equilibrium positions
    eq_pos = np.sqrt(b/a) * 1e9  # in nm
    
    # Add horizontal lines at equilibrium positions
    ax.axhline(y=eq_pos, color='r', linestyle='--', alpha=0.7, label=f'Equilibrium at ±{eq_pos:.1f} nm')
    ax.axhline(y=-eq_pos, color='r', linestyle='--', alpha=0.7)
    
    set_axis_properties(
        ax,
        xlabel='t [s]',
        ylabel='x [nm]',
        title='Kramers Transitions in a Double-Well Potential'
    )
    
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig, 'double_well_potential.png', save_path)
    
    return fig
