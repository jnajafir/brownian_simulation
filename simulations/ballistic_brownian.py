# Ballistic to Brownian diffusion simulation module.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import (
    KB, T, R, M, ETA, C, TAU, D, 
    calculate_velocity_autocorrelation, 
    calculate_mean_square_displacement,
    setup_figure, save_figure, plot_trajectory, set_axis_properties
)

def simulate_brownian_motion_with_inertia(t_max, dt):

    # Calculate number of steps
    n_steps = int(t_max / dt) + 1

    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)  # For velocity calculation
    
    # Gaussian white noise
    w = np.random.normal(0, 1, n_steps)
    
    for i in range(2, n_steps):
        # Calculate position using the finite difference equation
        x[i] = ((2 + dt * (C/M)) * x[i-1] - x[i-2]) / (1 + dt * (C/M)) + \
               np.sqrt(2 * KB * T * C) / (M * (1 + dt * (C/M))) * (dt**1.5) * w[i]
        
        # Calculate velocity (for velocity autocorrelation)
        if i > 0:
            v[i-1] = (x[i] - x[i-1]) / dt
    
    # Convert positions to nm
    x = x * 1e9
    
    return t, x, v

def simulate_brownian_motion_without_inertia(t_max, dt):
    # Calculate number of steps
    n_steps = int(t_max / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)  # For velocity calculation
    
    # Generate Gaussian white noise
    w = np.random.normal(0, 1, n_steps)
    
    # Implement Eq. (10) from the paper
    for i in range(1, n_steps):
        x[i] = x[i-1] + np.sqrt(2 * D * dt) * w[i]
        
        # Calculate velocity (for velocity autocorrelation)
        v[i-1] = (x[i] - x[i-1]) / dt
    
    # Convert positions to nm
    x = x * 1e9
    
    return t, x, v

def run_simulation(params):

    dt = params.get('dt', 1e-9)  # Time step much smaller than tau
    t_max_factor = params.get('t_max_factor', 10)  # Simulate for 10 times the momentum relaxation time
    
    # Calculate maximum simulation time
    t_max = t_max_factor * TAU
    
    # Simulate Brownian motion with and without inertia
    t_inertia, x_inertia, v_inertia = simulate_brownian_motion_with_inertia(t_max, dt)
    t_no_inertia, x_no_inertia, v_no_inertia = simulate_brownian_motion_without_inertia(t_max, dt)
    
    # Calculate velocity autocorrelation
    t_lag_v_inertia, Cv_inertia = calculate_velocity_autocorrelation(v_inertia, dt)
    t_lag_v_no_inertia, Cv_no_inertia = calculate_velocity_autocorrelation(v_no_inertia, dt)
    
    # Calculate mean square displacement
    t_lag_msd_inertia, msd_inertia = calculate_mean_square_displacement(x_inertia, dt)
    t_lag_msd_no_inertia, msd_no_inertia = calculate_mean_square_displacement(x_no_inertia, dt)
    
    # Store results
    results = {
        'dt': dt,
        't_max': t_max,
        't_max_factor': t_max_factor,
        'tau': TAU,
        'with_inertia': {
            't': t_inertia,
            'x': x_inertia,
            'v': v_inertia,
            't_lag_v': t_lag_v_inertia,
            'Cv': Cv_inertia,
            't_lag_msd': t_lag_msd_inertia,
            'msd': msd_inertia
        },
        'without_inertia': {
            't': t_no_inertia,
            'x': x_no_inertia,
            'v': v_no_inertia,
            't_lag_v': t_lag_v_no_inertia,
            'Cv': Cv_no_inertia,
            't_lag_msd': t_lag_msd_no_inertia,
            'msd': msd_no_inertia
        }
    }
    
    return results

def visualize_results(results, params, save_path=None):

    tau = results['tau']
    t_inertia = results['with_inertia']['t']
    x_inertia = results['with_inertia']['x']
    t_lag_v_inertia = results['with_inertia']['t_lag_v']
    Cv_inertia = results['with_inertia']['Cv']
    t_lag_msd_inertia = results['with_inertia']['t_lag_msd']
    msd_inertia = results['with_inertia']['msd']
    
    t_no_inertia = results['without_inertia']['t']
    x_no_inertia = results['without_inertia']['x']
    t_lag_v_no_inertia = results['without_inertia']['t_lag_v']
    Cv_no_inertia = results['without_inertia']['Cv']
    t_lag_msd_no_inertia = results['without_inertia']['t_lag_msd']
    msd_no_inertia = results['without_inertia']['msd']
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot trajectories for short time scales (t ≤ tau)
    ax1 = fig.add_subplot(gs[0, 0])
    short_time_idx = int(tau / results['dt'])  # Index corresponding to t = tau
    plot_trajectory(ax1, t_inertia[:short_time_idx] / tau, x_inertia[:short_time_idx], label='With inertia')
    plot_trajectory(ax1, t_no_inertia[:short_time_idx] / tau, x_no_inertia[:short_time_idx], label='Without inertia', linestyle='--')
    set_axis_properties(
        ax1,
        xlabel='t/τ',
        ylabel='X (nm)',
        title='(a) Trajectories for t ≤ τ',
        xlim=(0, 1)
    )
    ax1.legend()
    
    # Plot trajectories for long time scales (t > tau)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_trajectory(ax2, t_inertia / tau, x_inertia, label='With inertia')
    plot_trajectory(ax2, t_no_inertia / tau, x_no_inertia, label='Without inertia', linestyle='--')
    set_axis_properties(
        ax2,
        xlabel='t/τ',
        ylabel='X (nm)',
        title='(b) Trajectories for t > τ',
        xlim=(1, 10)
    )
    ax2.legend()
    
    # Plot velocity autocorrelation function
    ax3 = fig.add_subplot(gs[1, 0])
    plot_trajectory(ax3, t_lag_v_inertia / tau, Cv_inertia, label='With inertia')
    plot_trajectory(ax3, t_lag_v_no_inertia / tau, Cv_no_inertia, label='Without inertia', linestyle='--')
    set_axis_properties(
        ax3,
        xlabel='t/τ',
        ylabel='Cv(t) (a.u.)',
        title='(c) Velocity Autocorrelation',
        xlim=(-6, 6)
    )
    ax3.legend()
    
    # Plot mean square displacement (log-log scale)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.loglog(t_lag_msd_inertia / tau, msd_inertia, 'k-', label='With inertia')
    ax4.loglog(t_lag_msd_no_inertia / tau, msd_no_inertia, 'k--', label='Without inertia')
    
    # Add reference lines for t and t^2 scaling
    t_ref = np.logspace(-2, 1, 100)
    ax4.loglog(t_ref, 2*D*1e18*t_ref*tau, 'k:', label='∝ t')
    ax4.loglog(t_ref, (KB*T/M)*1e18*(t_ref*tau)**2, 'k-.', label='∝ t²')
    
    set_axis_properties(
        ax4,
        xlabel='t/τ',
        ylabel='<x(t)²> (nm²)',
        title='(d) Mean Square Displacement'
    )
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig, 'ballistic_to_brownian.png', save_path)
    
    return fig
