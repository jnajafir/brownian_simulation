#White noise simulation module for Brownian particle simulation.

import numpy as np
import matplotlib.pyplot as plt
from utils import setup_figure, save_figure, plot_trajectory, set_axis_properties

def simulate_white_noise(dt, t_max):
    # Calculate number of steps
    n_steps = int(t_max / dt)
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    w = np.zeros(n_steps)
    x = np.zeros(n_steps)
    
    # Generate white noise with variance 1/dt
    w[1:] = np.random.normal(0, 1/np.sqrt(dt), n_steps-1)
    
    # Simulate free diffusion using Eq. (5): x_i = x_{i-1} + sqrt(dt) * w_i
    for i in range(1, n_steps):
        x[i] = x[i-1] + np.sqrt(dt) * w[i]
    
    return t, w, x

def simulate_multiple_trajectories(dt, t_max, n_trajectories=10000):

    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    
    # Initialize array to store all trajectories
    all_x = np.zeros((n_trajectories, n_steps))
    
    # Generate multiple trajectories
    for j in range(n_trajectories):
        _, _, x = simulate_white_noise(dt, t_max)
        all_x[j, :] = x
    
    # Calculate statistics
    x_mean = np.mean(all_x, axis=0)
    x_std = np.std(all_x, axis=0)
    
    return t, x_mean, x_std

def run_simulation(params):

    # Extract parameters
    dt_values = params.get('dt_values', [0.05, 0.1, 0.5, 1.0])
    t_max = params.get('t_max', 20.0)
    n_trajectories = params.get('n_trajectories', 10000)
    
    # Initialize results dictionary
    results = {
        'dt_values': dt_values,
        't_max': t_max,
        'simulations': []
    }
    
    # Run simulation for each dt value
    for dt in dt_values:
        # Simulate white noise and free diffusion
        t, w, x = simulate_white_noise(dt, t_max)
        
        # Simulate multiple trajectories for statistical analysis
        t_multi, x_mean, x_std = simulate_multiple_trajectories(dt, t_max, n_trajectories)
        
        # Store results
        results['simulations'].append({
            'dt': dt,
            't': t,
            'w': w,
            'x': x,
            't_multi': t_multi,
            'x_mean': x_mean,
            'x_std': x_std
        })
    
    return results

def visualize_results(results, params, save_path=None):

    # Extract results
    dt_values = results['dt_values']
    t_max = results['t_max']
    simulations = results['simulations']
    
    # Create figure with 2 rows and len(dt_values) columns
    fig, axes = plt.subplots(2, len(dt_values), figsize=(16, 8))
    
    # Plot white noise and trajectories for each dt value
    for i, sim in enumerate(simulations):
        dt = sim['dt']
        t = sim['t']
        w = sim['w']
        x = sim['x']
        t_multi = sim['t_multi']
        x_mean = sim['x_mean']
        x_std = sim['x_std']
        
        # Plot white noise (top row)
        ax1 = axes[0, i]
        plot_trajectory(ax1, t, w)
        set_axis_properties(
            ax1, 
            xlabel='t', 
            ylabel='$w_i$', 
            title=f'(a{i+1}) White Noise, Δt = {dt}',
            xlim=(0, t_max),
            ylim=(-8, 8)
        )
        
        # Plot trajectory (bottom row)
        ax2 = axes[1, i]
        plot_trajectory(ax2, t, x)
        
        # Add shaded area for standard deviation
        ax2.fill_between(t_multi, x_mean - x_std, x_mean + x_std, color='gray', alpha=0.3)
        
        set_axis_properties(
            ax2, 
            xlabel='t', 
            ylabel='$x_i$', 
            title=f'(b{i+1}) Free Diffusion, Δt = {dt}',
            xlim=(0, t_max),
            ylim=(-8, 8)
        )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig, 'white_noise_simulation.png', save_path)
    
    return fig
