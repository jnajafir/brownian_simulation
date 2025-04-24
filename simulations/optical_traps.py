#Optical traps simulation module.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from utils import (
    KB, T, R, M, ETA, C, D,
    calculate_position_autocorrelation, calculate_mean_square_displacement,
    setup_figure, save_figure, plot_3d_trajectory, plot_probability_distribution_2d,
    add_colorbar, set_axis_properties
)

def simulate_optical_trap_3d(k_x, k_y, k_z, t_max, dt):

    # Stiffness from fN/nm to N/m
    k_x = k_x * 1e-6  # fN/nm to N/m
    k_y = k_y * 1e-6  # fN/nm to N/m
    k_z = k_z * 1e-6  # fN/nm to N/m
    
    # Calculate number of steps
    n_steps = int(t_max / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    
    # Generate Gaussian white noise
    w_x = np.random.normal(0, 1, n_steps)
    w_y = np.random.normal(0, 1, n_steps)
    w_z = np.random.normal(0, 1, n_steps)
    
    # Implement Eq. (16) from the paper
    for i in range(1, n_steps):
        x[i] = x[i-1] - (k_x/C) * x[i-1] * dt + np.sqrt(2*D*dt) * w_x[i]
        y[i] = y[i-1] - (k_y/C) * y[i-1] * dt + np.sqrt(2*D*dt) * w_y[i]
        z[i] = z[i-1] - (k_z/C) * z[i-1] * dt + np.sqrt(2*D*dt) * w_z[i]
    
    # Convert positions to nm
    x = x * 1e9
    y = y * 1e9
    z = z * 1e9
    
    return t, x, y, z

def run_simulation(params):
    # Extract parameters for 3D trajectory
    k_x = params.get('k_x', 1.0)  # trap stiffness in x-direction in fN/nm
    k_y = params.get('k_y', 1.0)  # trap stiffness in y-direction in fN/nm
    k_z = params.get('k_z', 0.2)  # trap stiffness in z-direction in fN/nm
    t_max = params.get('t_max', 200.0)  # simulation time in seconds
    dt = params.get('dt', 0.001)  # time step in seconds
    
    # Simulate optical trap in 3D
    t, x, y, z = simulate_optical_trap_3d(k_x, k_y, k_z, t_max, dt)
    
    # Calculate position autocorrelation
    t_lag, Cx = calculate_position_autocorrelation(x, dt)
    
    # Calculate mean square displacement
    t_lag_msd, msd = calculate_mean_square_displacement(x, dt)
    
    # Simulate for different stiffness values for variance vs. stiffness plot
    k_xy_values = [0.2, 1.0, 5.0]  # trap stiffness values in fN/nm
    variance_data = []
    
    for k_xy in k_xy_values:
        # Simulate optical trap in 2D (using same stiffness for x and y)
        _, x_sim, y_sim, _ = simulate_optical_trap_3d(k_xy, k_xy, k_xy/5, 10.0, dt)
        
        # Calculate variance in xy-plane
        var_xy = np.var(x_sim) + np.var(y_sim)
        
        variance_data.append({
            'k_xy': k_xy,
            'var_xy': var_xy,
            'x': x_sim,
            'y': y_sim
        })
    
    # Store results
    results = {
        'k_x': k_x,
        'k_y': k_y,
        'k_z': k_z,
        't_max': t_max,
        'dt': dt,
        't': t,
        'x': x,
        'y': y,
        'z': z,
        't_lag': t_lag,
        'Cx': Cx,
        't_lag_msd': t_lag_msd,
        'msd': msd,
        'variance_data': variance_data
    }
    
    return results

def visualize_results(results, params, save_path=None):

    figures = []
    
    # Extract results
    k_x = results['k_x']
    k_y = results['k_y']
    k_z = results['k_z']
    t = results['t']
    x = results['x']
    y = results['y']
    z = results['z']
    t_lag = results['t_lag']
    Cx = results['Cx']
    t_lag_msd = results['t_lag_msd']
    msd = results['msd']
    variance_data = results['variance_data']
    
    # Figure 1: 3D trajectory and probability distributions
    fig1 = plt.figure(figsize=(15, 5))
    
    # Plot 3D trajectory
    ax1 = fig1.add_subplot(131, projection='3d')
    plot_3d_trajectory(ax1, x, y, z)
    
    # Add ellipsoid to represent equiprobability surface
    # Calculate standard deviations
    std_x = np.std(x)
    std_y = np.std(y)
    std_z = np.std(z)
    
    # Create meshgrid for ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ex = std_x * np.outer(np.cos(u), np.sin(v))
    ey = std_y * np.outer(np.sin(u), np.sin(v))
    ez = std_z * np.outer(np.ones_like(u), np.cos(v))
    
    # Plot ellipsoid
    ax1.plot_surface(ex, ey, ez, color='green', alpha=0.2)
    
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_title('(a) 3D Trajectory')
    
    # Plot probability distribution in z-plane
    ax2 = fig1.add_subplot(132)
    contour = plot_probability_distribution_2d(ax2, x, z)
    add_colorbar(fig1, contour, ax2)
    set_axis_properties(
        ax2,
        xlabel='X (nm)',
        ylabel='Z (nm)',
        title='(b) Probability Distribution in Z-plane'
    )
    
    # Plot probability distribution in y-plane
    ax3 = fig1.add_subplot(133)
    contour = plot_probability_distribution_2d(ax3, x, y)
    add_colorbar(fig1, contour, ax3)
    set_axis_properties(
        ax3,
        xlabel='X (nm)',
        ylabel='Y (nm)',
        title='(c) Probability Distribution in Y-plane'
    )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig1, 'optical_trap_3d_trajectory.png', save_path)
    
    figures.append(fig1)
    
    # Figure 2: Variance vs. trap stiffness and probability distributions
    fig2 = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig2)
    
    # Plot for variance vs. stiffness
    ax1 = fig2.add_subplot(gs[0, :])
    
    # Theoretical curve: variance ∝ 1/k
    k_theory = np.linspace(0.1, 6.0, 100)
    var_theory = KB * T / (k_theory * 1e-6) * 1e18  # Convert to nm²
    
    ax1.plot(k_theory, var_theory, 'k-', label='Theory: σ² ∝ 1/k')
    
    # Plot simulation results
    k_sim = [data['k_xy'] for data in variance_data]
    var_sim = [data['var_xy'] for data in variance_data]
    
    ax1.plot(k_sim, var_sim, 'ko', markersize=8, label='Simulation')
    
    set_axis_properties(
        ax1,
        xlabel='k_xy (fN/nm)',
        ylabel='σ²_xy (nm²)',
        title='(a) Variance vs. Trap Stiffness'
    )
    ax1.legend()
    
    # Plot probability distributions for different stiffness values
    for i, data in enumerate(variance_data):
        ax = fig2.add_subplot(gs[1, i])
        
        contour = plot_probability_distribution_2d(ax, data['x'], data['y'])
        add_colorbar(fig2, contour, ax)
        
        set_axis_properties(
            ax,
            xlabel='X (nm)',
            ylabel='Y (nm)',
            title=f'(b{i+1}) k_xy = {data["k_xy"]} fN/nm'
        )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig2, 'optical_trap_variance_stiffness.png', save_path)
    
    figures.append(fig2)
    
    # Figure 3: Position autocorrelation and mean square displacement
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot position autocorrelation
    ax1.plot(t_lag, Cx, 'k-')
    
    # Calculate characteristic time phi = c/k
    phi = C / (k_x * 1e-6)  # Convert k from fN/nm to N/m
    
    # Add vertical line at t = phi
    ax1.axvline(x=phi, color='k', linestyle='--', alpha=0.5)
    
    set_axis_properties(
        ax1,
        xlabel='t (s)',
        ylabel='C_x(t) (a.u.)',
        title='(a) Position Autocorrelation'
    )
    
    # Plot mean square displacement
    ax2.plot(t_lag_msd, msd, 'k-')
    
    # Add vertical line at t = phi
    ax2.axvline(x=phi, color='k', linestyle='--', alpha=0.5)
    
    set_axis_properties(
        ax2,
        xlabel='t (s)',
        ylabel='<x(t)²> (nm²)',
        title='(b) Mean Square Displacement'
    )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig3, 'optical_trap_autocorrelation_msd.png', save_path)
    
    figures.append(fig3)
    
    return figures
