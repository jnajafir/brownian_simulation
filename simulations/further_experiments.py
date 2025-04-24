#Further experiments simulation module.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import (
    KB, T, R, M, ETA, C, D,
    calculate_cross_correlation,
    setup_figure, save_figure, plot_trajectory, plot_histogram,
    set_axis_properties
)

def simulate_constant_force(k, Fc, t_max, dt):

    # Convert trap stiffness from fN/nm to N/m
    k = k * 1e-6  # fN/nm to N/m
    # Convert force from fN to N
    Fc = Fc * 1e-15  # fN to N
    
    # Calculate number of steps
    n_steps = int(t_max / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    
    # Generate Gaussian white noise
    w = np.random.normal(0, 1, n_steps)
    
    # Implement the Langevin equation with constant force
    for i in range(1, n_steps):
        # Apply constant force after t = t_max/2
        force = Fc if t[i] > t_max/2 else 0
        
        # Update position
        x[i] = x[i-1] - (k/C) * x[i-1] * dt + (force/C) * dt + np.sqrt(2*D*dt) * w[i]
    
    # Convert positions to nm
    x = x * 1e9
    
    return t, x

def simulate_rotational_force(k, Omega, t_max, dt):

    # Convert trap stiffness from fN/nm to N/m
    k = k * 1e-6  # fN/nm to N/m
    
    # Calculate number of steps
    n_steps = int(t_max / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    
    # Generate Gaussian white noise
    wx = np.random.normal(0, 1, n_steps)
    wy = np.random.normal(0, 1, n_steps)
    
    # Implement the Langevin equation with rotational force
    for i in range(1, n_steps):
        # Force components with rotation
        Fx = -k * x[i-1] + C * Omega * y[i-1]
        Fy = -k * y[i-1] - C * Omega * x[i-1]
        
        # Update positions
        x[i] = x[i-1] + (Fx/C) * dt + np.sqrt(2*D*dt) * wx[i]
        y[i] = y[i-1] + (Fy/C) * dt + np.sqrt(2*D*dt) * wy[i]
    
    # Convert positions to nm
    x = x * 1e9
    y = y * 1e9
    
    return t, x, y

def simulate_double_well(a, b, t_max, dt):
   
    # Calculate number of steps
    n_steps = int(t_max / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    
    # Start at one of the equilibrium positions
    x[0] = np.sqrt(b/a)  # in meters
    
    # Generate Gaussian white noise
    w = np.random.normal(0, 1, n_steps)
    
    # Use a smaller coefficient to avoid numerical instability
    a_scaled = 1.0e-7 * a  # Scale down by 10^7
    b_scaled = 1.0e-6 * b  # Scale down by 10^6
    
    # Implement the Langevin equation with double-well potential
    for i in range(1, n_steps):
        # Force from double-well potential: F(x) = -a*x^3 + b*x
        force = -a_scaled * (x[i-1]**3) + b_scaled * x[i-1]
        
        # Update position with force clamping to avoid instability
        force = np.clip(force, -1e-12, 1e-12)  # Limit force magnitude
        x[i] = x[i-1] + (force/C) * dt + np.sqrt(2*D*dt) * w[i]
    
    # Convert positions to nm
    x = x * 1e9
    
    return t, x

def run_simulation(params):

    # Extract parameters for constant force
    constant_force_params = params.get('constant_force', {})
    k_cf = constant_force_params.get('k', 1.0)  # trap stiffness in fN/nm
    Fc = constant_force_params.get('Fc', 200)  # constant force in fN
    t_max_cf = constant_force_params.get('t_max', 2.0)  # simulation time in seconds
    dt_cf = constant_force_params.get('dt', 0.001)  # time step in seconds
    
    # Simulate with constant force
    t_cf, x_cf = simulate_constant_force(k_cf, Fc, t_max_cf, dt_cf)
    
    # Extract parameters for rotational force
    rotational_force_params = params.get('rotational_force', {})
    k_rf = rotational_force_params.get('k', 1.0)  # trap stiffness in fN/nm
    Omega = rotational_force_params.get('Omega', 132.6)  # rotational component in rad/s
    t_max_rf = rotational_force_params.get('t_max', 0.1)  # simulation time in seconds
    dt_rf = rotational_force_params.get('dt', 0.0001)  # time step in seconds
    
    # Simulate with rotational force
    t_rf, x_rf, y_rf = simulate_rotational_force(k_rf, Omega, t_max_rf, dt_rf)
    
    # Calculate cross-correlation
    t_lag_rf, Cxy = calculate_cross_correlation(x_rf, y_rf, dt_rf)
    
    # Extract parameters for double-well potential
    double_well_params = params.get('double_well', {})
    a = double_well_params.get('a', 1.0e7)  # coefficient for x^4 term in N/m^3
    b = double_well_params.get('b', 1.0e6)  # coefficient for x^2 term in N/m
    t_max_dw = double_well_params.get('t_max', 10.0)  # simulation time in seconds
    dt_dw = double_well_params.get('dt', 0.001)  # time step in seconds
    
    # Simulate with double-well potential
    t_dw, x_dw = simulate_double_well(a, b, t_max_dw, dt_dw)
    
    # Store results
    results = {
        'constant_force': {
            'k': k_cf,
            'Fc': Fc,
            't_max': t_max_cf,
            'dt': dt_cf,
            't': t_cf,
            'x': x_cf
        },
        'rotational_force': {
            'k': k_rf,
            'Omega': Omega,
            't_max': t_max_rf,
            'dt': dt_rf,
            't': t_rf,
            'x': x_rf,
            'y': y_rf,
            't_lag': t_lag_rf,
            'Cxy': Cxy
        },
        'double_well': {
            'a': a,
            'b': b,
            't_max': t_max_dw,
            'dt': dt_dw,
            't': t_dw,
            'x': x_dw
        }
    }
    
    return results

def visualize_results(results, params, save_path=None):

    # Extract results
    # Constant force
    t_cf = results['constant_force']['t']
    x_cf = results['constant_force']['x']
    k_cf = results['constant_force']['k']
    Fc = results['constant_force']['Fc']
    t_max_cf = results['constant_force']['t_max']
    
    # Rotational force
    t_rf = results['rotational_force']['t']
    x_rf = results['rotational_force']['x']
    y_rf = results['rotational_force']['y']
    t_lag_rf = results['rotational_force']['t_lag']
    Cxy = results['rotational_force']['Cxy']
    Omega = results['rotational_force']['Omega']
    
    # Double-well potential
    t_dw = results['double_well']['t']
    x_dw = results['double_well']['x']
    a = results['double_well']['a']
    b = results['double_well']['b']
    
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Part (a): Constant force
    # Calculate histograms for before and after force application
    before_idx = t_cf < t_max_cf/2
    after_idx = t_cf >= t_max_cf/2
    
    hist_before, bins_before = np.histogram(x_cf[before_idx], bins=30, density=True)
    hist_after, bins_after = np.histogram(x_cf[after_idx], bins=30, density=True)
    
    # Plot histograms
    bin_centers_before = (bins_before[:-1] + bins_before[1:]) / 2
    bin_centers_after = (bins_after[:-1] + bins_after[1:]) / 2
    
    ax1.bar(bin_centers_before, hist_before, width=(bins_before[1]-bins_before[0]), 
            alpha=0.5, color='k', label='Before force')
    ax1.bar(bin_centers_after, hist_after, width=(bins_after[1]-bins_after[0]), 
            alpha=0.5, color='gray', label='After force')
    
    # Calculate theoretical distributions
    # Before force: centered at 0
    # After force: centered at Fc/k
    x_theory = np.linspace(min(bins_before[0], bins_after[0]), 
                           max(bins_before[-1], bins_after[-1]), 1000)
    
    # Standard deviation from equipartition theorem: σ² = kB*T/k
    sigma = np.sqrt(KB * T / (k_cf * 1e-6)) * 1e9  # in nm
    
    # Theoretical distributions
    pdf_before = norm.pdf(x_theory, 0, sigma)
    pdf_after = norm.pdf(x_theory, Fc/k_cf, sigma)  # Centered at Fc/k
    
    ax1.plot(x_theory, pdf_before, 'k-', linewidth=2)
    ax1.plot(x_theory, pdf_after, 'k--', linewidth=2)
    
    set_axis_properties(
        ax1,
        xlabel='x (nm)',
        ylabel='Probability density',
        title='(a) Constant Force'
    )
    ax1.legend()
    
    # Part (b): Rotational force
    # Plot cross-correlation
    t_lag_ms = t_lag_rf * 1000  # Convert to ms
    ax2.plot(t_lag_ms, Cxy, 'b-', label='Cxy(t)')
    
    # Plot theoretical cross-correlation (sinusoidal)
    t_theory = np.linspace(0, max(t_lag_rf), 1000)
    Cxy_theory = np.sin(Omega * t_theory)
    ax2.plot(t_theory * 1000, Cxy_theory, 'k--', label='Theory')
    
    set_axis_properties(
        ax2,
        xlabel='t (ms)',
        ylabel='Cxy(t) (a.u.)',
        title='(b) Rotational Force',
        xlim=(-200, 200),
        ylim=(-1, 1)
    )
    ax2.legend()
    
    # Inset: trajectory
    inset = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    inset.plot(x_rf, y_rf, 'k-', linewidth=0.5)
    inset.set_xlabel('x (nm)')
    inset.set_ylabel('y (nm)')
    inset.grid(True)
    
    # Part (c): Double-well potential
    # Plot trajectory
    ax3.plot(t_dw, x_dw, 'k-')
    
    # Add horizontal lines at the two equilibrium positions
    eq_pos = np.sqrt(b/a) * 1e9  # in nm
    ax3.axhline(y=eq_pos, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=-eq_pos, color='k', linestyle='--', alpha=0.5)
    
    set_axis_properties(
        ax3,
        xlabel='t (s)',
        ylabel='x (nm)',
        title='(c) Double-Well Potential',
        xlim=(0, 200),
        ylim=(-500, 500)
    )
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_figure(fig, 'further_numerical_experiments.png', save_path)
    
    return fig
