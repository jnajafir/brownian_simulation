#Statistical analysis utilities for Brownian particle simulation.


import numpy as np

def calculate_position_autocorrelation(x, dt, max_lag=None):

    if max_lag is None:
        max_lag = len(x) // 10  # Use 10% of the data points by default
    
    Cx = np.zeros(max_lag)
    
    # Calculate autocorrelation for each lag
    for lag in range(max_lag):
        # Ensure we don't go out of bounds
        if lag < len(x):
            # Calculate correlation between x(t) and x(t+lag)
            Cx[lag] = np.mean(x[:-lag] * x[lag:]) if lag > 0 else np.mean(x * x)
    
    # Normalize by the zero-lag value
    Cx = Cx / Cx[0]
    
    # Create time lag array
    t_lag = np.arange(max_lag) * dt
    
    return t_lag, Cx

def calculate_velocity_autocorrelation(v, dt, max_lag=None):
    if max_lag is None:
        max_lag = len(v) // 10  # Use 10% of the data points by default
    
    Cv = np.zeros(max_lag)
    
    # Calculate autocorrelation for each lag
    for lag in range(max_lag):
        # Ensure we don't go out of bounds
        if lag < len(v):
            # Calculate correlation between v(t) and v(t+lag)
            Cv[lag] = np.mean(v[:-lag] * v[lag:]) if lag > 0 else np.mean(v * v)
    
    # Normalize by the zero-lag value
    Cv = Cv / Cv[0]
    
    # Create time lag array
    t_lag = np.arange(max_lag) * dt
    
    return t_lag, Cv

def calculate_cross_correlation(x, y, dt, max_lag=None):
   
    if max_lag is None:
        max_lag = len(x) // 10  # Use 10% of the data points by default
    
    Cxy = np.zeros(max_lag)
    
    # Calculate cross-correlation for each lag
    for lag in range(max_lag):
        # Ensure we don't go out of bounds
        if lag < len(x):
            # Calculate correlation between x(t) and y(t+lag)
            Cxy[lag] = np.mean(x[:-lag] * y[lag:]) if lag > 0 else np.mean(x * y)
    
    # Normalize
    Cxy = Cxy / np.sqrt(np.mean(x**2) * np.mean(y**2))
    
    # Create time lag array
    t_lag = np.arange(max_lag) * dt
    
    return t_lag, Cxy

def calculate_mean_square_displacement(x, dt, max_lag=None):
   
    if max_lag is None:
        max_lag = len(x) // 10  # Use 10% of the data points by default
    
    msd = np.zeros(max_lag)
    
    # Calculate MSD for each lag
    for lag in range(max_lag):
        # Ensure we don't go out of bounds
        if lag < len(x):
            # Calculate mean square displacement
            msd[lag] = np.mean((x[lag:] - x[:-lag])**2) if lag > 0 else 0
    
    # Create time lag array
    t_lag = np.arange(max_lag) * dt
    
    return t_lag, msd
