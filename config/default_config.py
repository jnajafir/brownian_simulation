"""
Default configuration values for Brownian particle simulation. This module defines the default configuration parameters for all simulations.
"""

DEFAULT_CONFIG = {
    "general": {
        "random_seed": 42,
        "output_dir": "output",
        "save_plots": True,
        "show_plots": True,
    },
    "white_noise": {
        "dt_values": [0.05, 0.1, 0.5, 1.0],
        "t_max": 20.0,
        "n_trajectories": 10000,
    },
    "ballistic_brownian": {
        "dt": 1e-9,  # Time step much smaller than tau
        "t_max_factor": 10,  # Simulate for 10 times the momentum relaxation time
    },
    "optical_traps": {
        "k_x": 1.0,  # trap stiffness in x-direction in fN/nm
        "k_y": 1.0,  # trap stiffness in y-direction in fN/nm
        "k_z": 0.2,  # trap stiffness in z-direction in fN/nm
        "t_max": 200.0,  # simulation time in seconds
        "dt": 0.001,  # time step in seconds
    },
    "further_experiments": {
        "constant_force": {
            "k": 1.0,  # trap stiffness in fN/nm
            "Fc": 200,  # constant force in fN
            "t_max": 2.0,  # simulation time in seconds
            "dt": 0.001,  # time step in seconds
        },
        "rotational_force": {
            "k": 1.0,  # trap stiffness in fN/nm
            "Omega": 132.6,  # rotational component in rad/s
            "t_max": 0.1,  # simulation time in seconds
            "dt": 0.0001,  # time step in seconds
        },
        "double_well": {
            "a": 1.0e7,  # coefficient for x^4 term in N/m^3
            "b": 1.0e6,  # coefficient for x^2 term in N/m
            "t_max": 10.0,  # simulation time in seconds
            "dt": 0.001,  # time step in seconds
        },
    },
    "double_well": {
        "a": 1.0e7,  # coefficient for x^4 term in N/m^3
        "b": 1.0e-6,  # coefficient for x^2 term in N/m
        "gamma": 1e-8,  # kg/s (typical nanoscale drag)
        "dt": 1e-4,  # time step in seconds
        "n_steps": 2000000,  # number of steps
    },
}
