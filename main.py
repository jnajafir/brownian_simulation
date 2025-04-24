import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from config import load_config, update_config_from_args
from utils import get_particle_info
from simulations import (
    run_white_noise, visualize_white_noise,
    run_ballistic_brownian, visualize_ballistic_brownian,
    run_optical_traps, visualize_optical_traps,
    run_further_experiments, visualize_further_experiments,
    run_double_well, visualize_double_well
)

def parse_arguments():

    parser = argparse.ArgumentParser(
        description='Brownian Particle Simulation based on the paper by Giorgio Volpe and Giovanni Volpe'
    )
    
    # General arguments
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to output directory')
    parser.add_argument('--no-show-plots', action='store_true', help='Do not display plots')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    # Simulation selection
    parser.add_argument('--simulation', type=str, choices=[
        'white_noise', 'ballistic_brownian', 'optical_traps', 
        'further_experiments', 'double_well', 'all'
    ], default='all', help='Simulation to run')
    
    # White noise simulation arguments
    white_noise_group = parser.add_argument_group('White Noise Simulation')
    white_noise_group.add_argument('--wn-dt-values', type=float, nargs='+', 
                                  help='Time step values for white noise simulation')
    white_noise_group.add_argument('--wn-t-max', type=float, 
                                  help='Maximum simulation time for white noise')
    white_noise_group.add_argument('--wn-n-trajectories', type=int, 
                                  help='Number of trajectories for statistical analysis')
    
    # Ballistic to Brownian simulation arguments
    ballistic_group = parser.add_argument_group('Ballistic to Brownian Simulation')
    ballistic_group.add_argument('--bb-dt', type=float, 
                                help='Time step for ballistic to Brownian simulation')
    ballistic_group.add_argument('--bb-t-max-factor', type=float, 
                                help='Maximum simulation time as a factor of tau')
    
    # Optical traps simulation arguments
    optical_group = parser.add_argument_group('Optical Traps Simulation')
    optical_group.add_argument('--ot-k-x', type=float, 
                              help='Trap stiffness in x-direction (fN/nm)')
    optical_group.add_argument('--ot-k-y', type=float, 
                              help='Trap stiffness in y-direction (fN/nm)')
    optical_group.add_argument('--ot-k-z', type=float, 
                              help='Trap stiffness in z-direction (fN/nm)')
    optical_group.add_argument('--ot-t-max', type=float, 
                              help='Maximum simulation time for optical traps')
    optical_group.add_argument('--ot-dt', type=float, 
                              help='Time step for optical traps simulation')
    
    # Further experiments simulation arguments
    further_group = parser.add_argument_group('Further Experiments Simulation')
    further_group.add_argument('--fe-cf-k', type=float, 
                              help='Trap stiffness for constant force experiment (fN/nm)')
    further_group.add_argument('--fe-cf-force', type=float, 
                              help='Constant force value (fN)')
    further_group.add_argument('--fe-rf-k', type=float, 
                              help='Trap stiffness for rotational force experiment (fN/nm)')
    further_group.add_argument('--fe-rf-omega', type=float, 
                              help='Rotational component (rad/s)')
    further_group.add_argument('--fe-dw-a', type=float, 
                              help='Coefficient for x^4 term in double-well potential (N/m^3)')
    further_group.add_argument('--fe-dw-b', type=float, 
                              help='Coefficient for x^2 term in double-well potential (N/m)')
    
    # Double well simulation arguments
    double_well_group = parser.add_argument_group('Double Well Simulation')
    double_well_group.add_argument('--dw-a', type=float, 
                                  help='Coefficient for x^4 term (N/m^3)')
    double_well_group.add_argument('--dw-b', type=float, 
                                  help='Coefficient for x^2 term (N/m)')
    double_well_group.add_argument('--dw-gamma', type=float, 
                                  help='Drag coefficient (kg/s)')
    double_well_group.add_argument('--dw-dt', type=float, 
                                  help='Time step for double well simulation')
    double_well_group.add_argument('--dw-n-steps', type=int, 
                                  help='Number of steps for double well simulation')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """
    Update configuration with command-line arguments.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    # Update general configuration
    if args.output_dir:
        config['general']['output_dir'] = args.output_dir
    
    if args.save_plots:
        config['general']['save_plots'] = True
    
    if args.no_show_plots:
        config['general']['show_plots'] = False
    
    if args.random_seed is not None:
        config['general']['random_seed'] = args.random_seed
    
    # Update white noise configuration
    if args.wn_dt_values:
        config['white_noise']['dt_values'] = args.wn_dt_values
    
    if args.wn_t_max:
        config['white_noise']['t_max'] = args.wn_t_max
    
    if args.wn_n_trajectories:
        config['white_noise']['n_trajectories'] = args.wn_n_trajectories
    
    # Update ballistic to Brownian configuration
    if args.bb_dt:
        config['ballistic_brownian']['dt'] = args.bb_dt
    
    if args.bb_t_max_factor:
        config['ballistic_brownian']['t_max_factor'] = args.bb_t_max_factor
    
    # Update optical traps configuration
    if args.ot_k_x:
        config['optical_traps']['k_x'] = args.ot_k_x
    
    if args.ot_k_y:
        config['optical_traps']['k_y'] = args.ot_k_y
    
    if args.ot_k_z:
        config['optical_traps']['k_z'] = args.ot_k_z
    
    if args.ot_t_max:
        config['optical_traps']['t_max'] = args.ot_t_max
    
    if args.ot_dt:
        config['optical_traps']['dt'] = args.ot_dt
    
    # Update further experiments configuration
    if args.fe_cf_k:
        config['further_experiments']['constant_force']['k'] = args.fe_cf_k
    
    if args.fe_cf_force:
        config['further_experiments']['constant_force']['Fc'] = args.fe_cf_force
    
    if args.fe_rf_k:
        config['further_experiments']['rotational_force']['k'] = args.fe_rf_k
    
    if args.fe_rf_omega:
        config['further_experiments']['rotational_force']['Omega'] = args.fe_rf_omega
    
    if args.fe_dw_a:
        config['further_experiments']['double_well']['a'] = args.fe_dw_a
    
    if args.fe_dw_b:
        config['further_experiments']['double_well']['b'] = args.fe_dw_b
    
    # Update double well configuration
    if args.dw_a:
        config['double_well']['a'] = args.dw_a
    
    if args.dw_b:
        config['double_well']['b'] = args.dw_b
    
    if args.dw_gamma:
        config['double_well']['gamma'] = args.dw_gamma
    
    if args.dw_dt:
        config['double_well']['dt'] = args.dw_dt
    
    if args.dw_n_steps:
        config['double_well']['n_steps'] = args.dw_n_steps
    
    return config

def run_white_noise_simulation(config):
    """
    Run white noise simulation.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    """
    print("\n=== Running White Noise Simulation ===")
    results = run_white_noise(config['white_noise'])
    fig = visualize_white_noise(results, config['white_noise'])
    
    if config['general']['save_plots']:
        output_dir = config['general']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'white_noise_simulation.png'), dpi=300)
        print(f"Figure saved to {os.path.join(output_dir, 'white_noise_simulation.png')}")
    
    if config['general']['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def run_ballistic_brownian_simulation(config):

    print("\n=== Running Ballistic to Brownian Simulation ===")
    results = run_ballistic_brownian(config['ballistic_brownian'])
    fig = visualize_ballistic_brownian(results, config['ballistic_brownian'])
    
    if config['general']['save_plots']:
        output_dir = config['general']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'ballistic_to_brownian.png'), dpi=300)
        print(f"Figure saved to {os.path.join(output_dir, 'ballistic_to_brownian.png')}")
    
    if config['general']['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def run_optical_traps_simulation(config):

    print("\n=== Running Optical Traps Simulation ===")
    results = run_optical_traps(config['optical_traps'])
    figures = visualize_optical_traps(results, config['optical_traps'])
    
    if config['general']['save_plots']:
        output_dir = config['general']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        filenames = [
            'optical_trap_3d_trajectory.png',
            'optical_trap_variance_stiffness.png',
            'optical_trap_autocorrelation_msd.png'
        ]
        
        for fig, filename in zip(figures, filenames):
            fig.savefig(os.path.join(output_dir, filename), dpi=300)
            print(f"Figure saved to {os.path.join(output_dir, filename)}")
    
    if config['general']['show_plots']:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

def run_further_experiments_simulation(config):

    print("\n=== Running Further Experiments Simulation ===")
    results = run_further_experiments(config['further_experiments'])
    fig = visualize_further_experiments(results, config['further_experiments'])
    
    if config['general']['save_plots']:
        output_dir = config['general']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'further_numerical_experiments.png'), dpi=300)
        print(f"Figure saved to {os.path.join(output_dir, 'further_numerical_experiments.png')}")
    
    if config['general']['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def run_double_well_simulation(config):

    print("\n=== Running Double Well Simulation ===")
    results = run_double_well(config['double_well'])
    fig = visualize_double_well(results, config['double_well'])
    
    if config['general']['save_plots']:
        output_dir = config['general']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'double_well_potential.png'), dpi=300)
        print(f"Figure saved to {os.path.join(output_dir, 'double_well_potential.png')}")
    
    if config['general']['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def main():
    # Main function to run the simulations.
   
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command-line arguments
    config = update_config_with_args(config, args)
    
    # Set random seed
    np.random.seed(config['general']['random_seed'])
    
    # Print particle information
    print(get_particle_info())
    
    # Create output directory if it doesn't exist
    os.makedirs(config['general']['output_dir'], exist_ok=True)
    
    # Run selected simulation(s)
    if args.simulation == 'all' or args.simulation == 'white_noise':
        run_white_noise_simulation(config)
    
    if args.simulation == 'all' or args.simulation == 'ballistic_brownian':
        run_ballistic_brownian_simulation(config)
    
    if args.simulation == 'all' or args.simulation == 'optical_traps':
        run_optical_traps_simulation(config)
    
    if args.simulation == 'all' or args.simulation == 'further_experiments':
        run_further_experiments_simulation(config)
    
    if args.simulation == 'all' or args.simulation == 'double_well':
        run_double_well_simulation(config)
    
    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    main()
