"""
Configuration loader module for Brownian particle simulation. This module provides functions to load and manage configuration parameters
"""

import json
import os
import yaml
from .default_config import DEFAULT_CONFIG

def load_config(config_path=None):
    """
    Load configuration from file and merge with default configuration.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file (JSON or YAML format)
        
    Returns:
    --------
    dict
        Configuration dictionary with all parameters
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # If config file is provided, load and merge with defaults
    if config_path and os.path.exists(config_path):
        file_extension = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_extension == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif file_extension in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_extension}")
                
            # Merge user config with default config
            _merge_configs(config, user_config)
            
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration instead.")
    
    return config

def _merge_configs(default_config, user_config):

    for key, value in user_config.items():
        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
            # If both are dictionaries, merge recursively
            _merge_configs(default_config[key], value)
        else:
            # Otherwise, override default value
            default_config[key] = value

def update_config_from_args(config, args):
    """
    Update configuration with command-line arguments.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary to update
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    # Update general configuration
    if hasattr(args, 'output_dir') and args.output_dir:
        config['general']['output_dir'] = args.output_dir
    
    if hasattr(args, 'save_plots'):
        config['general']['save_plots'] = args.save_plots
    
    if hasattr(args, 'no_show_plots'):
        config['general']['show_plots'] = not args.no_show_plots
    
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        config['general']['random_seed'] = args.random_seed
    
    # Update simulation-specific configurations based on args
    # This would be expanded based on the specific arguments for each simulation
    
    return config
