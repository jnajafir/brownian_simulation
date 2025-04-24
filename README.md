# Brownian Particle Simulation

A implementation of the Brownian particle simulation based on the paper "Simulation of a Brownian particle in an optical trap" by Giorgio Volpe and Giovanni Volpe.

## Features

- White noise simulation
- Ballistic to Brownian diffusion simulation
- Optical traps simulation in 3D
- Further numerical experiments (constant force, rotational force, double-well potential)
- Double-well potential simulation with Kramers transitions

## Installation

1. Clone the repository:
```
git clone https://github.com/jnajafir/brownian_simulation.git
cd brownian_simulation
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

The simulation can be run using the command-line interface:

```
python main.py 
```

### Options

- `--simulation`: Specify which simulation to run (`white_noise`, `ballistic_brownian`, `optical_traps`, `further_experiments`, `double_well`, or `all`)
- `--config`: Path to a custom configuration file (JSON or YAML)
- `--output-dir`: Directory to save output files
- `--save-plots`: Save plots to the output directory
- `--no-show-plots`: Do not display plots
- `--random-seed`: Set random seed for reproducibility

### Examples

Run all simulations with default parameters:
```
python main.py
```

Run only the white noise simulation:
```
python main.py --simulation white_noise
```

Run the optical traps simulation with custom parameters:
```
python main.py --simulation optical_traps --ot-k-x 2.0 --ot-k-y 2.0 --ot-k-z 0.5
```

Save plots without displaying them:
```
python main.py --save-plots --no-show-plots --output-dir results
```

## Project Structure

- `main.py`: Main entry point with CLI interface
- `config/`: Configuration management
  - `default_config.py`: Default configuration values
  - `config_loader.py`: Configuration loading utilities
- `utils/`: Common utilities
  - `constants.py`: Physical constants
  - `statistics.py`: Statistical analysis functions
  - `visualization.py`: Common plotting functions
- `simulations/`: Simulation implementations
  - `white_noise.py`: White noise simulation (Section I)
  - `ballistic_brownian.py`: Ballistic to Brownian diffusion (Section III)
  - `optical_traps.py`: Optical traps simulation (Section IV)
  - `further_experiments.py`: Further numerical experiments (Section V)
  - `double_well.py`: Double-well potential simulation

## Configuration

The simulation parameters can be configured through:
1. Default values in `config/default_config.py`
2. Custom configuration file (JSON or YAML)
3. Command-line arguments

## Acknowledgments

- Giorgio Volpe and Giovanni Volpe for the original paper and simulations
- The American Journal of Physics for publishing the paper
