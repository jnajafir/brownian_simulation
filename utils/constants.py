#Physical constants and particle parameters for Brownian particle simulation.

# Physical constants
KB = 1.38e-23  # Boltzmann constant in J/K
T = 300  # Temperature in Kelvin

# Particle parameters
R = 1e-6  # Particle radius in m
M = 11e-15  # Mass in kg
ETA = 1e-3  # Viscosity in N·s/m²
C = 6 * 3.14159265359 * ETA * R  # Friction coefficient
TAU = M / C  # Momentum relaxation time (inertial time)
D = KB * T / C  # Diffusion coefficient

def get_particle_info():
    info = [
        f"Particle parameters:",
        f"Radius: {R*1e6:.1f} µm",
        f"Mass: {M*1e12:.1f} pg",
        f"Viscosity: {ETA:.3f} N·s/m²",
        f"Friction coefficient: {C:.3e} N·s/m",
        f"Temperature: {T} K",
        f"Momentum relaxation time (tau): {TAU*1e6:.1f} µs",
        f"Diffusion coefficient: {D*1e12:.3f} µm²/s"
    ]
    return "\n".join(info)
