#Visualization 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os

def setup_figure(nrows=1, ncols=1, figsize=(10, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes

def save_figure(fig, filename, output_dir="output", dpi=300):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to save figure
    filepath = os.path.join(output_dir, filename)
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    return filepath

def plot_trajectory(ax, t, x, label=None, color='k', linestyle='-', alpha=1.0):
    
    line = ax.plot(t, x, color=color, linestyle=linestyle, label=label, alpha=alpha)[0]
    return line

def plot_histogram(ax, data, bins=30, density=True, color='k', alpha=0.5, label=None):

    n, bins, patches = ax.hist(data, bins=bins, density=density, color=color, alpha=alpha, label=label)
    return n, bins, patches

def plot_3d_trajectory(ax, x, y, z, color='blue', linewidth=0.5, alpha=0.7, label='Trajectory'):

    line = ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha, label=label)[0]
    
    # Mark start point (green)
    ax.scatter(x[0], y[0], z[0], color='green', s=30, label='Start')
    
    # Mark end point (red)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=30, label='End')
    
    return line

def plot_probability_distribution_2d(ax, x, y, bins=50, cmap='viridis'):
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    
    # Create meshgrid for contour plot
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
    
    # Plot contour
    contour = ax.contourf(X, Y, hist.T, cmap=cmap)
    
    return contour

def add_colorbar(fig, contour, ax, label='Probability density'):
 
    cbar = fig.colorbar(contour, ax=ax, label=label)
    return cbar

def set_axis_properties(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True)
