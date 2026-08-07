import mesa_reader as mr
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
# from matplotlib.animation import FuncAnimation
import glob
import pandas as pd

# Important Custom Functions:
from betelbuddy_func_lib import get_history, print_first_file_columns, calc_filter_apparent_mag, add_filter_apparent_column, load_isochrone_data, build_isochrone_multi, plot_four_panel

# My usual plotting formatting:
# 1. Font Style: Serif and LaTeX-like math formatting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # Gives that classic LaTeX look for equations
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# 2. Tick Style: Inward pointing, on all sides, with minor ticks
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True       # Ticks on the top edge
plt.rcParams['ytick.right'] = True     # Ticks on the right edge
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# 3. Line and Tick Thickness: Bolder for print readability
plt.rcParams['axes.linewidth'] = 1.5   # Bounding box thickness
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.major.size'] = 6   # Length of major ticks
plt.rcParams['xtick.minor.size'] = 3   # Length of minor ticks
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['lines.linewidth'] = 2.0  # Thickness of your plotted lines


# IMPORTANT!!! CHANGE PATH TO WHEREVER YOU'RE STORING DR DENNIS' TRACKS
directories = ["/mnt/d/mesa_storage/betelbuddy/history_files_Bband/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_alopeke/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_zimpol/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_stis_fuv/history_m*.data",]

# Load the data
dataframes = load_isochrone_data(directories)

# DEFINE FILTERS / AXES LABELS, ORDER MATTERS
filters = [
    {'x_var': 'B', 'y_var': 'Mag_app', 'x_title': r'Johnson B-Band', 'y_title': r'$M_{\text{app}}$', 'invert_y': True},
    {'x_var': 'EO_466', 'y_var': 'Mag_app', 'x_title': r'Gemini Alopeke F466', 'y_title': r'$M_{\text{app}}$', 'invert_y': True},
    {'x_var': 'Cnt_Ha', 'y_var': 'Mag_app', 'x_title': r'VLTI Cnt_Ha', 'y_title': r'$M_{\text{app}}$', 'invert_y': True},
    {'x_var': '25MAMA', 'y_var': 'Mag_app', 'x_title': r'HST FUV-STIS', 'y_title': r'$M_{\text{app}}$', 'invert_y': True}
]

# DEFINE TARGET AGES AND MASSES FOR ISOCHRONES AND MASSOCHRONES
target_ages = [5e6, 10e6, 15e6]
target_masses = [1.5,2.0,3.0,4.0]

# CHANGE TO YOUR DIRECTORY THAT YOU WOULD LIKE TO SAVE FIGURES TO
savedir = "figures/"

# NAME IT HOW YA LIKE
savename = "four_buddies_grid.png"

# Plot, show, and save that bad boy
plot_four_panel(dataframes, filters, target_ages=target_ages, target_masses=target_masses, savename=savename, savedir=savedir)