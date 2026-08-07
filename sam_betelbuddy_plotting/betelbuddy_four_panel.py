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


# Quick aside: Calculate Absolute magnitudes in specific filter bands (for now just Montarges, can add HST stuff here)

def calc_abs_mag_from_flux(flux_density, filter_width, distance_pc, zp_flux):
    # Integrate the flux density over the filter width 
    # total_flux = flux_density * filter_width		# SDB well we found out this is irrelevant, but it's here if we need to add back later
    
    # Calculate the apparent magnitude
    app_mag = -2.5 * np.log10(flux_density / zp_flux)
    # print(app_mag)
    
    # Apply the distance modulus to find absolute magnitude
    abs_mag = app_mag - 5 * np.log10(distance_pc) + 5
    
    return abs_mag


Cnt_Ha_ZP = 2.18842e-9 # SVO, erg cm^-2 Angstrom^-1
distance_pc = 168 # parsec, Joyce 2020
filt_width = 40.92 # Angstrom, SVO FWHM / Montarges width 
montarges_fluxes = [3e-11, 3.88e-11] # W m^-2 um^-1, Bounds from paper in Section 4.2
montarges_mags = []
for montarges_flux in montarges_fluxes:
    montarges_flux = montarges_flux*0.1  # Converting from W m^-2 um^-1 to erg s^-1 cm^-2 Angstrom^-1
    montarges_mag = calc_abs_mag_from_flux(montarges_flux, filt_width, 168, Cnt_Ha_ZP)
    # print(montarges_mag)
    montarges_mags.append(montarges_mag)



# --- NOW ONTO SOME PLOTTING ---


# IMPORTANT!!! CHANGE PATH TO WHEREVER YOU'RE STORING DR DENNIS' TRACKS
directories = ["/mnt/d/mesa_storage/betelbuddy/history_files_Bband/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_alopeke/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_zimpol/history_m*.data",
               "/mnt/d/mesa_storage/betelbuddy/history_files_stis_fuv/history_m*.data",]

# Load the data
dataframes = load_isochrone_data(directories)


# --- DEFINE FILTERS / AXES LABELS, ORDER MATTERS ---
# Zoom Keys:
# inset_pos: [left, bottom, width, height]
# zoom_lims: [x_start, x_end, y_start, y_end] (in graph coordinates)

# Change variables as you like, I hope it doesn't break (it really shouldn't). The 'Mag_bol_filter' column (frequently a y-axis variable) is a column created by the script
# 	when you give it a valid 'filter'

filters = [
    {'filter':None, 'x_var': 'log_Teff', 'y_var': 'M_bol', 'x_title': r'$\log T_{eff}$', 'y_title': r'$M_{\text{bol}}$', 
         'invert_x':True, 'invert_y': True,},
    {'filter':'EO_466', 'x_var': 'log_Teff', 'y_var': 'Mag_bol_filter', 'x_title': r'$\log T_{eff}$', 'y_title': r'$M_{\text{abs}}$ Gemini Alopeke F466', 
         'invert_x':True, 'invert_y': True,
        # 'inset_pos': [0.45, 0.55, 0.50, 0.4], # Top-right corner of the panel
        # 'zoom_lims': [9000, 7000, 18.95, 18.85]    # The specific data bounds to zoom into
        # Shade box for parameters from other studies (feel free to add one of these blocks to the HST plot too, it should work exactly the same):
        'shades': [
            {
                'x_lims': None,        # Teff constraints
                'y_lims': [3.4, 1.4],            # Magnitude constraints, straight from Howell et al 2025 Section 4.2
                'label': 'Howell+ 2025', 
                'color': 'red', 
                'alpha': 0.6
            }
        ]
    },
    {'filter':'Cnt_Ha','x_var': 'log_Teff', 'y_var': 'Mag_bol_filter', 'x_title': r'$\log T_{eff}$', 'y_title': r'$M_{\text{abs}}$ SPHERE Cnt_Ha', 
         'invert_x':True, 'invert_y': True,
        'shades': [
            {
                'x_lims': None,        # Teff constraints
                'y_lims': montarges_mags,            # Magnitude constraints (calculated earlier)
                'label': 'Montarges+ 2026', 
                'color': 'red', 
                'alpha': 0.6
            }
        ]
    },
    # NOTE THE FILTER USED HERE!! YOU MIGHT WANT TO CHANGE
    {'filter':'25MAMA','x_var': 'log_Teff', 'y_var': 'Mag_bol_filter', 'x_title': r'$\log T_{eff}$', 'y_title': r'$M_{\text{abs}}$ HST FUV-STIS', 
         'invert_x':True, 'invert_y': True}
]

# DEFINE TARGET AGES AND MASSES FOR ISOCHRONES AND MASSOCHRONES
target_ages = [5e6, 10e6, 15e6]
step=0.25
target_masses = np.arange(1.5,5.0+step,step)

# CHANGE TO YOUR DIRECTORY THAT YOU WOULD LIKE TO SAVE FIGURES TO
savedir = "figures/"

# NAME IT HOW YA LIKE
savename = "four_buddies_grid.png"

# Plot, show, and save that bad boy
plot_four_panel(dataframes, filters, target_ages=target_ages, 
                target_masses=target_masses, 
                savename=savename, savedir=savedir)