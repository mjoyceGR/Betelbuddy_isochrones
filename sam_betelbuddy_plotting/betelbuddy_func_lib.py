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
# import lightkurve as lk
# from astropy.timeseries import LombScargle
# from scipy.signal import find_peaks
# from scipy.optimize import linear_sum_assignment, minimize_scalar
# from scipy.interpolate import CubicSpline
# from scipy.odr import ODR, Model, RealData
# from scipy.optimize import curve_fit

def get_history(files):
    """
    Reads MESA history files and compiles them into a Pandas DataFrame.
    Each cell contains a full 1D NumPy array of the star's evolution over time.
    """
    rows = []
    
    for file in files:
        try:
            # Attempt to read history file
            h = mr.MesaData(file)
        except TypeError:
            print(f"Empty history file {file}")
            continue
            
        # Start an empty dictionary for this specific star's row
        row_dict = {}
        
        # Dynamically loop through every column header in the MESA file and pull the corresponding array of data
        for col_name in h.bulk_names:
            row_dict[col_name] = np.array(h.data(col_name))
            
        # Safely extract the star's mass array from MESA directly.
        # We save it under the 'Mass' key to keep Sam's dumb code running
        if 'star_mass' in h.bulk_names:
            row_dict['Mass'] = np.array(h.data('star_mass'))
        else:
            print(f"Warning: 'star_mass' missing in {file}")
            row_dict['Mass'] = np.array([])

            
        # Append this star's completed dictionary to the main list
        rows.append(row_dict)
        
    # Compile all dictionaries into a single DataFrame at once
    df_final = pd.DataFrame(rows)
    
    return df_final


def print_first_file_columns(directories):
    """
    Finds the first MESA history file in each directory pattern 
    and prints all available column headers. I used this to select filters
    """
    for pattern in directories:
        print(f"--- Scanning pattern: {pattern} ---")
        
        # Grab all files matching the pattern
        files = glob.glob(pattern)
        
        if not files:
            print("No files found matching this pattern.\n")
            continue
            
        # Isolate the first file in the returned list
        first_file = files[0]
        print(f"Reading file: {first_file}")
        
        try:
            # Load the data using MesaReader
            h = mr.MesaData(first_file)
            
            # Extract the column names using the bulk_names attribute
            columns = h.bulk_names
            
            print(f"Total columns found: {len(columns)}")
            print(columns)
            print("\n")
            
        except TypeError:
            print(f"File {first_file} appears to be empty or corrupted.\n")
        except Exception as e:
            print(f"An error occurred reading {first_file}: {e}\n")


def calc_filter_apparent_mag(M_bol, BC_filter, distance_pc):
    """
    Calculates the apparent magnitude in a specific filter using a Bolometric Correction.
    Assumes the convention: BC = M_bol - M_filter.
    
    Parameters:
    M_bol (float or np.ndarray): Absolute bolometric magnitude(s).
    BC_filter (float or np.ndarray): Bolometric correction value(s) for the specific filter.
    distance_pc (float): Distance in parsecs.
    
    Returns:
    float or np.ndarray: The apparent magnitude(s) in the specific filter.
    """
    if distance_pc <= 0:
        raise ValueError("Distance must be greater than zero.")
        
    # 1. Convert absolute bolometric to absolute filter magnitude
    M_filter = M_bol - BC_filter
    
    # 2. Apply distance modulus
    m_apparent = M_filter + 5 * np.log10(distance_pc) - 5
    
    return m_apparent


def add_filter_apparent_column(df, M_bol_col, BC_col, new_col_name, distance_pc):
    """
    Calculates apparent filter magnitudes from arrays of M_bol and BC, 
    and adds them to the existing DataFrame.
    """
    # axis=1 tells Pandas to apply the function row by row
    df[new_col_name] = df.apply(
        lambda row: calc_filter_apparent_mag(
            np.array(row[M_bol_col]), 
            np.array(row[BC_col]), 
            distance_pc
        ), 
        axis=1
    )
    
    return df

def load_isochrone_data(history_directories):
    """
    Loads MESA history files from disk into a list of DataFrames.
    Kept this separate when running plotting so I don't have to pull from disk every time I re-plot
    """
    dataframes = [] 
    # base_dennis = r"history_m([0-9.]+).data"
    
    for history in history_directories:
        # Grab files from disk 
        files = glob.glob(history)
        
        # Get history files into a DataFrame
        df = get_history(files)
        dataframes.append(df)
        
    return dataframes


def build_isochrone_multi(df, ax, target_ages, x_var, y_var, x_title, y_title, target_masses=None, invert_y=True, invert_x=False):
    """
    Plots multiple isochrones (target_ages) colored by stellar mass. 
    Overlays full evolutionary tracks for specific target_masses in the background.

    Returns axes element for plotting in larger figure
    """
    all_x_vals = []
    all_y_vals = []
    last_colored_line = None

    # --- 1. Plot Evolutionary Tracks (Mass Tracks) in Background ---
    if target_masses is not None:
        for index, row in df.iterrows():
            mass = row['Mass']
            mass_val = mass[0] if isinstance(mass, (list, np.ndarray)) else mass
            
            # Check if this star's mass matches any in our target list (with slight tolerance)
            if any(np.isclose(mass_val, target_masses, atol=1e-3)):
                if x_var in df.columns and y_var in df.columns:

                    # 1. Grab the age array for this specific star
                    ages = np.array(row['star_age'])

                    # 2. Hardcode your lower age bound here (e.g., 1e6 for 1 Myr)
                    min_age_limit = 1e6  

                    # 3. Create a mask of indices where the age is above the limit
                    valid_idx = ages >= min_age_limit

                    # 4. Apply the mask to slice the X and Y arrays
                    x_track = np.array(row[x_var])[valid_idx]
                    y_track = np.array(row[y_var])[valid_idx]

                    # Only plot if there is data left after the cut
                    if len(x_track) > 0:
                        # Plot the truncated age evolution as a faint line
                        ax.plot(x_track, y_track, color='grey', alpha=0.5, zorder=1, linewidth=1.5)

                        # Expand our bounding box logic to ensure the whole track is visible
                        all_x_vals.extend(x_track)
                        all_y_vals.extend(y_track)

    # --- 2. Plot the Isochrones ---
    sorted_ages = sorted(target_ages)
    total_ages = len(sorted_ages)

    # Loop through each target age provided in the list
    for target_age in target_ages:
        isochrone_x_vals = []
        isochrone_y_vals = []
        masses = []

        for index, row in df.iterrows():
            ages = np.array(row['star_age'])
            
            # Safety Check: Skip if the star died before the target age
            if np.max(ages) < target_age:
                continue
                
            # Interpolate the exact fractional X value at the target age
            if x_var in df.columns:
                x_array = np.array(row[x_var])
                x_v = np.interp(target_age, ages, x_array)
            else:
                continue
                
            # Interpolate the exact fractional Y value at the target age
            if y_var in df.columns:
                y_array = np.array(row[y_var])
                y_v = np.interp(target_age, ages, y_array)
            else:
                continue
            
            mass = row['Mass']
            mass_val = mass[0] if isinstance(mass, (list, np.ndarray)) else mass
            
            isochrone_x_vals.append(x_v)
            isochrone_y_vals.append(y_v)
            masses.append(mass_val)

        if not masses:
            continue  # Skip this age if no stars reach it

        # Sort the extracted points by mass so the plotted line connects smoothly
        sorted_indices = np.argsort(masses)
        x_vals = np.array(isochrone_x_vals)[sorted_indices]
        y_vals = np.array(isochrone_y_vals)[sorted_indices]
        m_vals = np.array(masses)[sorted_indices]

        # Collect points for scaling axis limits
        all_x_vals.extend(x_vals)
        all_y_vals.extend(y_vals)

        # Build the LineCollection for this age
        points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='viridis', linewidth=5, alpha=0.9, zorder=2)
        lc.set_array(m_vals)
        colored_line = ax.add_collection(lc)
        last_colored_line = colored_line  # Save reference for the colorbar

        # --- Add Age Label at the Highest Mass Member ---
        # The highest mass member is the last element in the sorted arrays
        high_mass_x = x_vals[0]
        high_mass_y = y_vals[0]
        age_label = f"{target_age/1e6:.0f} Myr"

        ax.text(
            high_mass_x, high_mass_y, f"  {age_label}", 
            fontsize=10, fontweight='bold', color='black',
            verticalalignment='center', horizontalalignment='left',
            zorder=3
        )


    # Scale the axis limits to encompass all points across tracks and ages
    if all_x_vals and all_y_vals:
        ax.update_datalim(np.column_stack([all_x_vals, all_y_vals]))
        
        # Add a 15% margin to the X-axis and 10% to the Y-axis to make room for the text
        ax.margins(x=0.15, y=0.1)
        
        ax.autoscale_view()

    # Invert the axes
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    
    ax.set_xlabel(x_title, fontsize=14)
    ax.set_ylabel(y_title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Force the axes to always show 1 decimal place
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    return last_colored_line


def plot_four_panel(dataframes, filters, target_ages, distance_pc=168, target_masses=None, savename=None, savedir=None):
    """
    Creates a 2x2 grid of isochrone plots from a pre-loaded list of DataFrames.
    """
    # Initialize the 2x2 figure grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten() # Flatten to a 1D array for easy iteration
    
    # Iterate over the 4 axes, the 4 DataFrames in memory, and the 4 filter settings
    for ax, df, filt in zip(axes, dataframes, filters):

        # Calculate Apparent magnitude column
        add_filter_apparent_column(df, 'Mag_bol', filt['x_var'], 'Mag_app', distance_pc)
        
        # Call the updated isochrone function and pass the 'ax' object
        scatter = build_isochrone_multi(
            df=df, 
            ax=ax, 
            target_ages=target_ages, 
            x_var=filt['x_var'], 
            y_var=filt['y_var'], 
            x_title=filt['x_title'], 
            y_title=filt['y_title'], 
            invert_y=filt.get('invert_y', True), 
            invert_x=filt.get('invert_x', False),
            target_masses=target_masses
        )
        
    # --- Figure-Level Formatting (The Manual Control Method) ---
    
    # Push the subplots down by lowering 'top' to 0.9 (reserving 10% of the space)
    fig.subplots_adjust(top=0.97, right=0.86, bottom=0.1, left=0.1, wspace=0.25, hspace=0.2)
    
    # Create a dedicated, absolute bounding box for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.1, 0.04, 0.87])
    
    # Draw the colorbar into that specific, isolated box
    cbar = fig.colorbar(scatter, cax=cbar_ax) 
    cbar.set_label(r'Stellar Mass ($M_{\odot}$)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Save and display
    if savename is not None:
        # Assumes 'saved' is defined globally in your script
        plt.savefig(savedir+savename, dpi=600, bbox_inches='tight') 
    plt.show()

