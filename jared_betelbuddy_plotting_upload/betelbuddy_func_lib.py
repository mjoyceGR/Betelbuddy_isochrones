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

def calc_Mbol(log_L):
    # Fixed calculation of bolometric magnitude from luminosity
    M_bol = 4.74-2.5*log_L
    return M_bol


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
    # print(f"M_bol_filter: {M_filter}")
    
    # 2. Apply distance modulus
    m_apparent = M_filter + 5 * np.log10(distance_pc) - 5
    
    return m_apparent

def calc_filt_bol_mag(M_bol, BC_filter):
    # Converts from bolometric Mag to filter Mag
    M_filter = M_bol + BC_filter
    return M_filter

def add_filter_bol_column(df, M_bol_col, BC_col, new_col_name):
    """
    Calculates bolometric filter magnitudes from arrays of M_bol and BC, 
    and adds them to the existing DataFrame.
    """
    # axis=1 tells Pandas to apply the function row by row
    df[new_col_name] = df.apply(
        lambda row: calc_filt_bol_mag(
            np.array(row[M_bol_col]), 
            np.array(row[BC_col])
        ), 
        axis=1
    )
    
    return df


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


def build_isochrone_multi(df, ax, target_ages, x_var, y_var, x_title, y_title, target_masses=None, invert_y=True, 
                          invert_x=False, inset_zoom_lims=None):
    """
    Plots multiple isochrones (target_ages) as thin background lines. 
    Overlays full evolutionary tracks for specific target_masses in thick, colored lines.

    Returns axes element for plotting in larger figure
    """
    all_x_vals = []
    all_y_vals = []
    last_colored_line = None

    # --- 1. Plot Evolutionary Tracks (Mass Tracks) as Thick Colored Lines ---
    tracks_to_plot = []
    plotted_masses = []

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
                    min_age_limit = 0.5e6  

                    # 3. Create a mask of indices where the age is above the limit
                    valid_idx = ages >= min_age_limit

                    # 4. Apply the mask to slice the X and Y arrays
                    x_track = np.array(row[x_var])[valid_idx]
                    y_track = np.array(row[y_var])[valid_idx]

                    # Only plot if there is data left after the cut
                    if len(x_track) > 1:
                        # Expand our bounding box logic to ensure the whole track is visible
                        all_x_vals.extend(x_track)
                        all_y_vals.extend(y_track)

                        # Save the track data to plot all at once later
                        tracks_to_plot.append((x_track, y_track))
                        plotted_masses.append(mass_val)

        # Build colormap and plot the standard lines
        if plotted_masses:
            # Create a dynamic color normalizer based on the min/max masses found
            norm = plt.Normalize(vmin=min(plotted_masses), vmax=max(plotted_masses))
            cmap = plt.get_cmap('viridis')
            
            for (x_trk, y_trk), m_val in zip(tracks_to_plot, plotted_masses):
                # Standard line plot, assigning the mapped color directly
                ax.plot(x_trk, y_trk, color=cmap(norm(m_val)), linewidth=2.5, alpha=1, zorder=2)
                
            # Create a "dummy" ScalarMappable so fig.colorbar() still works perfectly
            last_colored_line = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # --- 2. Plot the Isochrones as Thin Grey Lines ---
    sorted_ages = sorted(target_ages)

    shade_isochrones = {} # Store the boundary isochrones for shading
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

        # Save the 5 Myr and 15 Myr isochrones for later shading
        if np.isclose(target_age, 5e6) or np.isclose(target_age, 15e6):
            shade_isochrones[target_age] = {
                'mass': np.array(masses)[sorted_indices],
                'x': x_vals.copy(),
                'y': y_vals.copy()
            }

        # Collect points for scaling axis limits
        all_x_vals.extend(x_vals)
        all_y_vals.extend(y_vals)

        # Plot the isochrone as a faint background line
        if len(x_vals) > 1:
            if target_age==1e7: 
                ax.plot(x_vals, y_vals, color='black', alpha=1, zorder=1000, linewidth=4) #zorder=1000, 
            else:
                ax.plot(x_vals, y_vals, color='black', alpha=1, zorder=1, linewidth=1.5) #zorder=1000, 


        # --- Dynamic Age Label Placement (Inset Only) ---
        if inset_zoom_lims is not None:
            # Unpack the limits and find the absolute min and max
            x1, x2, y1, y2 = inset_zoom_lims
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # Create a mask to find only the points that fall inside the zoom box
            inside_mask = (x_vals >= x_min) & (x_vals <= x_max) & (y_vals >= y_min) & (y_vals <= y_max)
            
            # If any part of this age track is inside the box, label it
            if np.any(inside_mask):
                x_inside = x_vals[inside_mask]
                y_inside = y_vals[inside_mask]
                
                # Pick the median index to place the label in the center of the segment
                mid_idx = len(x_inside) // 2
                label_x = x_inside[mid_idx]
                label_y = y_inside[mid_idx]
                
                age_label = f"{target_age/1e6:.1f} Myr"
                
                ax.text(
                    label_x, label_y, f"  {age_label}", 
                    fontsize=10, fontweight='bold', color='black',
                    verticalalignment='center', horizontalalignment='left',
                    zorder=3
                )

    

    # --- Shade between the 5 Myr and 15 Myr isochrones ---
    # Jared comment: This is a sketchy hard-coded chatgpt suggestion but whatever, it works. I know I can do better but I'm lazy. 
    if 5e6 in shade_isochrones and 15e6 in shade_isochrones:

        iso_5 = shade_isochrones[5e6]
        iso_15 = shade_isochrones[15e6]

        # Find mass range shared by both isochrones
        mass_min = max(iso_5['mass'].min(), iso_15['mass'].min())
        mass_max = min(iso_5['mass'].max(), iso_15['mass'].max())

        # Common mass grid
        mass_common = np.linspace(mass_min, mass_max, 300)

        # Interpolate x and y positions onto the same mass grid
        x_5 = np.interp(mass_common, iso_5['mass'], iso_5['x'])
        y_5 = np.interp(mass_common, iso_5['mass'], iso_5['y'])

        x_15 = np.interp(mass_common, iso_15['mass'], iso_15['x'])
        y_15 = np.interp(mass_common, iso_15['mass'], iso_15['y'])

        # Construct a closed polygon between the two isochrones
        shade_x = np.concatenate([x_5, x_15[::-1]])
        shade_y = np.concatenate([y_5, y_15[::-1]])

        ax.fill(
            shade_x,
            shade_y,
            color='lightgrey',
            alpha=1,
            edgecolor='none',
            zorder=0
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
    ax.grid(True, linestyle='--', alpha=0.5)

    # Force the axes to always show 1 decimal place
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    return last_colored_line


def plot_four_panel(dataframes, filters, target_ages, distance_pc=168, target_masses=None, savename=None, savedir=None):
    """
    Creates a 2x2 grid of isochrone plots from a pre-loaded list of DataFrames.
    Includes optional labeled shaded regions for filter/detection bounds.
    """
    # Initialize the 2x2 figure grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten() # Flatten to a 1D array for easy iteration
    
    # Iterate over the 4 axes, the 4 DataFrames in memory, and the 4 filter settings
    for ax, df, filt in zip(axes, dataframes, filters):

        # Add REAL bolometric magnitude column:
        df['M_bol'] = calc_Mbol(df['log_L'])

        # Calculate Apparent magnitude column
        if filt['y_var'] == 'Mag_bol_filter':
            add_filter_bol_column(df, 'M_bol', filt['filter'], 'Mag_bol_filter')
        
        # --- 1. Plot the Main Axis ---
        scatter = build_isochrone_multi(
            df=df, 
            ax=ax, 
            target_ages=target_ages, 
            target_masses=target_masses,
            x_var=filt['x_var'], 
            y_var=filt['y_var'], 
            x_title=filt['x_title'], 
            y_title=filt['y_title'], 
            invert_y=filt.get('invert_y', True), 
            invert_x=filt.get('invert_x', False)
        )
        
        # --- 2. Add the Inset Zoom (if specified in the filter) ---
        axins = None
        if 'inset_pos' in filt and 'zoom_lims' in filt:
            
            axins = ax.inset_axes(filt['inset_pos'])
            
            # Call plotting function for inset
            build_isochrone_multi(
                df=df, 
                ax=axins, 
                target_ages=target_ages,
                target_masses=target_masses, 
                x_var=filt['x_var'], 
                y_var=filt['y_var'], 
                x_title="", 
                y_title="", 
                invert_y=filt.get('invert_y', True), 
                invert_x=filt.get('invert_x', False),
                inset_zoom_lims=filt['zoom_lims']
            )
            
            # Apply manual data limits to zoomed box
            x1, x2, y1, y2 = filt['zoom_lims']
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            
            rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, zorder=1)
            for line in connectors:
                line.set_visible(False)

        # --- 3. Add Shaded Regions & Legend ---
        if 'shades' in filt:
            has_labels = False 
            
            for shade in filt['shades']:
                # Use .get() so it defaults to None if you completely forget to include the key
                x_lims = shade.get('x_lims')
                y_lims = shade.get('y_lims')
                color = shade.get('color', 'gray')
                alpha = shade.get('alpha', 0.25)
                label = shade.get('label', '')
                try: 
                    upper = shade.get('upperlimit', '')
                except:
                    upper = False
                
                if label:
                    has_labels = True
                
                target_axes = [ax] if axins is None else [ax, axins]

                for target_ax in target_axes:
                    
                    # Only assign the label to the main axis to prevent legend duplication
                    current_label = label if target_ax == ax else "_nolegend_"
                    
                    if x_lims is not None and y_lims is not None:
                        # Both constraints exist: Draw the bounded 2D target box
                        x_sorted = sorted(x_lims)
                        target_ax.fill_between(
                            x_sorted, y_lims[0], y_lims[1], 
                            color=color, alpha=alpha, zorder=0, label=current_label
                        )
                        
                    elif (x_lims is None and y_lims is not None) and upper: # Jared added for an upper limit
                        print('using upper limit')
                        # use only upper limit
                        ymin = ax.get_ylim()[0]
                        target_ax.axhspan(
                            ymin, y_lims[0], 
                            color=color, alpha=alpha, zorder=0, label=current_label
                        )
                        ax.set_ylim(ymin, ax.get_ylim()[-1])

                    elif x_lims is None and y_lims is not None:
                        # Only magnitude constraints: Draw a horizontal band
                        target_ax.axhspan(
                            y_lims[0], y_lims[1], 
                            color=color, alpha=alpha, zorder=0, label=current_label
                        )


                    elif y_lims is None and x_lims is not None:
                        # Only temperature constraints: Draw a vertical band
                        target_ax.axvspan(
                            x_lims[0], x_lims[1], 
                            color=color, alpha=alpha, zorder=0, label=current_label
                        )
            
            if has_labels:
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # --- Figure-Level Formatting ---
    fig.subplots_adjust(top=0.97, right=0.86, bottom=0.1, left=0.1, wspace=0.25, hspace=0.2)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.04, 0.87])
    
    cbar = fig.colorbar(scatter, cax=cbar_ax) 
    cbar.set_label(r'Stellar Mass ($M_{\odot}$)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Save and display
    if savename is not None:
        plt.savefig(savedir + savename, dpi=600, bbox_inches='tight') 
    plt.show()

