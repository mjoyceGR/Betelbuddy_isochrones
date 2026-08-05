#!/usr/bin/env python3

import mesa_reader as mr 
import numpy as np
import glob
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
sys.path.append('/home/mjoyce/Betelgeuse/Jared_Betelshock_mass_estimate/Betelbuddy_grids')
import functions_library as fl

alopeke_dir = 'alopeke/'
zimpol_dir = 'zimpol/'
HST_dir = 'HST_stis_FUV/' ## doesn't exist yet
B_dir = 'B_band/'

dirs = [B_dir] #[alopeke_dir, zimpol_dir, HST_dir, B_dir]

for directory in dirs:
	d = directory
	#print("d: ", d)
	#sys.exit()
	filter_system = str(d).split('/')[0]

	fig, ax = plt.subplots(figsize=(16,20))	

	for f in glob.glob(d+'/hist*data'):
		#print("this is f: ",f)
		
		md = mr.MesaData(f)
		
		star_age = md.star_age
		star_age = star_age/1e6 ## to Myr
		#print("star_age: ", star_age)
		#sys.exit()

		star_mass = md.star_mass

		x,y = fl.which_columns(f, filter_system) ## ingests file, filter system; returns two mr.data arrays

		plt.plot(x, y, '.-', markersize=4, alpha=1, color = 'k')
		     #, label=f.split('history_')[1].split('.data')[0])

		target_ages = [5,10,15]
		#delta = 0.8
		for target_age in target_ages:
			selected_age = fl.find_interpolated_nearest(target_age, star_age, number_of_points=1000)


			# selected_log_Teff = np.interp(target_age, star_age, log_Teff)
			# selected_log_L = np.interp(target_age, star_age, log_L)
			# plt.plot(selected_log_Teff, selected_log_L, '*', color=fl.which_color(target_age), markersize=20, alpha=1)

			selected_x = np.interp(target_age, star_age, x)
			selected_y = np.interp(target_age, star_age, y)
			plt.plot(selected_x, selected_y, '*', color=fl.which_color(target_age), markersize=20, alpha=1)

	plt.xlabel('Teff', fontsize=40)
	plt.ylabel('Log L', fontsize=40)

	plt.gca().invert_xaxis()

	ax.tick_params(axis='both', which='major', labelsize=26)
	ax.tick_params(axis='both', which='minor', labelsize=26)

	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(which='both', width=4)
	ax.tick_params(which='major', length=12)
	ax.tick_params(which='minor', length=8, color='black')

	plt.legend(loc=2, fontsize=18)
	plt.show()

	#plt.savefig('four_buddies_'+str(d).split('/')[0]+'.png')
	plt.close()