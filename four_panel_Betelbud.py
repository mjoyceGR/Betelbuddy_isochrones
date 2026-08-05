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
#HST_dir = '' ## doesn't exist yet
B_dir = 'B_band/'

dirs = [alopeke_dir]# [alopeke_dir, zimpol_dir, B_dir]


for directory in dirs:
	d = directory
	#print("d: ", d)
	#sys.exit()

	fig, ax = plt.subplots(figsize=(16,20))	

	for f in glob.glob(d+'/hist*data'):
		#print("this is f: ",f)
		
		md = mr.MesaData(f)
		
		star_age = md.star_age
		star_age = star_age/1e6 ## to Myr
		#print("star_age: ", star_age)
		#sys.exit()

		star_mass = md.star_mass

		log_Teff = md.log_Teff
		log_L = md.log_L
		log_R = md.log_R
		log_g = md.log_g

		Teff = 10.0**log_Teff
		L = 10.0**log_L
		R = 10.0**log_R

		plt.plot(log_Teff, log_L, '.-', markersize=4, alpha=1, color = 'k')
		     #, label=f.split('history_')[1].split('.data')[0])

		target_ages = [5,10,15]
		#delta = 0.8
		for target_age in target_ages:
			#selected_age = find_nearest(target_age,star_age)

			selected_age = fl.find_interpolated_nearest(target_age, star_age, number_of_points=1000)
			#print('selected_age: ',selected_age)

			selected_log_Teff = np.interp(target_age, star_age, log_Teff)
			selected_log_L = np.interp(target_age, star_age, log_L)

			# plt.plot(log_Teff[selected_age], log_L[selected_age], '*', color=fl.which_color(target_age), markersize=20, alpha=1)
			plt.plot(selected_log_Teff, selected_log_L, '*', color=fl.which_color(target_age), markersize=20, alpha=1)

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