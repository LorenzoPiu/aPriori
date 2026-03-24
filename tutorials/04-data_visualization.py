#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:21:37 2024

@author: lorenzo piu
"""

import os
import aPriori as ap
import json

# Uncomment the following line if you did not download the dataset yet
# ap.download(dataset='h2_lifted')

directory = os.path.join('.','Lifted_H2_subdomain') # change this with your path to the data folder

# Check the folder with the data exists in your system
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")
else:
    print("Folder path OK\n")

DNS_field = ap.Field(directory)

# Visualize the data
# Default plotting of a variable along the x, y, z midplanes
DNS_field.plot_z_midplane('YH2O2')
DNS_field.plot_y_midplane('YH2O2')
DNS_field.plot_x_midplane('YH2O2')

# Advanced settings to plot. We'll consider the z midplane for simplicity
DNS_field.plot_z_midplane('YOH',
                          levels=[3e-3,10e-3,15e-3], # isocontour lines
                          vmin=3e-3,                 # minimum temperature to plot
                          title=r'$Y_{H2O2}$',       # figure title
                          linewidth=1,               # isocontour lines thickness
                          transpose=True,            # inverts x and y axes
                          x_name='y [mm]',           # x axis label
                          y_name='x [mm]',           # y axis label
                          colormap='inferno',        # change colormap
                          )

# Scatter variables as functions of the mixture fraction z
# Compute the mixture fraction with the compute_mixture_fraction method
DNS_field.ox = 'O2'     # Defines the species to consider as oxydizer
DNS_field.fuel = 'H2'   # Defines the species to consider as fuel
Y_ox_2=0.233  # Oxygen mass fraction in the oxydizer stream (air)
Y_f_1=0.65*2/(0.65*2+0.35*28) # Hydrogen mass fraction in the fuel stream
# (the fuel stream is composed by X_H2=0.65 and X_N2=0.35)

DNS_field.compute_mixture_fraction(Y_ox_2=Y_ox_2, Y_f_1=Y_f_1, s=2)

# Scatter plot variables as functions of the mixture fraction Z
DNS_field.scatter_Z('T', # the variable to plot on the y axis
                    c=DNS_field.YOH.value, # set color of the points
                    y_name='T [K]', 
                    cbar_title=r'$Y_{OH}$'
                    )

DNS_field.scatter_Z('YH2O2', # the variable to plot on the y axis
                    c=DNS_field.YOH.value, # set color of the points
                    y_name=r'$Y_{H2O2}$', 
                    cbar_title=r'$Y_{OH}$'
                    )
