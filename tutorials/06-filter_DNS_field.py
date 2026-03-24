#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:44 2024

@author: lorenzo piu
"""

import os
import aPriori as ap
from aPriori import DNS
import json

# Uncomment the following line if you did not download the dataset yet
# ap.download(dataset='h2_lifted')

directory = os.path.join('.', 'Lifted_H2_subdomain') # change this with your path to the data folder

T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    print(f"The path '{T_path}' does not exist in your system. Downloading the dataset from Github...")
    ap.download(dataset='h2_lifted')
    
DNS_field = ap.Field(directory)

filter_size = 32

# Favre filter with Gaussian filter kernel
field_filt_name = DNS_field.filter_favre(filter_size, filter_type='Gauss')
DNS_field_filt_gauss = ap.Field(field_filt_name)

# Favre filter with box filter kernel
# The box filter can take a long time as the convolution operation is not
# optimized. If you have any suggestion on how to improve it, please
# open an issue on the Github page!
DNS_field_filt_box = ap.Field(DNS_field.filter_favre(filter_size, filter_type='Box'))

# Plot the different filtering results
DNS_field.plot_z_midplane('T',
                          title='DNS',
                          cbar_title='T [K]',
                          colormap='inferno',
                          vmin=500,
                          vmax=2300,
                          transparent=False,
                          transpose=True,
                          remove_cbar=True
                          )

DNS_field_filt_box.plot_z_midplane('T',
                          title='$\Delta = 32$, Box',
                          cbar_title='T [K]',
                          colormap='inferno',
                          vmin=500,
                          vmax=2300,
                          transparent=False,
                          transpose=True,
                          remove_cbar=True,
                          remove_y=True
                          )

DNS_field_filt_gauss.plot_z_midplane('T',
                          title='$\Delta = 32$, Gauss',
                          cbar_title='T [K]',
                          colormap='inferno',
                          vmin=500,
                          vmax=2300,
                          transparent=False,
                          transpose=True,
                          remove_y=True
                          )
