#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:50:44 2024

@author: lorenzo piu
"""

import os
import aPriori as ap
from aPriori.DNS import Field
from aPriori import DNS
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

DNS_field = Field(directory)

cut_field_name = DNS_field.cut([40,50,10])

DNS_field_cut = Field(cut_field_name)

DNS_field.plot_z_midplane('U_X',
                          vmin=100,
                          vmax=280, 
                          transparent=False,
                          cbar_title=r'$U_x [m/s]$',
                          title='Uncut field',
                          dpi=500
                          )


DNS_field_cut.plot_z_midplane('U_X',
                              vmin=100,
                              vmax=280, 
                              transparent=False,
                              cbar_title=r'$U_x [m/s]$',
                              title='Cut field',
                              dpi=500
                              )
