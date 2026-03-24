#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:10:20 2024

@author: lorenzo piu
"""

import os
from aPriori.DNS import Field
from aPriori import DNS
import aPriori as ap
import json

# Comment the following line if you already downloaded the dataset
ap.download(dataset='h2_lifted')

# Change this with your path to the data folder if necessary
directory = os.path.join('.','Lifted_H2_subdomain') 

# Check the folder with the data exists in your system
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")
else:
    print("Folder path OK\n")

# Blastnet's data contain information about the shape of the field in the info.json file
import json
with open(os.path.join(directory,'info.json'), 'r') as file:
    info = json.load(file)
DNS_shape = info['global']['Nxyz']
    
DNS_field = Field(directory)

DNS_field.T

DNS_field.U_Y._3D

filt_YO2 = DNS.filter_3D(DNS_field.YO2._3D, 8)

DNS_field.plot_z_midplane('YH2O2')


