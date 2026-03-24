#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:00:20 2024

@author: lorenzo piu
"""

import aPriori.DNS as DNS
from aPriori.DNS import Scalar
import numpy as np

shape = [4, 2, 4]
array = np.ones(shape)
array[2:3, :, 2:3] = 2
print(array)

filter_size = 2
filtered_field = DNS.filter_3D(array, filter_size, favre=False, filter_type='Gauss')
print(filtered_field)

import os
directory = os.path.join('..','data','Lifted_H2_subdomain') # change this with your path to the data folder
T_path = os.path.join(directory,'data', 'T_K_id000.dat')
print(f"\nChecking the path \'{T_path}\' is correct...")
if not os.path.exists(T_path):
    raise ValueError("The path '{T_path}' does not exist in your system. Check to have the correct path to your data folder in the code")

# Blastnet's data contain information about the shape of the field in the info.json file
import json
with open(os.path.join(directory,'info.json'), 'r') as file:
    info = json.load(file)
DNS_shape = info['global']['Nxyz']
    
# Now we have the shape and the path of the file, we can define the Scalar object:
T = Scalar(shape=DNS_shape, path=T_path)
# Try to access the value of the temperature field:
print("Temperature value in the cells 5:8, 5:8, 5:8")
print(T._3D[5:8, 5:8, 5:8])

filter_size = 20
T_filt_gauss = DNS.filter_3D(T._3D, filter_size, favre=False, filter_type='Gauss')
print("Filtered temperature values in the cells 5:8, 5:8, 5:8")
print(T_filt_gauss[5:8, 5:8, 5:8])

T_filt_box = DNS.filter_3D(T._3D, filter_size, favre=False, filter_type='Box')
print("Filtered temperature values in the cells 5:8, 5:8, 5:8")
print(T_filt_box[5:8, 5:8, 5:8])
