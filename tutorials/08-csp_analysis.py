#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 24 16:41:09 2026

@author: lorenzo piu
"""

import os
import aPriori as ap
from aPriori import DNS
import json

# comment the following line if you don't want to download the example dataset
# ap.download(dataset='csp')

directory = os.path.join('.', 'Lifted_H2_subdomain_csp') # change this with your path to the data folder
figures_folder = 'figures_csp'

dns_field = ap.Field(directory)

dns_field.compute_csp(
    API_species=['H2O', 'OH'], 
    n_chunks=100,
    )

os.makedirs(figures_folder, exist_ok=True)

dns_field.plot_api(
    api_type='TSR', 
    n=10, 
    n_cols=5, 
    cmap='RdGy_r', 
    vmin=-1, vmax=1,
    save_path=os.path.join(figures_folder, 'api_tsr.png')
    )

dns_field.plot_api(
    api_type='OH', 
    n=10, 
    n_cols=5, 
    cmap='RdGy_r', 
    vmin=-1, vmax=1,
    save_path=os.path.join(figures_folder, 'api_oh.png')
    )

dns_field.plot_z_midplane(
    'TSR_DNS',
    colormap='gist_heat',
    transpose=True,
    remove_title=True,
    cbar_title='TSR $[s^{-1}]$',
    vmax=0,
    transparent=False,
    save_path=os.path.join(figures_folder, 'TSR.png')
    )

dns_field.plot_z_midplane(
    'TSR_DNS',
    colormap='gist_heat_r',
    transpose=True,
    remove_title=True,
    cbar_title='TSR $[s^{-1}]$',
    vmin=0,
    transparent=False,
    remove_y=True,
    save_path=os.path.join(figures_folder, 'TSR_pos.png')
    )

dns_field.plot_z_midplane(
    'YOH',
    colormap='gist_heat_r',
    transpose=True,
    remove_title=True,
    cbar_title=r'$Y_{OH}$ $[-]$',
    save_path=os.path.join(figures_folder, 'YOH.png')
    )

dns_field.plot_z_midplane(
    'T',
    colormap='gist_heat_r',
    transpose=True,
    remove_title=True,
    remove_y=True,
    cbar_title=r'$T$ $[K]$',
    save_path=os.path.join(figures_folder, 'T.png')
    )

