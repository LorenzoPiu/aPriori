<p align="center">
  <img src="branding/img/Graphical_abstract_2.png" width="100%">
</p>

[![Static Badge](https://img.shields.io/badge/GPL%20v3.0%20-%20%23F5CF27?label=License&logo=gplv3&logoColor=white)](https://www.gnu.org/licenses/gpl-3.0.en.html) 
[![Static Badge](https://img.shields.io/badge/Documentation-ReadTheDocs-green?logo=readthedocs&logoColor=white)](https://apriori.readthedocs.io)
[![Static Badge](https://img.shields.io/badge/Issues-Github-02bfa6?&logo=Github&logoColor=white)](https://github.com/LorenzoPiu/aPrioriDNS/issues)
[![Static Badge](https://img.shields.io/pepy/dt/aPrioriDNS?logo=pypi&logoColor=white&color=03b1fc)](https://pepy.tech/projects/aprioridns)
[![Static Badge](https://img.shields.io/badge/Journal%20article-blue?label=DOI)](https://doi.org/10.1007/s10494-024-00626-3)
[![Static Badge](https://img.shields.io/badge/Preprint-0864ba?label=arXiv&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.24363)

A Python package to process Direct Numerical Simulations of reacting and non-reacting flows.

## Purpose of the project
The project seeks to make large DNS (Direct Numerical Simulation) datasets more accessible to a broad audience, including both specialists in Combustion and Fluid Dynamics and researchers from other disciplines.
Processing DNS data can be challenging in several ways. This package offers:
- **Field3D**: An object that automatically reads formatted data aligned with [Blastnet](https://blastnet.github.io) [1, 2], an open source scientific repository.
- **Scalar3D**: An object that efficiently manages pointers to local files, preventing memory overload.
- **Plotting utilities**: Generate visualizations with just one line of code, simplifying the validation process.

## Functionalities
This library simplifies the standard workflow commonly used for a priori validation with DNS data. 
A priori validation is typically applied to assess turbulence and combustion models. More recently, 
this approach has been extended to train and evaluate machine learning models, which are increasingly 
utilized in the fluid dynamics community to enhance the accuracy of source term modeling.

The following figure displays the typical set of operations that using aPriori can be performed with a few lines of
code:

<p align="center">
  <img src="branding/img/Graphical_abstract.png" width="85%">
</p>

## Installation

Run the following command to install:

```python
pip install aPrioriDNS
```

This will automatically install or update the following dependencies if necessary:

- numpy>=1.18.0,
- scipy>=1.12.0,
- matplotlib>=3.2.0,
- cantera>=3.0.0,
- tabulate>=0.9.0,
- requests>=2.32.0.
- PyCSP>=1.4.0
- findiff>=0.12

## Documentation

The complete software documentation is built with sphynx and hosted using ReadTheDocs at the following website:

https://apriori.readthedocs.io

The old documentation on [Gitbook](https://apriori.gitbook.io/apriori-documentation-1) is not updated and will soon be deprecated.

## How to cite
The library's software paper is currently under preparation.

If you use aPriori in your work, please **temporarily cite** the following
[peer-reviewed publication](doi:10.1007/s10494-024-00626-3), which introduces the methodology and workflows
on which the library is based:

```
Piu L, Péquin A, Freitas RSM, Iavarone S, Pitsch H, Parente A.
A data-driven approach to refine the partially stirred reactor closure
for turbulent premixed flames. Flow Turbulence Combust. 2025.
doi:10.1007/s10494-024-00626-3.
```

## Third party software
The present library relies on the code [PyCSP](https://github.com/rmalpica/PyCSP) package for Computational Singular Perturbation (CSP) analysis. PyCSP is developed and maintained by Professor Riccardo Malpica Galassi at Sapienza University of Rome. For detailed documentation and further information, please refer to the original repository.

All the libraries the software relies on are automatically installed at installation, to ensure a tested and updated environment. These libraries are mentioned in the [third party notice file](./THIRD_PARTY_NOTICES.md).

## License
This project is licensed under the [GNU General Public License](https://creativecommons.org/licenses/by/4.0/deed.en) (GPL v3.0).

© 2026 Lorenzo Piu, Heinz Pitsch and Alessandro Parente.

You are free to share and adapt this work, provided appropriate credit is given.

## Quickstart
The following code can be used to test the library once installed. A detailed explanation of the workflow presented is available [here](https://apriori.gitbook.io/apriori-documentation-1/getting-started/quickstart).

```python
"""
import aPriori as ap

# Download the dataset
ap.download(dataset='h2_lifted')

# Initialize 3D DNS field
field_DNS = ap.Field3D('Lifted_H2_subdomain')

#----------------------------Visualize the dataset-----------------------------

# Plot Temperature on the xy midplane (transposed as yx plane)
field_DNS.plot_z_midplane('T',                 # plots the Temperature
                          levels=[1400, 2000], # isocontours at 1400 and 2000
                          vmin=1400,           # minimum temperature to plot
                          title='T [K]',       # figure title
                          linewidth=2,         # isocontour lines thickness
                          transpose=True,      # inverts x and y axes
                          x_name='y [mm]',     # x axis label
                          y_name='x [mm]')     # y axis label
# Plot Temperature on the xz midplane (transposed as zx plane)
field_DNS.plot_y_midplane('T', 
                          levels=[1400, 2000], 
                          vmin=1400, 
                          title='T [K]', 
                          linewidth=2,
                          transpose=True, 
                          x_name='z [mm]', 
                          y_name='x [mm]')
# Plot Temperature on the yz midplane
field_DNS.plot_x_midplane('T', levels=[1400, 2000], vmin=1400, 
                          title='T [K]', linewidth=2)
# Plot OH mass fraction on the transposed xy midplane
field_DNS.plot_z_midplane('YOH', title=r'$Y_{OH}$', colormap='inferno',
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

#--------------------------Compute DNS reaction rates--------------------------
field_DNS.compute_reaction_rates()

# Plot reaction rates
field_DNS.plot_z_midplane('RH2O_DNS', 
                          title=r'$\dot{\omega}_{H2O}$ $[kg/m^3/s]$', 
                          colormap='inferno',
                          transpose=True, x_name='z [mm]', y_name='x [mm]')
field_DNS.plot_z_midplane('ROH_DNS', 
                          title=r'$\dot{\omega}_{OH}$ $[kg/m^3/s]$', 
                          colormap='inferno',
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

# compute kinetic energy
field_DNS.compute_kinetic_energy()

# Compute mixture fraction
field_DNS.ox = 'O2'     # Defines the species to consider as oxydizer
field_DNS.fuel = 'H2'   # Defines the species to consider as fuel
Y_ox_2=0.233  # Oxygen mass fraction in the oxydizer stream (air)
Y_f_1=0.65*2/(0.65*2+0.35*28) # Hydrogen mass fraction in the fuel stream
# (the fuel stream is composed by X_H2=0.65 and X_N2=0.35)

field_DNS.compute_mixture_fraction(Y_ox_2=Y_ox_2, Y_f_1=Y_f_1, s=2)

# Scatter plot variables as functions of the mixture fraction Z
field_DNS.scatter_Z('T', # the variable to plot on the y axis
                    c=field_DNS.YOH.value, # set color of the points
                    y_name='T [K]', 
                    cbar_title=r'$Y_{OH}$'
                    )

field_DNS.scatter_Z('ROH_DNS',
                    c=field_DNS.HRR_DNS.value, 
                    y_name=r'$\dot{\omega}_{OH}$ $[kg/m^3/s]$', 
                    cbar_title=r'$\dot{Q}_{DNS}$'
                    )

#-------------------------------Filter DNS field-------------------------------
# perform favre filtering (high density gradients)
# the output of the function is a string with the new folder's name, f_string
f_string = field_DNS.filter_favre(filter_size=16, # filter amplitude
                                        filter_type='Gauss') # 'Gauss' or 'Box'

# The string with the folder's name is now used to initialize the filered field
field_filtered = ap.Field3D(f_string)

# Visualize the effect of filtering on the Heat Release Rate
field_DNS.plot_z_midplane('HRR_DNS',
                          title=r'$\dot{Q}_{DNS}$', 
                          colormap='inferno',
                          vmax=8*1e9,
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

field_filtered.plot_z_midplane('HRR_DNS',
                          title=r'$\overline{\dot{Q}_{DNS}}$', 
                          colormap='inferno',
                          vmax=8*1e9,
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

#-------------------------Compute reaction rates (LFR)-------------------------
# Computing reaction rates directly from the filtered field (LFR approximation)
field_filtered.compute_reaction_rates()

# Compare visually the results 
field_filtered.plot_z_midplane('RH2_DNS',
                          title=r'$\overline{\dot{\omega}}_{H2,DNS}$', 
                          vmax=-20,
                          vmin=field_filtered.RH2_LFR.z_midplane.min(),
                          levels=[-300, -50, -20],
                          labels=True,
                          colormap='inferno',
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

# Compare visually the results in the z midplane
field_filtered.plot_z_midplane('RH2_LFR',
                          title=r'$\overline{\dot{\omega}}_{H2,LFR}$', 
                          vmax=-20,
                          vmin=field_filtered.RH2_LFR.z_midplane.min(),
                          levels=[-300, -50, -20],
                          labels=True,
                          colormap='inferno',
                          transpose=True, x_name='z [mm]', y_name='x [mm]')

# Compare the heat release rate results with a parity plot
f = ap.parity_plot(field_filtered.HRR_DNS.value,  # x
                   field_filtered.HRR_LFR.value,  # y
                   density=True,                  # KDE coloured
                   x_name=r'$\overline{\dot{\omega}}_{H2,DNS}$',
                   y_name=r'$\overline{\dot{\omega}}_{H2,LFR}$'
                   )

```

## Bibliography
[1] W. T. Chung, B. Akoush, P. Sharma, A. Tamkin, K. S. Jung, J. H. Chen, J. Guo, D. Brouzet, M. Talei, B. Savard, A.Y. Poludnenko & M. Ihme. Turbulence in Focus: Benchmarking Scaling Behavior of 3D Volumetric Super-Resolution with BLASTNet 2.0 Data. Advances in Neural Information Processing Systems (2023) 36.

[2] W. T. Chung, M. Ihme, K. S. Jung, J. H. Chen, J. Guo,  D. Brouzet, M. Talei, B. Jiang, B. Savard, A. Y. Poludnenko, B. Akoush, P. Sharma & A. Tamkin. BLASTNet Simulation Dataset (Version 2.0), 2023. https://blastnet.github.io/. 

[//]: # (The following two badges are commented out: )
[//]: # (https://img.shields.io/pypi/dm/aprioridns?color=03b1fc&logo=pypi&logoColor=white)
[//]: # (https://static.pepy.tech/personalized-badge/aprioridns?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=Downloads)
