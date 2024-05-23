#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:17:23 2024

@author: Lorenzo Piu
"""

import json
import os
import sys
from tabulate import tabulate
import shutil
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.signal import tukey
from scipy.ndimage import gaussian_filter1d

from ._variables import variables_list
from ._variables import mesh_list
from ._utils import (
    check_data_files,
    check_folder_structure,
    extract_species,
    find_kinetic_mechanism,
    extract_filter,
    change_folder_name,
    check_mass_fractions,
    check_reaction_rates
)
from ._data_struct import folder_structure


###########################################################
#                       Field3d
###########################################################
class Field3D():
    """
    Class representing a 3D field with various attributes and methods for visualization and data management.

    Description:
    ------------
    
    The `Field3D` class encapsulates a 3D field used in DNS, allowing for 
    operations such as data loading, filtering, and accessing various properties 
    of the field. This class is central to handling the volumetric data in 
    computational fluid dynamics simulations.
    
    The field to use as input must be formatted as Blastnet's datasets: 
    https://blastnet.github.io
    
    Attributes:
    -----------
    
    variables : dict
        Dictionary containing variable names and their corresponding settings.
    
    mesh : Mesh3D object
        Object containing the mesh. See the class Mesh3D for more information
    
    __field_dimension : int
        Dimensionality of the field, always set to 3
        

    Methods:
    --------
    
    __init__(self, folder_path): 
        Constructor method to initialize a 3D field object.
        
        dynamic attributes : Scalar3D objects
        depending on the files contained in the main folder folder_path, 
        initializes various attributes as Scalar3D objects. 
        Example :
        Field attributes :
        +-------------+---------------------------------------------------------+
        |  Attribute  |                          Path                           |
        +-------------+---------------------------------------------------------+
        |    S_DNS    |     ../data/DNS_DATA_TEST/data/S_DNS_s-1_id000.dat      |
        |      P      |        ../data/DNS_DATA_TEST/data/P_Pa_id000.dat        |
        |     RHO     |     ../data/DNS_DATA_TEST/data/RHO_kgm-3_id000.dat      |
        |      T      |        ../data/DNS_DATA_TEST/data/T_K_id000.dat         |
        |     U_X     |      ../data/DNS_DATA_TEST/data/UX_ms-1_id000.dat       |
        |     U_Y     |      ../data/DNS_DATA_TEST/data/UY_ms-1_id000.dat       |
        |     U_Z     |      ../data/DNS_DATA_TEST/data/UZ_ms-1_id000.dat       |
        |     YH2     |        ../data/DNS_DATA_TEST/data/YH2_id000.dat         |
        |     YO2     |        ../data/DNS_DATA_TEST/data/YO2_id000.dat         |
        |    YH2O     |        ../data/DNS_DATA_TEST/data/YH2O_id000.dat        |
        |     YH      |         ../data/DNS_DATA_TEST/data/YH_id000.dat         |
        |     YO      |         ../data/DNS_DATA_TEST/data/YO_id000.dat         |
        |     YOH     |        ../data/DNS_DATA_TEST/data/YOH_id000.dat         |
        |    YHO2     |        ../data/DNS_DATA_TEST/data/YHO2_id000.dat        |
        |    YH2O2    |       ../data/DNS_DATA_TEST/data/YH2O2_id000.dat        |
        |     YN2     |        ../data/DNS_DATA_TEST/data/YN2_id000.dat         |
        |   RH2_DNS   |  ../data/DNS_DATA_TEST/data/RH2_DNS_kgm-3s-1_id000.dat  |
        |   RO2_DNS   |  ../data/DNS_DATA_TEST/data/RO2_DNS_kgm-3s-1_id000.dat  |
        |  RH2O_DNS   | ../data/DNS_DATA_TEST/data/RH2O_DNS_kgm-3s-1_id000.dat  |
        |   RH_DNS    |  ../data/DNS_DATA_TEST/data/RH_DNS_kgm-3s-1_id000.dat   |
        |   RO_DNS    |  ../data/DNS_DATA_TEST/data/RO_DNS_kgm-3s-1_id000.dat   |
        |   ROH_DNS   |  ../data/DNS_DATA_TEST/data/ROH_DNS_kgm-3s-1_id000.dat  |
        |  RHO2_DNS   | ../data/DNS_DATA_TEST/data/RHO2_DNS_kgm-3s-1_id000.dat  |
        |  RH2O2_DNS  | ../data/DNS_DATA_TEST/data/RH2O2_DNS_kgm-3s-1_id000.dat |
        |   RN2_DNS   |  ../data/DNS_DATA_TEST/data/RN2_DNS_kgm-3s-1_id000.dat  |
        +-------------+---------------------------------------------------------+
    
    build_attributes_list(self): 
        Build lists of attribute names and corresponding file paths based on the configuration specified in variables_list.
        
    check_valid_attribute(self, input_attribute):
        check if the input attribute is assigned to the Field3D object.
    
    compute_chemical_timescale(self, mode='SFR', verbose=False):
        Computes the chemical timescale for the field, useful in Partially Stirred Reactor (PaSR) modeling.
    
    compute_kinetic_energy(self):
        Computes the velocity module and saves it to a file.
    
    compute_mixing_timescale(self, mode='Kolmo'):
        Computes the mixing timescale for the field, useful for turbulence modeling.
    
    compute_reaction_rates(self, n_chunks = 5000):
        Computes the source terms for a given chemical reaction system.
        The reaction rates are computed using Arrhenius equation for
        the given kinetic mechanism.
        
    compute_reaction_rates_batch(self, n_chunks = 5000, tau_c='SFR', tau_m='Kolmo'):
        Computes the source terms for a given chemical reaction system.
        The reaction rates are computed integrating an ideal reactor in time.
        This formulation corresponds to the one used in the PaSR model, without
        multiplying the output for the cell reacting fraction gamma
    
    compute_residual_kinetic_energy(self, mode='Yosh'):
        Function to compute the residual kinetic energy.
    
    compute_residual_dissipation_rate(self, mode='Smag'):
        Computes the residual dissipation rate for a filtered velocity field
    
    compute_strain_rate(self, save_derivatives=False, save_tensor=True, verbose=False):
        This function computes the strain rate of the velocity 
        components (U, V, W) over a 3D mesh.
        
    compute_tau_r(self, mode='Smag', save_tensor_components=True):
        Computes the anisotropic part of the residual stress tensor, denoted as \(\tau_r\).
        
    compute_velocity_module(self):
        Computes the velocity module and saves it to a file.
        
    cut(self, cut_size, mode='xyz'):
        Cut a field into a section based on a specified cut size.
        
    filter_favre(self, filter_size, filter_type='Gauss'):
        Filter every scalar in the field object using Favre-averaging.
        
    filter(self, filter_size, filter_type='Gauss'):
        Filter every scalar in the field object.
    
    find_path(self, attr):
        Finds a specified attribute in the attributes list and returns the corresponding element 
        in the paths list.
    
    plot_x_midplane(self, attribute): 
        Plot the midplane along the x-axis for the specified attribute.
        
    plot_y_midplane(self, attribute): 
        Plot the midplane along the y-axis for the specified attribute.
        
    plot_z_midplane(self, attribute): 
        Plot the midplane along the z-axis for the specified attribute.
    
    print_attributes(self):
        Prints the valid attributes of the class and their corresponding file paths.
    
    update(self, verbose=False): 
        Update the attributes of the class based on the existence of files in 
        the specified data path.
    """
    variables = variables_list
    mesh = mesh_list
    
    __field_dimension = 3
    
    
    def __init__(self, folder_path):
        print("\n---------------------------------------------------------------")
        print("Initializing 3D Field\n")
        # check the folder structure and files
        check_folder_structure(folder_path)
        _, ids = check_data_files(folder_path)
        print("Folder structure OK")
        
        self.folder_path = folder_path
        self.data_path = os.path.join(folder_path, folder_structure["data_path"])
        self.chem_path = os.path.join(folder_path, folder_structure["chem_path"])
        self.grid_path = os.path.join(folder_path, folder_structure["grid_path"])
        
        self.filter_size = extract_filter(folder_path)
        
        with open(os.path.join(self.folder_path,'info.json'), 'r') as file:
            self.info = json.load(file)
        
        self.shape = self.info['global']['Nxyz']
        
        self.id_string = ids
        
        print("\n---------------------------------------------------------------")
        print ("Building mesh attribute...")
        X=Scalar3D(self.shape, path=os.path.join(self.grid_path,mesh_list["X_mesh"][0]) )
        Y=Scalar3D(self.shape, path=os.path.join(self.grid_path,mesh_list["Y_mesh"][0]) )
        Z=Scalar3D(self.shape, path=os.path.join(self.grid_path,mesh_list["Z_mesh"][0]) )
        
        self.mesh = Mesh3D(X, Y, Z)
        print ("Mesh fields read correctly")
        
        
        print("\n---------------------------------------------------------------")
        print("Reading kinetic mechanism...")
        self.kinetic_mechanism = find_kinetic_mechanism(self.chem_path)
        print(f"Kinetic mechanism file found: {self.kinetic_mechanism}")
        self.species = extract_species(self.kinetic_mechanism)
        print("Species:")
        print(self.species)
        
        
        print("\n---------------------------------------------------------------")
        print ("Building scalar attributes...")
        self.attr_list, self.paths_list = self.build_attributes_list()
        self.update(verbose=True)
        
                        
    def build_attributes_list(self):
        """
        Build lists of attribute names and corresponding file paths 
        based on the configuration specified in variables_list.
        This operation is needed to handle different variables depending on
        the kinetic mechanism, and to build attributes corresponding to different
        models.
    
        Returns:
        --------
        attr_list : int
            List of attribute names.
            
        paths_list : list
            List of corresponding file paths.
        """
        attr_list = []
        paths_list = []
        # bool_list = []
        for attribute_name in variables_list:
            if variables_list[attribute_name][1] == False: # non-species-dependent names
                if variables_list[attribute_name][2] == None:
                    file_name = variables_list[attribute_name][0].format(self.id_string)
                    path = os.path.join(self.data_path, file_name)
                    paths_list.append(path)
                    attr_list.append(attribute_name)
                else: # Handling multiple models variables
                    if variables_list[attribute_name][3] == False:
                        for model in variables_list[attribute_name][2]:
                            file_name = variables_list[attribute_name][0].format(model, self.id_string)
                            path = os.path.join(self.data_path, file_name)
                            paths_list.append(path)
                            attr_list.append(attribute_name.format(model))
                    else: # handling tensors that have multiple models. NOTE: 
                          # for the moment I'm not handling species tensors or 
                          # tensors without models
                        for model in variables_list[attribute_name][2]:
                            for j in range(1,4):
                                for i in range(1,4):
                                    if (not variables_list[attribute_name][3] == 'Symmetric') or i<=j:
                                        file_name = variables_list[attribute_name][0].format(i,j,model, self.id_string)
                                        path = os.path.join(self.data_path, file_name)
                                        paths_list.append(path)
                                        attr_list.append(attribute_name.format(i,j,model))
            else: #handling the species attributes
                for specie in self.species:
                    file_name = variables_list[attribute_name][0].format(specie, self.id_string)
                    path = os.path.join(self.data_path, file_name)
                    paths_list.append(path)
                    attr_list.append(attribute_name.format(specie))
                    
        return attr_list, paths_list
    
    def check_valid_attribute(self, input_attribute):
        """
        Check if the input attribute is valid.
    
        Parameters:
        - input_attribute (str): The attribute to be checked for validity.
    
        Raises:
        - ValueError: If the input_attribute is not valid.
        """
        valid_attributes = [attr for attr, is_present in zip(self.attr_list, self.bool_list) if is_present]
        if input_attribute not in valid_attributes:
            valid_attributes_str = '\n'.join(valid_attributes)
            raise ValueError(f"The attribute '{input_attribute}' is not valid. \nValid attributes are: \n{valid_attributes_str}")


    def compute_chemical_timescale(self, mode='SFR', verbose=False):
        '''
        Computes the chemical timescale for the field, useful in Partially Stirred Reactor (PaSR) modeling.
        
        Description:
        ------------
        This method calculates the chemical timescale for the field using different modes. The available modes are 'SFR' (Slowest Fastest Reaction), 'FFR' (Fastest Fastest Reaction), and 'Ch' (Chomiak timescale). The computation is valid only for filtered fields, and it leverages reaction rates and species molar concentrations.
        
        Parameters:
        -----------
        mode : str, optional
            The mode of timescale computation. Valid options are 'SFR', 'FFR', and 'Ch'. Default is 'SFR'.
        verbose : bool, optional
            If True, prints detailed information during the computation. Default is False.
        
        Raises:
        -------
        ValueError
            If the field is not a filtered field or if the length of species and reaction rates lists do not match the number of species.
        AttributeError
            If the required attributes 'fuel' and 'ox' are not defined when mode is 'Ch'.
        
        Workflow:
        ---------
        1. Validation
           - Checks if `mode` is valid and if the field is filtered.
           - Ensures species and reaction rates lists match the number of species.
    
        2. Mode: 'SFR' or 'FFR'
           - Collects paths to reaction rates and species concentrations.
           - Computes the timescales \(\tau_c^{SFR}\) and \(\tau_c^{FFR}\).
           - Saves the computed timescales to files.
    
        3. Mode: 'Ch'
           - Validates the presence of 'fuel' and 'ox' attributes.
           - Computes the Chomiak timescale.
           - Saves the computed timescale to a file.
    
        Returns:
        --------
        None
        '''
        valid_modes = ['SFR', 'FFR', 'Ch']
        check_input_string(mode, valid_modes, 'mode')
        
        # Check that the field is a filtered field, because in this
        # version of the code we only provide the computation of the 
        # chemical timescale for modelling purposes, that is to be used in the 
        # PaSR.
        # TODO: understand how to mix the computation of the timescales with
        # different models (e.g. Tau_c with R_LFR and R_DNS, Tau_m with 
        # Smagorinsky, DNS, Germano, etc...) cause it can be interesting to
        # compute the sensitivity of the PaSR on the access to DNS data.
        # I mean, compute Tau_c with access to DNS data and without, try to 
        # train a model or compute the PaSR directly, and then see the difference
        if self.filter_size == 1:
            raise ValueError("The field is not a filtered field.\n"
                             "This version of the code only allows the computation of the chemical timescale for filtered fields."
                             "You can filter the entire field with the command:\n>>> your_field_object.filter(filter_size)"
                             "\nor:\n>>>your_field_object.filter_favre(filter_size)")
        
        if mode=='SFR' or mode=='FFR':
            
            reaction_rates_paths = []
            for attr, path in zip(self.attr_list, self.paths_list):
                if attr.startswith('R') and ('LFR' in attr): 
                    reaction_rates_paths.append(path)# To compute the 
                # chemical timescale we use the LFR rates cause it's a modelled
                # quantity, and in a posteriori LES we don't have access to DNS
                # information
            species_paths = []
            for attr, path in zip(self.attr_list, self.paths_list):
                if attr.startswith('Y'):
                    species_paths.append(path)
            
            if (len(species_paths)!=len(self.species)) or(len(reaction_rates_paths)!=len(self.species)):
                raise ValueError("Lenght of the lists must be equal to the number of species. "
                                 "Check that all the species molar concentrations and Reaction Rates are in the data folder."
                                 "\nYou can compute the reaction rates with the command:"
                                 "\n>>> your_filt_field.compute_reaction_rates()"
                                 "\n\nOperation aborted.")
            
            inf_time = 1e20 #value to use as infinite for the comparison
            
            tau_c_SFR = np.ones_like(self.RHO.value)*(-inf_time)
            tau_c_FFR = np.ones_like(self.RHO.value)*(inf_time)
            
            if verbose:
                print('Computing chemical Timescales...\n')
            for Y_path, R_path in zip(species_paths, reaction_rates_paths):
                if verbose:
                    print('...')
                Y          = Scalar3D(self.shape, path=Y_path)
                R          = Scalar3D(self.shape, path=R_path)
                tau_2      = np.abs(Y.value/R.value)
                idx        = R.value<1e-10 # indexes of the dormant species
                tau_2[idx] = -inf_time     # in this way the dormant species should not be considered
                tau_c_SFR  = np.maximum(tau_c_SFR, tau_2)
                tau_2[idx] = inf_time
                tau_c_FFR  = np.minimum(tau_c_FFR, tau_2)
    
            tau_c_SFR = self.RHO.value * tau_c_SFR
            tau_c_FFR = self.RHO.value * tau_c_FFR
                
            save_file(tau_c_SFR, self.find_path('Tau_c_SFR'))
            save_file(tau_c_FFR, self.find_path('Tau_c_FFR'))
            
        elif mode=='Ch':
            
            if (not hasattr(self, 'fuel')) or (not hasattr(self, 'ox')):
                raise AttributeError("The attributes 'fuel' and 'ox'"
                    " are not defined for this field, so it is not possible"
                    " to compute Chomiak time scale." 
                    " \nPlease specify the fuel and oxidizer in your mixture"
                    " with the following command:"
                    " \n>>> # Example:"
                    " \n>>> your_field_name.fuel = 'CH4'"
                    " \n>>> your_field_name.ox   = 'O2'")
            
            Y_ox   = Scalar3D(shape=self.shape, path=self.find_path(f"Y{self.ox}"))
            Y_fuel = Scalar3D(shape=self.shape, path=self.find_path(f"Y{self.fuel}"))
            R_ox   = Scalar3D(shape=self.shape, path=self.find_path(f"R{self.ox}_LFR"))
            R_fuel = Scalar3D(shape=self.shape, path=self.find_path(f"R{self.fuel}_LFR"))
            RHO    = Scalar3D(shape=self.shape, path=self.find_path('RHO'))
            
            tau_chomiak = RHO.value*np.minimum( Y_ox.value/np.maximum(np.abs(R_ox.value),1e-10), Y_fuel.value/np.maximum(np.abs(R_fuel.value),1e-10) )
            save_file(tau_chomiak, self.find_path('Tau_c_Ch'))
            
        self.update()
        return
    
    
    def compute_kinetic_energy(self):
        """
        Computes the velocity module and saves it to a file.
    
        This method calculates the velocity module by squaring the values of U_X, U_Y, and U_Z, 
        summing them up, and then taking the square root of the result. The computed velocity 
        module is then saved to a file using the `save_file` function. The file path is determined 
        by the `find_path` method with 'U' as the argument. After saving the file, the `update` 
        method is called to refresh the attributes of the class.
    
        Note: 
        -----
            
        - `self.U_X`, `self.U_Y`, and `self.U_Z` are assumed to be attributes of the class 
          representing components of velocity. Make sure to check you have the relative files in your
          data folder. To check, use the method <your_field_name>.print_attributes. 
        """
        if self.filter_size==1:
            closure = 'DNS'
        else:
            closure = 'LES'
        
        attr_name = 'Kappa_{}'.format(closure)
        if hasattr(self, attr_name):
            K = 0.5*getattr(self, attr_name).value
        else:
            K  = self.U_X.value**2
            K += self.U_Y.value**2
            K += self.U_Z.value**2
            K  = 0.5*K
            
            
        save_file(K, self.find_path(attr_name))
        
        self.update()
        pass
    
    def compute_mixing_timescale(self, mode='Kolmo'):
        '''
        Computes the mixing timescale for the field, useful for turbulence modeling.
        
        Description:
        ------------
        This method calculates the mixing timescale using either the Kolmogorov model or the integral length scale model. The computation relies on residual kinetic energy and residual dissipation rate fields.
        
        Parameters:
        -----------
        mode : str, optional
            The mode of timescale computation. Valid options are 'Kolmo' (Kolmogorov) or 'Int' (Integral length scale). Default is 'Kolmo'.
        
        Raises:
        -------
        ValueError
            If the specified mode is not valid.
        Warning
            If the 'C_mix' attribute is not defined for the integral length scale model.
        
        Workflow:
        ---------
        1. Validation
           - Checks if `mode` is valid.
        
        2. Mode: 'Kolmo'
           - Computes the Kolmogorov timescale:
             \[
             \tau_m^{Kolmo} = \sqrt{\frac{k_r}{\epsilon_r} \sqrt{\frac{\mu}{\rho \epsilon_r}}}
             \]
           - Saves the computed timescale to a file.
        
        3. Mode: 'Int'
           - Ensures `C_mix` is defined, initializes to 0.1 if not.
           - Computes the integral length scale timescale:
             \[
             \tau_m^{Int} = \frac{C_{mix} k_r}{\epsilon_r}
             \]
           - Saves the computed timescale and `C_mix` value to files.
        
        Returns:
        --------
        None
        
        Example:
        --------
        >>> field = Field3D('your_folder_path')
        >>> field.compute_mixing_timescale(mode='Kolmo')
        '''
        
        # At the moment this function only supports the Yoshizawa model for
        # the residual kinetic energy and the Smagorinski model for the
        # residual dissipation rate.
        
        valid_modes = self.variables["Tau_m_{}"][2]
        check_input_string(mode, valid_modes, 'mode')
        
        k_r        = Scalar3D(self.shape, path=self.find_path('K_r_Yosh'))
        epsilon_r  = Scalar3D(self.shape, path=self.find_path('Epsilon_r_Smag'))
        Mu         = Scalar3D(self.shape, path=self.find_path(f'Mu'))
        
        if mode.lower() == 'kolmo':
            
            tau_m_kolmo = np.sqrt( k_r.value/epsilon_r.value * np.sqrt(Mu.value/self.RHO.value/epsilon_r.value) )
    
            save_file(tau_m_kolmo, self.find_path(f'Tau_m_{mode.capitalize()}'))

        elif mode.lower() == 'int':
            # Check that the C_mix constant is defined
            if not hasattr(self, 'C_mix'):
                Warning("The field does not have an attribute C_mix.\n The integral lengthscale model constant C_mix will be initialized by default to 0.1")
                self.C_mix = 0.1
            
            C_mix = self.C_mix
            tau_m_integral = C_mix*k_r.value/epsilon_r.value
            
            with open(os.path.join(self.folder_path, 'C_mix.txt'), 'w') as f:
                # Write the value of the C_I constant to the file
                f.write(str(C_mix))
    
            save_file(tau_m_integral, self.find_path(f'Tau_m_{mode.capitalize()}'))
            
        self.update()
        
        return
    
    def compute_residual_kinetic_energy(self, mode='Yosh'):
        """
        Description
        -----------
        Function to compute the residual kinetic energy.
        
        Real value computed with information at DNS level:
            
            .. math::   k_{SGS} = \\bar{U_i U_i} - \\bar{U_i} \\bar{U_i}
        
        Yoshizawa expression:
        
            .. math::  k_{SGS} = 2 C_I \\bar{\\rho} \\Delta^2 |\\tilde{S}|^2

        Parameters
        ----------
        mode : TYPE, optional
            The default is 'Yosh'.

        Returns
        -------
        None.

        """
        
        valid_modes = ['DNS', 'Yosh']
        check_input_string(mode, valid_modes, 'mode')
        
        # Check that the field is a filtered field, if not it does not make sense
        # to compute the closure for the residual quantities
        if self.filter_size == 1:
            raise ValueError("The field is not a filtered field.\n"
                             "Computing residual quantities only makes sense for filtered fields."
                             "You can filter the entire field with the command:\n>>>your_field_object.filter(filter_size)"
                             "\nor:\n>>>your_field_object.filter_favre(filter_size)")
        
        if mode == 'DNS':
            if not hasattr(self, 'DNS_folder_path'):
                raise AttributeError("The filtered field does not have a value to identify the associated unfiltered data.\n"
                                 "The path of the unfiltered data folder must be assigned with the following command:\n"
                                 ">>> your_filtered_field.DNS_folder_path = 'your_unfiltered_DNS_folder_path'")
            DNS_field = Field3D(self.DNS_folder_path)
            
            # Check what filter was used for the folder and keep consistency
            if 'favre' in self.folder_path.lower():
                favre = True
            if 'box' in self.folder_path.lower():
                filter_type = 'box'
            elif 'gauss' in self.folder_path.lower():
                filter_type = 'gauss'
            else:
                raise ValueError("Only filter types box and gauss are supported for this function")
            
            # Compute residual kinetic energy
            K_r_DNS = 0.5*(DNS_field.U_X._3d**2 + DNS_field.U_Y._3d**2 + DNS_field.U_Z._3d**2) - 0.5*(self.U_X._3d**2 + self.U_Y._3d**2 + self.U_Z._3d**2)
            save_file(K_r_DNS, self.find_path("K_r_DNS"))
            del K_r_DNS    # release memory
            self.update()
            
        if mode=='Yosh':
            # Check that the Yoshizawa constant C_i is defined
            if not hasattr(self, 'Ci'):
                Warning("The field does not have an attribute Ci.\n The Yoshizawa model constant Ci will be initialized by default to 0.1")
                self.Ci = 0.1
            Ci = self.Ci
            K_r_Yosh = 2*Ci*self.RHO._3d*(self.filter_size*self.mesh.l)**2*self.S_LES._3d**2
            save_file(K_r_Yosh, self.find_path("K_r_Yosh"))
            del K_r_Yosh    # release memory
            with open(os.path.join(self.folder_path, 'C_I_Yoshizawa_model.txt'), 'w') as f:
                # Write the value of the C_I constant to the file
                f.write(str(Ci))
            
            self.update()
            
        return
    
    def compute_residual_dissipation_rate(self, mode='Smag'):
        """
            This function computes the residual dissipation rate for a filtered velocity field,
            based on the specified mode: 'DNS' or 'Smag'. It requires that the field has been 
            filtered and performs different calculations depending on the selected mode.
        
            Parameters:
            ----------
            mode : str, optional
                The mode of operation. It can be either 'Smag' or 'DNS'. Defaults to 'Smag'.
                
                - 'Smag': Uses the Smagorinsky model to compute the residual dissipation rate.
                - 'DNS': Uses Direct Numerical Simulation data to compute the residual dissipation rate.
        
            Returns:
            --------
            None:
                The function does not return any values but saves the computed residual dissipation rate 
                as a file in the main folder of the field.
        
            Raises:
            -------
            ValueError:
                - If the field is not a filtered field (i.e., `filter_size` is 1).
                - If the filter type used is not 'box' or 'gauss'.
                
            AttributeError:
                - If the 'DNS' mode is selected and the `DNS_folder_path` attribute is not set.
                - If the 'Smag' mode is selected and the `S_LES` attribute is not set.
                
            Warning:
                - If the 'Smag' mode is selected and the `Cs` attribute is not set, it initializes `Cs` to 0.1 by default.
        
            Detailed Description:
            ---------------------
            This function first updates the internal state of the field. It then checks the validity 
            of the provided mode against the allowed modes stored in the `variables` dictionary.
            
            If the field is not filtered (i.e., `filter_size` is 1), it raises a `ValueError` 
            indicating that residual quantities can only be computed for filtered fields and provides 
            instructions on how to filter the field.
        
            Depending on the mode, the function performs different computations:
            
            1. **DNS Mode**:
                - Ensures the `DNS_folder_path` attribute is set, raising an `AttributeError` if not.
                - Loads the associated unfiltered DNS field.
                - Determines the filter type (either 'box' or 'gauss') used for the folder to ensure consistency.
                - Computes the anisotropic residual stress tensor and then the residual dissipation rate using the 
                  filtered DNS field and the LES strain rate.
                - Saves the computed residual dissipation rate to a file.
        
            2. **Smag Mode**:
                - Checks if the `Cs` attribute is set, issuing a warning and initializing `Cs` to 0.1 if not.
                - Ensures the `S_LES` attribute is set, raising an `AttributeError` if not.
                - Computes the residual dissipation rate using the Smagorinsky model.
                - Saves the computed residual dissipation rate to a file.
            
            Finally, the function updates the internal state of the field again.
        """
        self.update()
        valid_modes = self.variables["Epsilon_r_{}"][2]
        check_input_string(mode, valid_modes, 'mode')
        
        if self.filter_size == 1:
            raise ValueError("The field is not a filtered field.\n"
                             "Computing residual quantities only makes sense for filtered fields."
                             "You can filter the entire field with the command:\n>>>your_field_object.filter(filter_size)"
                             "\nor:\n>>>your_field_object.filter_favre(filter_size)")
        
        
        if mode=='DNS':
            if not hasattr(self, 'DNS_folder_path'):
                raise AttributeError("The filtered field does not have a value to identify the associated unfiltered data.\n"
                                 "The path of the unfiltered data folder must be assigned with the following command:\n"
                                 ">>> your_filtered_field.DNS_folder_path = 'your_unfiltered_DNS_folder_path'")
            DNS_field = Field3D(self.DNS_folder_path)
            
            #--------- Compute Anisotropic Residual Stress Tensor ------------#
            # Check what filter was used for the folder and keep consistency
            if 'favre' in self.folder_path.lower():
                favre = True
            if 'box' in self.folder_path.lower():
                filter_type = 'box'
            elif 'gauss' in self.folder_path.lower():
                filter_type = 'gauss'
            else:
                raise ValueError("Only filter types box and gauss are supported for this function")
            
            # Compute epsilon_r = -tau_R_ij*S_ij
            direction = ['X', 'Y', 'Z']
            epsilon_r    = np.zeros(self.shape, dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    if j>=i:
                        # compute filtered(Ui*Uj)_DNS
                        file_name = self.find_path(f"TAU_r_{i+1}{j+1}_{mode}")
                        Ui_Uj_DNS = getattr(DNS_field, f'U_{direction[i]}')._3d * getattr(DNS_field, f'U_{direction[j]}')._3d
                        
                        if favre:
                            Ui_Uj_DNS = filter_3D(Ui_Uj_DNS, self.filter_size, 
                                                  self.RHO._3d, favre=True, 
                                                  filter_type=filter_type)
                        else:
                            Ui_Uj_DNS = filter_3D(Ui_Uj_DNS, self.filter_size, 
                                                  RHO=None, favre=False, 
                                                  filter_type=filter_type)
                            
                        Tau_r_ij = Ui_Uj_DNS - (getattr(self, f'U_{direction[i]}')._3d * getattr(self, f'U_{direction[j]}')._3d)
                        # TODO: check that this formulation is consistent for compressible flows
                        # with Favre averaging. Source: Poinsot pag 173 footnote
                        del Ui_Uj_DNS
                        epsilon_r       += -Tau_r_ij*getattr(self, f"S{i+1}{j+1}_LES")._3d
                        if i!=j:  # take into account the sub-diagonal element in the computation of the module
                            epsilon_r    += -Tau_r_ij*getattr(self, f"S{i+1}{j+1}_LES")._3d
                        del Tau_r_ij
            file_name=  self.find_path(f"Epsilon_r_{mode}")
            save_file(epsilon_r, file_name)
            
            
        if mode=='Smag':
            # Check that the smagorinsky constant is defined
            if not hasattr(self, 'Cs'):
                Warning("The field does not have an attribute Cs.\n The Smagorinsky constant Cs will be initialized by default to 0.1")
                self.Cs = 0.1
            Cs = self.Cs
            if not hasattr(self, 'S_LES'):
                raise AttributeError("The field does not have a value for the Strain rate at LES scale.\n"
                                 "The strain rate can be computed with the command:\n"
                                 ">>> your_filtered_field.compute_strain_rate()")
            epsilon_r = (Cs*self.filter_size*self.mesh.l)**2 * self.S_LES._3d**3
            file_name=  self.find_path(f"Epsilon_r_{mode}")
            save_file(epsilon_r, file_name)
            with open(os.path.join(self.folder_path, 'C_s_Smagorinsky_model.txt'), 'w') as f:
                # Write the value of the C_S constant to the file
                f.write(str(Cs))
        
        self.update()
            
    def compute_reaction_rates(self, n_chunks = 5000):
        """
        Computes the source terms for a given chemical reaction system.
        
        This function performs several steps:
        1. Checks that all the mass fractions are in the folder.
        2. Determines if the reaction rates to be computed are in DNS or LFR mode based on the filter size.
        3. Builds a list with reaction rates paths and one with the species' Mass fractions paths.
        4. Checks that the files of the reaction rates do not exist yet. If they do, asks the user if they want to overwrite them.
        5. Computes the reaction rates in chunks to handle large data sets efficiently.
        6. Saves the computed reaction rates, heat release rate, and dynamic viscosity to files.
        7. Updates the object's state.
        
        Parameters:
        n_chunks (int, optional): The number of chunks to divide the data into for efficient computation. Default is 5000.
        
        Returns:
        None
        
        Raises:
        SystemExit: If the user chooses not to overwrite existing reaction rate files, or if there is a mismatch in the number of species and the length of the species paths list.
        
        Note:
        This function uses the Cantera library to compute the reaction rates, heat release rate, and dynamic viscosity. It assumes that the object has the following attributes: attr_list, bool_list, folder_path, filter_size, species, shape, kinetic_mechanism, T, P, and paths_list. It also assumes that the object has the following methods: find_path and update.
        """
        # Step 1: Check that all the mass fractions are in the folder
        check_mass_fractions(self.attr_list, self.bool_list, self.folder_path)
        # Step 2: Understand if the reaction rates to be computed are in DNS or LFR mode
        if self.filter_size == 1:
            mode = 'DNS'
        else:
            mode = 'LFR'
            
        # Step 4: build a list with reaction rates paths and one with the species' Mass fractions paths
        reaction_rates_paths = []
        for attr, path in zip(self.attr_list, self.paths_list):
            if attr.startswith('R'):
                if mode in attr:
                    reaction_rates_paths.append(path)
        species_paths = []
        for attr, path in zip(self.attr_list, self.paths_list):
            if attr.startswith('Y'):
                species_paths.append(path)
                
        if (len(species_paths)!=len(self.species)) or(len(reaction_rates_paths)!=len(self.species)):
            raise ValueError("Lenght of the lists must be equal to the number of species. "
                             "Check that all the species molar concentrations and Reaction Rates are in the data folder."
                             "\nYou can compute the reaction rates with the command:"
                             "\n>>> your_filt_field.compute_reaction_rates()"
                             "\n\nOperation aborted.")
        
        # Step 3: Check that the files of the reaction rates do not exist yet
        temp = check_reaction_rates(self.attr_list, self.bool_list, self.folder_path) # at this point it's probably redundant, cause I already have the reaction_rates_path list so I could use that
        if temp:
            user_input = input(
                    f"The folder '{self.data_path}' already contains the reaction rates. "
                    f"\nThis operation will overwrite the content of the folder. "
                    f"\nDo you want to continue? ([yes]/no): "
                                )
            if user_input.lower() != "yes":
                print("Operation aborted.")
                sys.exit()
            else:
                for path in reaction_rates_paths:
                    os.remove(path)
                os.remove(self.find_path('Mu'))
                os.remove(self.find_path(f'HRR_{mode}'))
            # TODO: check that the files do not exist and if they exist I have to 
            # delete them, if not the code will append the values computed
            # to the existing files
                    
        # Step 5: Compute reaction rates
        chunk_size = self.shape[0] * self.shape [1] * self.shape[2] // n_chunks
        gas = ct.Solution(self.kinetic_mechanism)
        
        # Open output files in writing mode
        output_files_R = [open(reaction_rate_path, 'ab') for reaction_rate_path in reaction_rates_paths]
        if mode == 'DNS':
            HRR_path = self.find_path('HRR_DNS')
        if mode =='LFR':
            HRR_path = self.find_path('HRR_LFR')
        output_file_HRR = open(HRR_path, 'ab')
        Mu_path = self.find_path('Mu')
        output_file_Mu = open(Mu_path, 'ab')
        
        # create generators to read files in chunks
        T_chunk_generator = read_variable_in_chunks(self.T.path, chunk_size)
        P_chunk_generator = read_variable_in_chunks(self.P.path, chunk_size)
        species_chunk_generator = [read_variable_in_chunks(specie_path, chunk_size) for specie_path in species_paths]
        print('Reading file in chunks: read 0/{}'.format(n_chunks))
        for i in range(n_chunks):
            T_chunk = next(T_chunk_generator)  # Read one step of this function
            P_chunk = next(P_chunk_generator)
            # Read a chunk for every specie
            Y_chunk = [next(generator) for generator in species_chunk_generator]
            Y_chunk = np.array(Y_chunk)  # Make the list an array
            # Initialize R for the source Terms and HRR
            R_chunk = np.zeros_like(Y_chunk)
            HRR_chunk = np.zeros_like(T_chunk)
            Mu_chunk = np.zeros_like(T_chunk) #if it's a scalar I use T_chunk as a reference size
            
            # iterate through the chunks and compute the Reaction Rates
            for j in range(len(T_chunk)):
                gas.TPY = T_chunk[j], P_chunk[j], Y_chunk[:, j]
                R_chunk[:, j] = gas.net_production_rates * gas.molecular_weights
                HRR_chunk[j] = gas.heat_release_rate 
                Mu_chunk[j] = gas.viscosity # dynamic viscosity, Pa*s
            
            # Save files
            save_file(HRR_chunk, output_file_HRR)
            save_file(Mu_chunk, output_file_Mu)
            R_chunk = R_chunk.tolist()
            for k in range(len(self.species)):
                save_file(np.array(R_chunk[k]), output_files_R[k])
            
            # Print advancement state
            print('Reading file in chunks: read {}/{}'.format(i + 1, n_chunks))

        # Close all output files
        for output_file in output_files_R:
            output_file.close()
        output_file_HRR.close()
        output_file_Mu.close()
        
        self.update()
        
        return
    
    def compute_reaction_rates_batch(self, n_chunks = 5000, tau_c='SFR', tau_m='Kolmo'):
        '''
        Computes the reaction rates in batches for a filtered field.
    
        Description:
        ------------
        This method calculates reaction rates in chunks for a filtered field, suitable for large datasets. The reaction rates can be computed in different modes specified by `tau_c` and `tau_m`.
    
        Parameters:
        -----------
        n_chunks : int, optional
            Number of chunks to divide the field into for batch processing. Default is 5000.
        tau_c : str, optional
            Mode for computing the chemical timescale. Default is 'SFR'.
        tau_m : str, optional
            Mode for computing the mixing timescale. Default is 'Kolmo'.
    
        Raises:
        -------
        ValueError
            If the field is not filtered or if the species' molar concentrations and reaction rates are not in the data folder.
        AttributeError
            If required attributes (`attr_list`, `bool_list`, `folder_path`) are not defined.
        
        Workflow:
        ---------
        1. Validation
           - Ensures all mass fractions are in the folder.
           - Confirms the field is filtered and sets the mode to 'Batch'.
    
        2. Paths Preparation
           - Builds lists for reaction rates paths and species' mass fractions paths.
           - Checks if the reaction rates files already exist and prompts the user for confirmation to overwrite them.
    
        3. Reaction Rates Computation
           - Initializes chunk generators for temperature, pressure, density, and timescales.
           - Computes reaction rates and heat release rates in chunks using the Cantera library.
           - Saves computed chunks to respective files.
    
        Returns:
        --------
        None
    
        Example:
        --------
        >>> field = Field3D('your_folder_path')
        >>> field.compute_reaction_rates_batch(n_chunks=5000, tau_c='SFR', tau_m='Kolmo')
        '''
        # Step 1: Check that all the mass fractions are in the folder
        check_mass_fractions(self.attr_list, self.bool_list, self.folder_path)
        # Step 2: Understand if the reaction rates to be computed are in DNS or LFR mode
        if self.filter_size == 1:
            raise ValueError("The field is not filtered. This closure is only applicable to filtered fields.")
        else:
            mode = 'Batch'
            
        # Step 4: build a list with reaction rates paths and one with the species' Mass fractions paths
        reaction_rates_paths = []
        for attr, path in zip(self.attr_list, self.paths_list):
            if attr.startswith('R'):
                if mode in attr:
                    reaction_rates_paths.append(path)
        species_paths = []
        for attr, path in zip(self.attr_list, self.paths_list):
            if attr.startswith('Y'):
                species_paths.append(path)
                
        if (len(species_paths)!=len(self.species)) or(len(reaction_rates_paths)!=len(self.species)):
            raise ValueError("Lenght of the lists must be equal to the number of species. "
                             "Check that all the species molar concentrations and Reaction Rates are in the data folder."
                             "\nYou can compute the reaction rates with the command:"
                             "\n>>> your_filt_field.compute_reaction_rates()"
                             "\n\nOperation aborted.")
        
        # Step 3: Check that the files of the reaction rates do not exist yet
        temp = check_reaction_rates(self.attr_list, self.bool_list, self.folder_path) # at this point it's probably redundant, cause I already have the reaction_rates_path list so I could use that
        if temp:
            user_input = input(
                    f"The folder '{self.data_path}' already contains the reaction rates. "
                    f"\nThis operation will overwrite the content of the folder. "
                    f"\nDo you want to continue? ([yes]/no): "
                                )
            if user_input.lower() != "yes":
                print("Operation aborted.")
                sys.exit()
            else:
                for path in reaction_rates_paths:
                    os.remove(path)
                os.remove(self.find_path(f'HRR_{mode}'))
            # TODO: check that the files do not exist and if they exist I have to 
            # delete them, if not the code will append the values computed
            # to the existing files
                    
        # Step 5: Compute reaction rates
        chunk_size = self.shape[0] * self.shape [1] * self.shape[2] // n_chunks
        gas = ct.Solution(self.kinetic_mechanism)
        
        # Open output files in writing mode
        output_files_R = [open(reaction_rate_path, 'ab') for reaction_rate_path in reaction_rates_paths]
        HRR_path = self.find_path(f'HRR_{mode}')
        output_file_HRR = open(HRR_path, 'ab')
        
        # create generators to read files in chunks
        T_chunk_generator       = read_variable_in_chunks(self.T.path, chunk_size)
        P_chunk_generator       = read_variable_in_chunks(self.P.path, chunk_size)
        RHO_chunk_generator     = read_variable_in_chunks(self.RHO.path, chunk_size)
        Tau_m_chunk_generator   = read_variable_in_chunks(self.find_path(f'Tau_m_{tau_m}'), chunk_size)
        Tau_c_chunk_generator   = read_variable_in_chunks(self.find_path(f'Tau_c_{tau_c}'), chunk_size)
        species_chunk_generator = [read_variable_in_chunks(specie_path, chunk_size) for specie_path in species_paths]
        print('Reading file in chunks: read 0/{}'.format(n_chunks))
        for i in range(n_chunks):
            T_chunk     = next(T_chunk_generator)  # Read one step of this function
            P_chunk     = next(P_chunk_generator)
            RHO_chunk   = next(RHO_chunk_generator)
            Tau_m_chunk = next(Tau_m_chunk_generator)
            Tau_c_chunk = next(Tau_c_chunk_generator)
            # Read a chunk for every specie
            Y_chunk = [next(generator) for generator in species_chunk_generator]
            Y_chunk = np.array(Y_chunk)  # Make the list an array
            # Initialize R for the source Terms and HRR
            R_chunk = np.zeros_like(Y_chunk)
            HRR_chunk = np.zeros_like(T_chunk)
            Mu_chunk = np.zeros_like(T_chunk) #if it's a scalar I use T_chunk as a reference size
            
            # iterate through the chunks and compute the Reaction Rates
            for j in range(len(T_chunk)):
                gas.TPY  = T_chunk[j], P_chunk[j], Y_chunk[:, j]
                tau_star = np.minimum(Tau_c_chunk[j], Tau_m_chunk[j])
                
                Y0       = gas.Y
                h0       = gas.enthalpy_mass # Specific enthalpy [J/kg].
                
                reactor  = ct.IdealGasReactor(gas)
                sim      = ct.ReactorNet([reactor])
                t_start  = 0
                t_end    = tau_star
                
                # integrate the batch reactor in time
                while t_start < t_end:
                    t_start = sim.step()
                
                Ystar    = gas.Y
                hstar    = gas.enthalpy_mass # Specific enthalpy [J/kg].
                
                R_chunk[:, j] = RHO_chunk[j] / tau_star * (Ystar - Y0)
                HRR_chunk[j]  = RHO_chunk[j] / tau_star * (hstar - h0)
                
            # Save files
            save_file(HRR_chunk, output_file_HRR)
            R_chunk = R_chunk.tolist()
            for k in range(len(self.species)):
                save_file(np.array(R_chunk[k]), output_files_R[k])
            
            # Print advancement state
            print('Reading file in chunks: read {}/{}'.format(i + 1, n_chunks))

        # Close all output files
        for output_file in output_files_R:
            output_file.close()
        output_file_HRR.close()
        
        self.update()
        
        return
    
    def compute_strain_rate(self, save_derivatives=False, save_tensor=True, verbose=False):
        """
        Computes the strain rate or the derivatives of the velocity components (U, V, W) over a 3D mesh.
    
        Parameters:
        ----------
        save_derivatives : bool, optional
            If True, saves the derivatives of the velocity components as files in the main folder. Defaults to False.
            
        save_tensor : bool, optional
            If True, saves the strain rate tensor as a file in the main folder. Defaults to True.
        
        verbose : bool, optional
            If True, prints out progress information. Defaults to False.
    
        Returns:
        --------
        None
    
        Raises:
        -------
        TypeError: 
            If U, V, W are not instances of Scalar3D or if mesh is not an instance of Mesh3D.
        ValueError: 
            If U, V, W and mesh do not have the same shape.
    
        Workflow:
        ---------
        1. Preprocess
           - Sets the closure type based on the filter size.
           - Retrieves necessary attributes and initializes variables.
    
        2. Compute Derivatives
           - Computes derivatives of the velocity components in all three directions.
           - Saves derivatives as files if `save_derivatives` is True.
    
        3. Compute Strain Rate Tensor
           - Computes the strain rate tensor components by averaging appropriate velocity component derivatives.
           - Saves the strain rate tensor as a file if `save_tensor` is True.
    
        Example:
        --------
        >>> field = Field3D('your_folder_path')
        >>> field.compute_strain_rate(save_derivatives=True, save_tensor=True, verbose=True)
        """
        
        if self.filter_size==1:
            closure = 'DNS'
        else:
            closure = 'LES'
        
        shape = self.shape
        filter_size = self.filter_size
        mesh = self.mesh
        
        # define the list to use to change the file name
        path_list = self.U_X.path.split('/')
        if hasattr(self.U_X, 'file_id'):
            file_id = self.U_X.file_id
        else:
            file_id = ''
            
        axes = ['X', 'Y', 'Z']
        for i in range(3): # index for the velocity component
            for j in range(3): # index for the derivative direction
                file_name = self.find_path("dU{}_dX{}_{}".format(i+1,j+1, closure))
                # Check if the file already exists
                if not os.path.exists(file_name):
                    if verbose:
                        print(f"Computing dU_{axes[i].lower()}/d{axes[j].lower()}...")
                    U = getattr(self, f"U_{axes[i]}")
                    if j == 0:
                        dU_dx = gradient_x(U, mesh, filter_size)
                    if j == 1:
                        dU_dx = gradient_y(U, mesh, filter_size)
                    if j == 2:
                        dU_dx = gradient_z(U, mesh, filter_size)
                    save_file(dU_dx, file_name)
                    del dU_dx
                else:
                    if verbose:
                        print("File {} already exists".format(file_name))
        self.update(verbose=verbose)
        
        #------------------------ Compute Strain Rate ----------------------------#
        for i in range(3):
            for j in range(3):
                if j>=i:
                    file_name = self.find_path(f"S{i+1}{j+1}_{closure}")
                    der1 = f"dU{i+1}_dX{j+1}_{closure}"
                    der2 = f"dU{j+1}_dX{i+1}_{closure}"
                    S = 0.5*( getattr(self, der1).value + getattr(self, der2).value )
                    save_file(S, file_name)
        
        #cancel files with the derivatives
        if not save_derivatives:
            for i in range(3): # index for the velocity component
                for j in range(3): # index for the derivative direction
                    file_name = self.find_path("dU{}_dX{}_{}".format(i+1,j+1, closure))
                    delete_file(file_name)
        
        self.update(verbose=verbose)
        
        S = np.zeros(shape)
        for i in range(3): # index for the velocity component
            for j in range(3): # index for the derivative direction
                if j>=i:
                    attr_name = "S{}{}_{}".format(i+1,j+1, closure)
                    file_name = self.find_path(attr_name)
                    temp = 2*(getattr(self, attr_name)._3d**2)
                    if i!=j:
                        temp = temp*2 # takes into account the sub-diagonal of the symmetric tensor
                    S += temp
                    
                    # cancel the files with the tensor if required
                    if save_tensor == False:
                        delete_file(file_name)
        S = np.sqrt(S) # square root of the sum of 2*Sij*Sij
        
        # Save file
        file_name     = self.find_path('S_{}'.format(closure))
        save_file(S, file_name)
        
            
        self.update(verbose=True)
        return
    
    def compute_tau_r(self, mode='Smag', save_tensor_components=True):
        '''
        Computes the anisotropic part of the residual stress tensor, denoted as \(\tau_r\), 
        for a given field in computational fluid dynamics simulations. The function can 
        operate in two modes: 'Smag' and 'DNS'.
        
        Description:
        ------------
        $\(\tau_r\)$ (TAU_r) is the **anisotropic part** of the residual stress tensor.
        
        Residual stress tensor:
        \[
        \tau^R_{i,j} = \widetilde{(U_i U_j)} - \widetilde{U}_i \cdot \widetilde{U}_j
        \]
        
        Anisotropic part:
        \[
        \tau^r_{i,j} = \tau^R_{i,j} - \frac{2}{3} k_r \cdot \delta_{i,j}
        \]
        
        where \( k_r \) is the residual kinetic energy:
        \[
        k_r = \frac{1}{2} \left( \widetilde{(U_i U_i)} - \widetilde{U}_i \cdot \widetilde{U}_i \right) = \frac{1}{2} \left( \widetilde{(U_i^2)} - \left(\widetilde{U}_i \right)^2 \right)
        \]
    
        Parameters:
        -----------
        mode : str, optional
            Mode of operation, either 'Smag' for the Smagorinsky model or 'DNS' for 
            Direct Numerical Simulation data. Default is 'Smag'.
    
        Raises:
        -------
        ValueError
            If the field is not a filtered field (i.e., `self.filter_size == 1`).
    
        AttributeError
            If required attributes (`Cs`, `S_LES`, `DNS_folder_path`) are not defined.
    
        Returns:
        --------
        None
    
        Workflow:
        ---------
        1. Initial Setup and Validation
           - The function starts by updating the field and checking if the field is filtered.
           - If `self.filter_size == 1`, it raises a `ValueError` because residual quantities computation only makes sense for filtered fields.
        
            2. Mode: 'Smag' (Smagorinsky Model)
               - Turbulent Viscosity:
                 - Checks if the Smagorinsky constant (`Cs`) is defined. If not, it initializes `Cs` to 0.1.
                 - Computes the turbulent viscosity (\(\mu_t\)) using:
                   $ \mu_t = (Cs \cdot \Delta \cdot l)^2 \cdot S_{LES} $
                   where \(\Delta\) is the filter size, \(l\) is the grid size, and \(S_{LES}\) is the strain rate at LES scale.
               - Anisotropic Residual Stress Tensor:
                 - Initializes `Tau_r` as a zero matrix.
                 - For each component \((i, j)\) of the tensor:
                   - Computes \( \tau^r_{ij} = -2\mu_t S_{ij}^{LES} \).
                   - Adjusts for compressibility by subtracting the isotropic part (\(S_{iso}\)) when \(i = j\).
                   - Computes the squared components and accumulates them.
                   - Saves the computed \(\tau^r_{ij}\) to a file.
        
            3. Mode: 'DNS' (Direct Numerical Simulation)
               - DNS Data Setup:
                 - Checks if the path to DNS data is defined.
                 - Initializes a `DNS_field` object to read DNS data.
                 - Determines the type of filter used (box or Gaussian).
               - Residual Kinetic Energy:
                 - Computes residual kinetic energy \( K_r^{DNS} \) as:
                   \[ K_r^{DNS} = 0.5 \left( U_x^2 + U_y^2 + U_z^2 \right)_{DNS} - 0.5 \left( U_x^2 + U_y^2 + U_z^2 \right) \]
                 - Saves \( K_r^{DNS} \) to a file.
               - Anisotropic Residual Stress Tensor:
                 - Initializes `Tau_r` as a zero matrix.
                 - For each component \((i, j)\) of the tensor:
                   - Computes the filtered product \(\widetilde{(U_i U_j)}_{DNS}\).
                   - Calculates \(\tau^r_{ij}\) as:
                     \[ \tau^r_{ij} = \widetilde{(U_i U_j)}_{DNS} - \widetilde{U}_i \widetilde{U}_j - \delta_{ij} \frac{2}{3} K_r^{DNS} \]
                   - Computes the squared components and accumulates them.
                   - Saves the computed \(\tau^r_{ij}\) to a file.
        '''
        self.update()
        valid_modes = self.variables["TAU_r_{}{}_{}"][2]
        
        # Check that the field is a filtered field, if not it does not make sense
        # to compute the closure for the residual quantities
        if self.filter_size == 1:
            raise ValueError("The field is not a filtered field.\n"
                             "Computing residual quantities only makes sense for filtered fields."
                             "You can filter the entire field with the command:\n>>>your_field_object.filter(filter_size)"
                             "\nor:\n>>>your_field_object.filter_favre(filter_size)")
        
        if mode == 'Smag':
            
            #----------------- Compute Turbulent Viscosity -------------------#
            # Check that the smagorinsky constant is defined
            if not hasattr(self, 'Cs'):
                Warning("The field does not have an attribute Cs.\n The Smagorinsky constant Cs will be initialized by default to 0.1")
                self.Cs = 0.1
            Cs = self.Cs
            if not hasattr(self, 'S_LES'):
                raise AttributeError("The field does not have a value for the Strain rate at LES scale.\n"
                                 "The strain rate can be computed with the command:\n"
                                 ">>> your_filtered_field.compute_strain_rate()")
            nu_t = (Cs*self.filter_size*self.mesh.l)**2 * self.S_LES._3d
            # I multiply delta(filter amplitude expressed in number of cells) by l that is the grid size in meters
            S_iso = 1/3*(self.S11_LES._3d+self.S22_LES._3d+self.S33_LES._3d)
            
            #--------- Compute Anisotropic Residual Stress Tensor ------------#
            Tau_r    = np.zeros(self.shape, dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    if j>=i:
                        file_name = self.find_path(f"TAU_r_{i+1}{j+1}_{mode}")
                        # S_ij    = getattr(self, "S{}{}_{}".format(i+1,j+1, mode))._3d
                        Tau_r_ij  = -2*nu_t*getattr(self, "S{}{}_LES".format(i+1,j+1))._3d  # TODO: check that it is always fine using the value at LES level
                        if i==j:
                            Tau_r_ij -= -2*nu_t*S_iso  #Take into account compressibility subtracting the trace of S
                            # See Poinsot pag 173 footnote
                        Tau_r    += 2*(Tau_r_ij**2)
                        if i!=j:  # take into account the sub-diagonal element in the computation of the module
                            Tau_r    += 2*(Tau_r_ij**2) 
                            
                        if save_tensor_components:
                            save_file(Tau_r_ij, file_name)
                        del Tau_r_ij
            Tau_r    =  np.sqrt(Tau_r)
            file_name=  self.find_path(f"TAU_r_{mode}")
            save_file(Tau_r, file_name)
            # Save the value of the constant Cs
            with open(os.path.join(self.folder_path, 'C_s_Smagorinsky_model.txt'), 'w') as f:
                # Write the value of the C_S constant to the file
                f.write(str(Cs))
            self.update()
            
        if mode == 'DNS':
            
            if not hasattr(self, 'DNS_folder_path'):
                raise AttributeError("The filtered field does not have a value to identify the associated unfiltered data.\n"
                                 "The path of the unfiltered data folder must be assigned with the following command:\n"
                                 ">>> your_filtered_field.DNS_folder_path = 'your_unfiltered_DNS_folder_path'")
            DNS_field = Field3D(self.DNS_folder_path)
            
            #--------- Compute Anisotropic Residual Stress Tensor ------------#
            # Check what filter was used for the folder and keep consistency
            if 'favre' in self.folder_path.lower():
                favre = True
            if 'box' in self.folder_path.lower():
                filter_type = 'box'
            elif 'gauss' in self.folder_path.lower():
                filter_type = 'gauss'
            else:
                raise ValueError("Only filter types box and gauss are supported for this function")
            
            # Compute residual kinetic energy
            K_r_DNS = 0.5*(DNS_field.U_X._3d**2 + DNS_field.U_Y._3d**2 + DNS_field.U_Z._3d**2) - 0.5*(self.U_X._3d**2 + self.U_Y._3d**2 + self.U_Z._3d**2)
            save_file(K_r_DNS, self.find_path("K_r_DNS"))
            del K_r_DNS    # release memory
            self.update()
            
            # Compute tau_r
            direction = ['X', 'Y', 'Z']
            Tau_r    = np.zeros(self.shape, dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    if j>=i:
                        # Dirac's delta
                        if i==j:
                            delta_dirac=1
                        else:
                            delta_dirac=0
                        # compute filtered(Ui*Uj)_DNS
                        file_name = self.find_path(f"TAU_r_{i+1}{j+1}_{mode}")
                        Ui_Uj_DNS = getattr(DNS_field, f'U_{direction[i]}')._3d * getattr(DNS_field, f'U_{direction[j]}')._3d
                        
                        if favre:
                            Ui_Uj_DNS = filter_3D(Ui_Uj_DNS, self.filter_size, 
                                                  self.RHO._3d, favre=True, 
                                                  filter_type=filter_type)
                        else:
                            Ui_Uj_DNS = filter_3D(Ui_Uj_DNS, self.filter_size, 
                                                  RHO=None, favre=False, 
                                                  filter_type=filter_type)
                            
                        Tau_r_ij = Ui_Uj_DNS - (getattr(self, f'U_{direction[i]}')._3d * getattr(self, f'U_{direction[j]}')._3d)
                        # TODO: check that this formulation is consistent for compressible flows
                        # with Favre averaging. Source: Poinsot pag 173 footnote
                        del Ui_Uj_DNS
                        Tau_r_ij -= delta_dirac*2/3*self.K_r_DNS._3d
                        Tau_r    += 2*(Tau_r_ij**2)
                        if i!=j:  # take into account the sub-diagonal element in the computation of the module
                            Tau_r    += 2*(Tau_r_ij**2) 
                        if save_tensor_components:
                            save_file(Tau_r_ij, file_name)
                        del Tau_r_ij
            Tau_r    =  np.sqrt(Tau_r)
            file_name=  self.find_path(f"TAU_r_{mode}")
            save_file(Tau_r, file_name)
            
            self.update()
        
        return
    
    
    def compute_velocity_module(self):
        """
        Computes the velocity module and saves it to a file.
    
        Description:
        ------------
        This method calculates the velocity module by squaring the values of `U_X`, `U_Y`, and `U_Z`, 
        summing them up, and then taking the square root of the result. The computed velocity 
        module is then saved to a file using the `save_file` function. The file path is determined 
        by the `find_path` method with 'U' as the argument. After saving the file, the `update` 
        method is called to refresh the attributes of the class.
    
        Note: 
        -----
        - `self.U_X`, `self.U_Y`, and `self.U_Z` are assumed to be attributes of the class 
          representing components of velocity. Make sure to check you have the relative files in your
          data folder. To check, use the method <your_field_name>.print_attributes. 
    
        Parameters:
        -----------
        None
    
        Returns:
        --------
        None
        """
        if self.filter_size==1:
            closure = 'DNS'
        else:
            closure = 'LES'
            
        U  = self.U_X.value**2
        U += self.U_Y.value**2
        U += self.U_Z.value**2
        U  = np.sqrt(U)
        save_file(U, self.find_path('U_{}'.format(closure)))
        
        self.update()
        return
    
    def cut(self, cut_size, mode='xyz'):
        """
        Cut a field into smaller sections based on a specified cut size.
    
        Description:
        ------------
        This method cuts a field into smaller sections based on the specified cut size and mode. 
        It creates a new folder to store the cut data and grid files. If the folder already exists, 
        it prompts the user for confirmation before overwriting the content. The chemical path is 
        copied to the new cut folder. Each attribute of the field is cut according to the specified 
        cut size and mode, and the resulting sections are saved to files. Finally, the information 
        file ('info.json') is updated with the new shape of the field.
    
        Parameters:
        -----------
        cut_size : int
            The size of the cut.
        mode : str, optional
            The mode of cutting. Default is 'xyz'.
    
        Returns:
        --------
        str
            Path of the cut folder.
    
        Note:
        -----
        Add different cutting modes to the function
        """
        print("\n---------------------------------------------------------------")
        print (f"Cutting Field '{self.folder_path}'...")
        
        cut_folder_path = self.folder_path+'_cut'
        cut_data_path   = os.path.join(cut_folder_path, folder_structure["data_path"])
        cut_grid_path   = os.path.join(cut_folder_path, folder_structure["grid_path"])    
        cut_chem_path   = os.path.join(cut_folder_path, folder_structure["chem_path"])  
        
        if not os.path.exists(cut_folder_path):
            os.makedirs(cut_folder_path)
        else:
            user_input = input("The folder already exists. This operation will overwrite the content of the folder. Do you want to continue? (yes/no): ")
            if user_input.lower() != "yes":
                print("Operation aborted.")
                sys.exit()
            else:
                pass
        if not os.path.exists(cut_data_path):
            os.makedirs(cut_data_path)
        if not os.path.exists(cut_grid_path):
            os.makedirs(cut_grid_path)
        if not os.path.exists(cut_chem_path):
            shutil.copytree(self.chem_path, cut_chem_path)
            
        for attribute, file_path, is_present in zip(self.attr_list, self.paths_list, self.bool_list):
            if is_present:
                file_name = os.path.basename(file_path)
                cut_path  = os.path.join(cut_data_path, file_name)
                
                scalar = getattr(self, attribute)
                scalar_cut = scalar.cut(n_cut=cut_size, mode=mode)
                
                save_file(scalar_cut, cut_path)
                
        new_shape = scalar_cut.shape
        
        info = self.info
        info['global']['Nxyz'] = new_shape
        
        with open(os.path.join(cut_folder_path, 'info.json'), "w") as json_file:
            json.dump(info, json_file)
            
        for attribute in ['X', 'Y', 'Z']:
            scalar = getattr(self.mesh, attribute)
            file_name = os.path.basename(scalar.path)
            cut_path  = os.path.join(cut_grid_path, file_name)
            scalar_cut = scalar.cut(n_cut=cut_size, mode=mode)
            
            save_file(scalar_cut, cut_path)
        
        print (f"Done cutting Field '{self.folder_path}'.")
        
        return cut_folder_path
        
    def filter_favre(self, filter_size, filter_type='Gauss'):
        """
        Filter every scalar in the field object using Favre-averaging.
        
        Description:
        ------------
        This method filters a field using the Favre-averaged filtering technique with the specified 
        filter size and type. It creates a new folder to store the filtered data and grid files. If 
        the folder already exists, it prompts the user for confirmation before overwriting the content. 
        The chemical path and information file ('info.json') are copied to the new filtered folder. 
        Each attribute of the field is filtered according to the specified filter size and type, and 
        the resulting filtered sections are saved to files.
    
        Parameters:
        -----------
        filter_size : int
            The size of the filter.
        filter_type : str, optional
            The type of filter to use. Default is 'gauss'.
    
        Raises:
        -------
        TypeError
            If filter_size is not an integer.
            If filter_type is not a string.
        ValueError
            If filter_type is not one of the valid options.
    
        Returns:
        --------
        str
            Path of the filtered field folder.
    
        Example:
        --------
        >>> field = Field(folder_path='../data/field1')
        >>> filtered_folder_path = field.filter_favre(filter_size=5)
        Filtering Field '../data/field1'...
        Done Filtering Field '../data/field1'.
        Filtered Field path: '../data/Filter5Favre'
        """
        print("\n---------------------------------------------------------------")
        print (f"Filtering Field '{self.folder_path}'...")
        
        valid_filter_types = ['gauss', 'box']
        
        if not isinstance(filter_size, int):
            raise TypeError("filter_size must be an integer")
        if not isinstance(filter_type, str):
            raise TypeError("filter_type must be a string")
        check_input_string(filter_type, valid_filter_types, 'filter_type')
        
        filt_folder_name = f"Filter{filter_size}Favre{filter_type.capitalize()}"
        filt_folder_path = change_folder_name(self.folder_path, filt_folder_name)
        filt_data_path   = os.path.join(filt_folder_path, folder_structure["data_path"])
        filt_grid_path   = os.path.join(filt_folder_path, folder_structure["grid_path"])    
        filt_chem_path   = os.path.join(filt_folder_path, folder_structure["chem_path"])  
        
        # check if the destination directory already exists
        # TODO: a better handling of this situation would be to check before 
        # filtering if the file already exists in the destination folder, and
        # if it does, leave there the old one.
        if not os.path.exists(filt_folder_path):
            os.makedirs(filt_folder_path)
        else:
            user_input = input(f"The folder '{filt_folder_path}' already exists. This operation will overwrite the content of the folder. Do you want to continue? (yes/no): ")
            if user_input.lower() != "yes":
                print("Operation aborted.")
                sys.exit()
            else:
                pass
        if not os.path.exists(filt_data_path):
            os.makedirs(filt_data_path)
        if not os.path.exists(filt_grid_path):
            shutil.copytree(self.grid_path, filt_grid_path)
        if not os.path.exists(filt_chem_path):
            shutil.copytree(self.chem_path, filt_chem_path)
        if not os.path.exists(os.path.join(filt_folder_path, 'info.json')):
            shutil.copy(os.path.join(self.folder_path, 'info.json'), os.path.join(filt_folder_path, 'info.json'))
        
        if filter_type.lower() == 'gauss':
            RHO_filt = filter_gauss(self.RHO._3d, filter_size)
        elif filter_type.lower() == 'box':
            RHO_filt = filter_box(self.RHO._3d, filter_size)
        else:
            raise ValueError("Check the filter_type input value")
        
        for attribute, file_path, is_present in zip(self.attr_list, self.paths_list, self.bool_list):
            if is_present:
                file_name = os.path.basename(file_path)
                filt_path  = os.path.join(filt_data_path, file_name)
                
                scalar = getattr(self, attribute)
                scalar_filt = scalar._3d * self.RHO._3d
                if filter_type.lower() == 'gauss':
                    scalar_filt = filter_gauss(scalar_filt, delta=filter_size)
                elif filter_type.lower() == 'box':
                    scalar_filt = filter_box(scalar_filt, delta=filter_size)
                scalar_filt = scalar_filt / RHO_filt
                
                save_file(scalar_filt, filt_path)
                
        print (f"Done Filtering Field '{self.folder_path}'.")
        print (f"Filtered Field path: '{filt_folder_path}'.")
        
        return filt_folder_path
    
    def filter(self, filter_size, filter_type='Gauss'):
        """
        Filter every scalar in the field object.
        
        Description:
        ------------
        This method filters a field using the specified filter size and type. It creates a new folder 
        to store the filtered data and grid files. If the folder already exists, it prompts the user for 
        confirmation before overwriting the content. The chemical path and information file ('info.json') 
        are copied to the new filtered folder. Each attribute of the field is filtered according to the 
        specified filter size and type, and the resulting filtered sections are saved to files.
    
        Parameters:
        -----------
        filter_size : int
            The size of the filter.
        filter_type : str, optional
            The type of filter to use. Default is 'gauss'.
    
        Raises:
        -------
        TypeError
            If filter_size is not an integer.
            If filter_type is not a string.
        ValueError
            If filter_type is not one of the valid options.
    
        Returns:
        --------
        str
            Path of the filtered field folder.
    
        Example:
        --------
        >>> field = Field(folder_path='../data/field1')
        >>> filtered_folder_path = field.filter(filter_size=5)
        Filtering Field '../data/field1'...
        Done Filtering Field '../data/field1'.
        Filtered Field path: '../data/Filter5Gauss'
        """
        print("\n---------------------------------------------------------------")
        print (f"Filtering Field '{self.folder_path}'...")
        
        valid_filter_types = ['gauss', 'box']
        
        if not isinstance(filter_size, int):
            raise TypeError("filter_size must be an integer")
        if not isinstance(filter_type, str):
            raise TypeError("filter_type must be a string")
        check_input_string(filter_type, valid_filter_types, 'filter_type')
        
        filt_folder_name = f"Filter{filter_size}{filter_type.capitalize()}"
        filt_folder_path = change_folder_name(self.folder_path, filt_folder_name)
        filt_data_path   = os.path.join(filt_folder_path, folder_structure["data_path"])
        filt_grid_path   = os.path.join(filt_folder_path, folder_structure["grid_path"])    
        filt_chem_path   = os.path.join(filt_folder_path, folder_structure["chem_path"])  
        
        if not os.path.exists(filt_folder_path):
            os.makedirs(filt_folder_path)
        else:
            user_input = input(f"The folder '{filt_folder_path}' already exists. This operation will overwrite the content of the folder. Do you want to continue? (yes/no): ")
            if user_input.lower() != "yes":
                print("Operation aborted.")
                sys.exit()
            else:
                pass
        if not os.path.exists(filt_data_path):
            os.makedirs(filt_data_path)
        if not os.path.exists(filt_grid_path):
            shutil.copytree(self.grid_path, filt_grid_path)
        if not os.path.exists(filt_chem_path):
            shutil.copytree(self.chem_path, filt_chem_path)
        if not os.path.exists(os.path.join(filt_folder_path, 'info.json')):
            shutil.copy(os.path.join(self.folder_path, 'info.json'), os.path.join(filt_folder_path, 'info.json'))
        
        for attribute, file_path, is_present in zip(self.attr_list, self.paths_list, self.bool_list):
            if is_present:
                file_name = os.path.basename(file_path)
                filt_path  = os.path.join(filt_data_path, file_name)
                
                scalar = getattr(self, attribute)
                scalar_filt = scalar._3d
                if filter_type.lower() == 'gauss':
                    scalar_filt = filter_gauss(scalar_filt, delta=filter_size)
                elif filter_type.lower() == 'box':
                    scalar_filt = filter_box(scalar_filt, delta=filter_size)
                else:
                    raise ValueError("Check the filter_type input value")
                
                save_file(scalar_filt, filt_path)
                
        print (f"Done Filtering Field '{self.folder_path}'.")
        print (f"Filtered Field path: '{filt_folder_path}'.")
        
        return filt_folder_path
    
    def find_path(self, attr):
        """
        Finds a specified attribute in the attributes list and returns the corresponding element 
        in the paths list.
        
        Parameters:
        -----------
        attr : str
            The specific element to find in the first list.
            
        Returns:
        --------
        str
            The corresponding element in the second list if the specific element is found in the first list.
            
        Raises:
        -------
        TypeError
            If 'attr' is not a string.
        ValueError
            If the specific element is not found in the attributes list.
        
        """
        if not isinstance(attr, str):
            raise TypeError("'attr' must be a string")
            
        # Find the specific element in list1 and access the corresponding element in list2
        try:
            index = self.attr_list.index(attr)
            corresponding_element = self.paths_list[index]
            return corresponding_element
        except ValueError:
            raise ValueError("The element is not in attr_list.")
        return
        
    def plot_x_midplane(self, attribute, vmin=None, vmax=None):
        """
        Plots the x midplane of a specified attribute in the Field3D class.
    
        Description:
        ------------
        This method plots the x midplane of a specified attribute in the Field3D class. It verifies 
        if the attribute is valid, and then uses the attribute's `plot_x_midplane` method to generate 
        the plot.
    
        Parameters:
        -----------
        attribute : str
            The name of the attribute to plot.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        self.check_valid_attribute(attribute)
        getattr(self, attribute).plot_x_midplane(self.mesh, title=attribute, vmin=vmin, vmax=vmax)
        
    def plot_y_midplane(self, attribute, vmin=None, vmax=None):
        """
        Plots the y midplane of a specified attribute in the Field3D class.
    
        Description:
        ------------
        This method plots the y midplane of a specified attribute in the Field3D class. It verifies 
        if the attribute is valid, and then uses the attribute's `plot_y_midplane` method to generate 
        the plot.
    
        Parameters:
        -----------
        attribute : str
            The name of the attribute to plot.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        self.check_valid_attribute(attribute)
        getattr(self, attribute).plot_y_midplane(self.mesh, title=attribute, vmin=vmin, vmax=vmax)
        
    def plot_z_midplane(self, attribute, vmin=None, vmax=None):
        """
        Plots the z midplane of a specified attribute in the Field3D class.
    
        Description:
        ------------
        This method plots the z midplane of a specified attribute in the Field3D class. It verifies 
        if the attribute is valid, and then uses the attribute's `plot_z_midplane` method to generate 
        the plot.
    
        Parameters:
        -----------
        attribute : str
            The name of the attribute to plot.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        self.check_valid_attribute(attribute)
        getattr(self, attribute).plot_z_midplane(self.mesh, title=attribute, vmin=vmin, vmax=vmax)
    
    def print_attributes(self):
        """
        Prints the valid attributes of the class and their corresponding file paths.
    
        Description
        -----------
        
        This method calls the `update` method with `print_valid_attributes` set to `True`. 
        As a result, it prints out the valid attributes (those that have corresponding files 
        in the data path) of the class and their corresponding file paths. This is useful 
        when you want to see which attributes are currently valid in the class instance.
        """
        self.update(print_valid_attributes=True)
    
    def update(self, verbose=False, print_valid_attributes=False):
        """
        Update the attributes of the class based on the existence of files in the specified data path.
        
        This method checks the existence of files corresponding to the attribute paths in the data path.
        If a file exists for an attribute and it was not present before, it initializes a new attribute
        in the class using Scalar3D with the file path. If verbose is True, it prints the new attributes
        initialized and the existing attributes with their paths.
    
        Parameters:
        -----------
        
        - verbose (bool, optional): If True, prints information about the initialization of new attributes.
                                   Default is False.
        """
        files_in_folder = os.listdir(self.data_path)
        bool_list = []
        for attribute_name, path in zip(self.attr_list, self.paths_list):
            file_name = os.path.basename(path)
            if file_name in files_in_folder:
                bool_list.append(True)
            else:
                bool_list.append(False)
        if not hasattr(self, 'bool_list'): # means that the field is being initialized
            new_list = bool_list
        else:
            new_list = [a and (not b) for a, b in zip(bool_list, self.bool_list)]
        self.bool_list = bool_list
        
        remove_list = []
        
        # Assign the new attributes to the class
        files_in_folder = os.listdir(self.data_path)
        for attribute_name, path, is_new in zip(self.attr_list, self.paths_list, new_list):
            file_name = os.path.basename(path)
            if (file_name in files_in_folder) and is_new:
                x = Scalar3D(self.shape, path=path)
                setattr(self, attribute_name, x)
                del x
            if (file_name not in files_in_folder) and hasattr(self, attribute_name):
                delattr(self, attribute_name)
                remove_list.append(True)
            else:
                remove_list.append(False)
        
        if verbose:
            if bool_list != new_list:
                new_attr = [attr for attr, is_new in zip(self.attr_list, new_list) if is_new]
                new_path = [path for path, is_new in zip(self.paths_list, new_list) if is_new]
                rem_attr = [attr for attr, is_removed in zip(self.attr_list, remove_list) if is_removed]
                rem_path = [path for path, is_removed in zip(self.paths_list, remove_list) if is_removed]
                data = zip(new_attr, new_path)
                print("New field attributes initialized:")
                print(tabulate(data, headers=['Attribute', 'Path'], tablefmt='pretty'))
                print("\n")
                data = zip(rem_attr, rem_path)
                print("Field attributes deleted:")
                print(tabulate(data, headers=['Attribute', 'Path'], tablefmt='pretty'))
                print("\n")
                
            
            got_attr = [attr for attr, is_present in zip(self.attr_list, self.bool_list) if is_present]
            got_path = [path for path, is_present in zip(self.paths_list, self.bool_list) if is_present]
            data = zip(got_attr, got_path)
            print("Field attributes:")
            print(tabulate(data, headers=['Attribute', 'Path'], tablefmt='pretty'))
            
        if print_valid_attributes:
            got_attr = [attr for attr, is_present in zip(self.attr_list, self.bool_list) if is_present]
            got_path = [path for path, is_present in zip(self.paths_list, self.bool_list) if is_present]
            data = zip(got_attr, got_path)
            print("Field attributes you can call and relative file paths:")
            print(tabulate(data, headers=['Attribute', 'Path'], tablefmt='pretty'))
            
        return
    
###############################################################################
#                                Scalar3D
###############################################################################
class Scalar3D:
    """
    A class representing a 3D scalar field.

    Attributes:
    -----------
    __scalar_dimension : int
        The dimension of the scalar field.
    test_phase : bool
        A flag indicating whether the test phase is active.
    VALID_DIMENSIONS : list
        A list of valid dimensions for the scalar field.

    Methods:
    --------
    __init__(shape, value=None, path=''):
        Initializes a Scalar3D object.
    value
        Getter and setter for the value attribute.
    shape
        Getter and setter for the shape attribute.
    path
        Getter and setter for the path attribute.
    Nx
        Getter and setter for the Nx attribute.
    Ny
        Getter and setter for the Ny attribute.
    Nz
        Getter and setter for the Nz attribute.
    file_name
        Getter and setter for the file_name attribute.
    file_id
        Getter and setter for the file_id attribute.
    filter_size
        Getter and setter for the filter_size attribute.
    _3d
        Getter for the 3D reshaped scalar field.
    is_light_mode():
        Checks if the scalar field is in light mode.
        
    reshape_3d():
        Reshapes the scalar field to 3D.
        
    reshape_column():
        Reshapes the scalar field to a column vector.
        
    reshape_line():
        Reshapes the scalar field to a row vector.
        
    cut(n_cut=1, mode='equal'):
        Cuts the scalar field.
        
    filter_gauss(delta, n_cut=0, mute=False):
        Filters the scalar field with a Gaussian function.
        
    plot_x_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
        Plots the x midplane of a 3D field.
        
    plot_y_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
        Plots the y midplane of a 3D field.
        
    plot_z_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
        Plots the z midplane of a 3D field.
    """
    
    __scalar_dimension = 3
    test_phase = True  # I only need it to be True when I debug the code
    
    VALID_DIMENSIONS = [3, 1] # I am still not using it but I want to insert a check that allows to input the scalar field also as a 1D vector
    
    def __init__(self, shape, value=None, path=''):
        """
        Initializes a Scalar3D object.

        Parameters:
        -----------
        shape : list
            The shape of the scalar field as a list of three integers.
        value : array-like, optional
            The value of the scalar field. Default is None.
        path : str, optional
            The file path of the scalar field. Default is an empty string.

        """
        # check that the shape of the field is a list of 3 integers
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Scalar3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        # setting the shape, Nx, Ny, Nz
        self.shape = shape
        
        # assign the value to the field and reshape it if it was initialized
        self._value = value
        if value is not None and np.ndim(value)!=3:
            self.reshape_3d()
        
        #assign the path
        if path != '':
            self.path = path
    
    # The value attribute contains the array with the values of the field.
    # By default it is reshaped in a 3d array
    @property
    def value(self):
        # if Scalar3D.test_phase is True:
        #     print("Getting scalar field...")
        if self._value is not None:
            # print("value assigned to the variable. returning the field in memory")
            return self._value
        else:
            if self.path != '':
                # print("Value not assigned, reading the file")
                return process_file(self.path)
            else:
                raise ValueError("To call the value of a Scalar3D object, you must specify either the value or the file path")    
    @value.setter
    def value(self, value):
        # assign value
        self._value = value
        # reshape to 3d field by default
        if np.ndim(value)!=3:
            self.reshape_3d()
    
    # shape getter and setter. The setter also set Nx, Ny, Nz. These variables
    # can be considered redundant but help coding. only set them using the shape setter.
    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, shape):
        # Check that the shape has the correct format
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Scalar3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        # assign the values
        self._Nx = shape[0]
        self._Ny = shape[1]
        self._Nz = shape[2]
        self._shape = (self.Nx, self.Ny, self.Nz)
        
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, path):
        import re
        # check the input is a string and that the path exist
        if not isinstance(path, str):
            TypeError("The file path must be a string")
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")
        
        self._path = path
        self._file_name = path.split('/')[-1]
        pattern = r'id\d+'
        match = re.search(pattern, self._file_name, re.IGNORECASE)
        if match:
            self._file_id = match.group()
        
        folder_name = path.split('/')[0]
        pattern = r'filter(\d+)'
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            self._filter_size = int(match.group(1))
        else:
            self._filter_size = None
            
        
    @property
    def Nx(self):
        return self._Nx
    @Nx.setter
    def Nx(self, Nx):
        self._Nx = Nx
        
    @property
    def Ny(self):
        return self._Ny
    @Ny.setter
    def Ny(self, Ny):
        self._Ny = Ny
    
    @property
    def Nz(self):
        return self._Nz
    @Nz.setter
    def Nz(self, Nz):
        self._Nz = Nz
    
    @property
    def file_name(self):
        return self._file_name
    @file_name.setter
    def file_name(self, file_name):
        if isinstance(file_name, str):
            self._file_name = file_name
        else:
            TypeError("The file name must be a string")
            
    @property
    def file_id(self):
        return self._file_id
    @file_id.setter
    def file_id(self, file_id):
        if isinstance(file_id, str):
            self._file_id = file_id
        else:
            TypeError("The file name must be a string")
            
    @property
    def filter_size(self):
        return self._filter_size
    @filter_size.setter
    def filter_size(self, filter_size):
        if isinstance(filter_size, int):
            self._filter_size = filter_size
        else:
            TypeError("The filter size must be an integer")
    
    @property
    def _3d(self):
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(self.Nx, self.Ny, self.Nz)        

    @property
    def _3D(self):
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(self.Nx, self.Ny, self.Nz)        

    
    # Class Methods
    def is_light_mode(self):
        """
        Checks if the scalar field is in light mode, that is when only the
        path is specified without specifying the value as an array.

        Returns:
        --------
        bool
            True if the scalar field is in light mode, False otherwise.

        Raises:
        -------
        ValueError
            If neither the value nor the path is specified.

        """
        if self._value is not None:
            return False
        else:
            if self.path != '':
                return True
            else:
                raise ValueError("To call the value of a Scalar3D object, you must specify either the value or the file path")    
    
    def reshape_3d(self):
        """
        Reshapes the scalar field to 3D.

        Returns:
        --------
        array-like
            The scalar field reshaped to 3D.

        """
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(self.Nx, self.Ny, self.Nz)
        
    def reshape_column(self):
        """
        Reshapes the scalar field to a column vector.

        Returns:
        --------
        array-like
            The scalar field reshaped to a column vector.

        """
        new_shape = [self.Nx*self.Ny*self.Nz, 1]
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(new_shape)
        
    def reshape_line(self):
        """
        Reshapes the scalar field to a row vector.

        Returns:
        --------
        array-like
            The scalar field reshaped to a row vector.
   
        """
        new_shape = [1, self.Nx*self.Ny*self.Nz]
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(new_shape)
        
    def cut(self, n_cut=1, mode='equal'):
        """
        Cuts the scalar field.

        Parameters:
        -----------
        n_cut : int or tuple, optional
            The number of samples to cut at the extrema. If mode is 'equal', n_cut is an integer 
            specifying the number of samples to cut from each side. If mode is 'xyz', n_cut is a 
            tuple specifying the number of samples to cut for each dimension. Default is 1.
        mode : {'equal', 'xyz'}, optional
            The mode of cutting. 'equal' cuts the same number of samples from each side, 
            'xyz' allows specifying the number of samples to cut for each dimension. Default is 'equal'.

        Returns:
        --------
        array-like
            The cut scalar field.

        Example:
        --------
        >>> # Example of cutting a scalar field
        >>> field = Scalar3D(shape=[10, 10, 10], value=np.random.rand(10, 10, 10))
        >>> # Cut the field with equal number of samples removed from each side
        >>> cut_field_equal = field.cut(n_cut=2, mode='equal')
        >>> print("Cut field with equal mode:")
        >>> print(cut_field_equal)
        >>> # Cut the field with specified number of samples removed for each dimension
        >>> cut_field_xyz = field.cut(n_cut=(1, 2, 3), mode='xyz')
        >>> print("\nCut field with xyz mode:")
        >>> print(cut_field_xyz)
        """
        if not self.is_light_mode():
            # TODO: update this function to handle the xyz mode also in this branch
            self.reshape_3d()
            if mode=='equal':
                self.value = self.value[n_cut:self.Nx-n_cut,n_cut:self.Ny-n_cut,n_cut:self.Nz-n_cut]
            self.shape = self.value.shape #update the shape of the field
        else:
            field_cut = self.reshape_3d()
            if mode=='equal':
                field_cut = field_cut[n_cut:self.Nx-n_cut,n_cut:self.Ny-n_cut,n_cut:self.Nz-n_cut]
            elif mode=='xyz':
                n_cut_x = n_cut[0]
                n_cut_y = n_cut[1]
                n_cut_z = n_cut[2]
                field_cut = field_cut[n_cut_x:self.Nx-n_cut_x,n_cut_y:self.Ny-n_cut_y,n_cut_z:self.Nz-n_cut_z]
                return field_cut
    
    def filter_gauss(self, delta,n_cut=0,mute=False):
        """
        Filters the scalar field with a Gaussian function.
        The variance sigma is considered equal to:
            sigma = sqrt(1/12*delta**2) 
        where delta is the filter size (in this case specified as
        the number of cells and not in meters)

        Parameters:
        -----------
        delta : float
            The amplitude of the Gaussian filter.
        n_cut : int, optional
            The number of samples to cut at the extrema. Default is 0.
        mute : bool, optional
            A flag indicating whether to mute the output. Default is False.

        Returns:
        --------
        array-like
            The filtered scalar field.

        """
        # delta is the amplitude of the gaussian filter
        # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
        
        # filter the field with a constant amplitude gaussian function. notice that in this case we consider that the points are equispaced. It should not be a problem also for non equispaced points.    
        field_filt = gaussian_filter(self.value, sigma=np.sqrt(1/12*delta**2), mode='constant')  

        return field_filt
    

    def plot_x_midplane(self, mesh, title='', colormap='viridis', vmin=None, vmax=None):
        """
        Plots the x midplane of a 3D field.
    
        Description:
        ------------
        This method plots the x midplane of a 3D field using the provided mesh. It uses the midpoint 
        of the x-axis to generate a 2D plot of the field values at that plane.
    
        Parameters:
        -----------
        mesh : object
            The mesh object containing the coordinates.
        title : str, optional
            The title of the plot. Default is an empty string.
        colormap : str, optional
            The colormap to use for the plot. Default is 'viridis'.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        Y,Z = 1e3*mesh.Y3D, 1e3*mesh.Z3D
        f = self.reshape_3d()
        
        # Calculate the midplane index
        x_mid = Y.shape[0] // 2

        # Plot the x midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(Y[x_mid, :, :], Z[x_mid, :, :], f[x_mid, :, :], shading='auto', cmap = colormap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('y (mm)', fontsize=18)
        ax.set_ylabel('z (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()
        
    def plot_y_midplane(self, mesh, title='', colormap='viridis', vmin=None, vmax=None):
        """
        Plots the y midplane of a 3D field.
    
        Description:
        ------------
        This method plots the y midplane of a 3D field using the provided mesh. It uses the midpoint 
        of the x-axis to generate a 2D plot of the field values at that plane.
    
        Parameters:
        -----------
        mesh : object
            The mesh object containing the coordinates.
        title : str, optional
            The title of the plot. Default is an empty string.
        colormap : str, optional
            The colormap to use for the plot. Default is 'viridis'.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        if not isinstance(mesh, Mesh3D):
            raise ValueError("mesh must be an object of the class Mesh3D")
        
        y_mid = mesh.shape[2]//2

        # Plot the y midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(1e3*mesh.X_midY, 1e3*mesh.Z_midY, self._3d[:, y_mid, :], shading='auto', cmap = colormap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x (mm)', fontsize=18)
        ax.set_ylabel('z (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()        
    
    def plot_z_midplane(self, mesh, title='', colormap='viridis', vmin=None, vmax=None):
        """
        Plots the x midplane of a 3D field.
    
        Description:
        ------------
        This method plots the x midplane of a 3D field using the provided mesh. It uses the midpoint 
        of the x-axis to generate a 2D plot of the field values at that plane.
    
        Parameters:
        -----------
        mesh : object
            The mesh object containing the coordinates.
        title : str, optional
            The title of the plot. Default is an empty string.
        colormap : str, optional
            The colormap to use for the plot. Default is 'viridis'.
        vmin : float, optional
            The minimum value for the color scale. Default is None.
        vmax : float, optional
            The maximum value for the color scale. Default is None.
    
        Returns:
        --------
        None
    
        """
        if not isinstance(mesh, Mesh3D):
            raise ValueError("mesh must be an object of the class Mesh3D")
        
        z_mid = mesh.shape[2]//2
        
        # Plot the z midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(1e3*mesh.X_midZ, 1e3*mesh.Y_midZ, self._3d[:, :, z_mid], shading='auto', cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x (mm)', fontsize=18)
        ax.set_ylabel('y (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()
  
class Mesh3D:
    """
    A class used to represent a 3D mesh.

    This class takes three Scalar3D objects representing the X, Y, and Z coordinates of a 3D mesh.
    It checks that the input objects are instances of the Scalar3D class and have the same shape.
    The shape of the mesh is expected to be a list of three integers.
    The class also provides properties to access the unique values of the X, Y, and Z coordinates and their 3D representations.
    It also provides properties to access the X, Y, and Z coordinates at the midpoints along each axis.

    Attributes:
    -----------
    shape : list
        The shape of the 3D mesh.
        
    Nx : int
        The size of the mesh along the X axis.
        
    Ny : int
        The size of the mesh along the Y axis.
        
    Nz : int
        The size of the mesh along the Z axis.
        
    X : Scalar3D
        The X coordinates of the mesh.
        
    Y : Scalar3D 
        The Y coordinates of the mesh.
        
    Z : Scalar3D 
        The Z coordinates of the mesh.

    Methods:
    --------
    X1D: 
        Returns the unique values of the X coordinates.
        
    Y1D: 
        Returns the unique values of the Y coordinates.
        
    Z1D: 
        Returns the unique values of the Z coordinates.
        
    X3D: 
        Returns the 3D representation of the X coordinates.
        
    Y3D: 
        Returns the 3D representation of the Y coordinates.
        
    Z3D: 
        Returns the 3D representation of the Z coordinates.
        
    X_midY: 
        Returns the X coordinates at the midpoint along the Y axis.
    
    X_midZ: 
        Returns the X coordinates at the midpoint along the Z axis.
    
    Y_midX: 
        Returns the Y coordinates at the midpoint along the X axis.
    
    Y_midZ: 
        Returns the Y coordinates at the midpoint along the Z axis.
    
    Z_midX: 
        Returns the Z coordinates at the midpoint along the X axis.
    
    Z_midY: 
        Returns the Z coordinates at the midpoint along the Y axis.
    """

    __scalar_dimension = 3

    VALID_DIMENSIONS = [3, 1] # I am still not using it but I want to insert a check that allows to input the scalar field also as a 1D vector
    
    def __init__(self, X, Y, Z):
        """
        Initializes a Mesh3D object.
    
        Parameters:
        -----------
        X : Scalar3D
            The X coordinates of the mesh.
            
        Y : Scalar3D 
            The Y coordinates of the mesh.
            
        Z : Scalar3D 
            The Z coordinates of the mesh.
        
        Raises:
        -------
        TypeError:
            If X, Y, or Z are not instances of the Scalar3D class.
        ValueError:
            If X, Y, and Z do not have the same dimensions.
        """
        # check that X, Y and Z are Scalar3D objects
        if not isinstance(X, Scalar3D):
            raise TypeError("X must be an object of the class Scalar3D")
        if not isinstance(Y, Scalar3D):
            raise TypeError("X must be an object of the class Scalar3D")
        if not isinstance(Z, Scalar3D):
            raise TypeError("Z must be an object of the class Scalar3D")
            
        # check that X, Y and Z have the same dimensions
        if not check_same_shape(X, Y, Z):
            raise ValueError("Z must be an object of the class Scalar3D")
        
        shape = X.shape
        
        # check that the shape of the field is a list of 3 integers
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Mesh3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        
        # setting the shape, Nx, Ny, Nz
        self.shape = shape
        self.Nx = shape[0]
        self.Ny = shape[1]
        self.Nz = shape[2]
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        x_mid = self.shape[0]//2
        y_mid = self.shape[1]//2
        z_mid = self.shape[2]//2
        
        self._X_midZ = X._3d[:, :, z_mid]
        self._Y_midZ = Y._3d[:, :, z_mid]
        
        self._X_midY = X._3d[:, y_mid, :]
        self._Z_midY = Z._3d[:, y_mid, :]
        
        self._Y_midX = Y._3d[x_mid, :, :]
        self._Z_midX = Y._3d[x_mid, :, :]
        
        # Characteristic mesh dimension
        self.l = (np.average(np.diff(self.X1D))*np.average(np.diff(self.Y1D))*np.average(np.diff(self.Z1D)))**(1/3)
        

    # The value attribute contains the array with the values of the field.
    # By default it is reshaped in a 3d array
    
    @property
    def X1D(self):
        self._X1D = np.unique(self.X.value)
        return self._X1D

    @property
    def Y1D(self):
        self._Y1D = np.unique(self.Y.value)
        return self._Y1D    

    @property
    def Z1D(self):
        self._Z1D = np.unique(self.Z.value)
        return self._Z1D    
    
    @property
    def X3D(self):
        return self.X._3d

    @property
    def Y3D(self):
        return self.Y._3d    

    @property
    def Z3D(self):
        return self.Z._3d
    
    @property
    def X_midY(self):
        return self._X_midY
    
    @property
    def X_midZ(self):
        return self._X_midZ
    
    @property
    def Y_midX(self):
        return self._Y_midX
    
    @property
    def Y_midZ(self):
        return self._Y_midZ
    
    @property
    def Z_midX(self):
        return self._Z_midX
    
    @property
    def Z_midY(self):
        return self._Z_midY
    
    
###############################################################################
#                               Functions
###############################################################################
def compute_cell_volumes(x, y, z):
    """
    Compute the volumes of the cells in a 3D mesh grid.

    This function calculates the cell volumes for a given set of x, y, and z coordinates.
    The coordinates are provided as 1D vectors. The function computes the distances between 
    consecutive points in each direction, constructs a 3D meshgrid of these distances, and
    then calculates the volume of each cell.

    Parameters:
    -----------
    x : array-like
        A 1D array of x coordinates.
    y : array-like
        A 1D array of y coordinates.
    z : array-like
        A 1D array of z coordinates.

    Returns:
    --------
    cell_volumes : ndarray
        A 3D array where each element represents the volume of a cell in the mesh grid.
        
    Example:
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 2])
    >>> z = np.array([0, 1, 2, 3, 4])
    >>> volumes = compute_cell_volumes(x, y, z)
    >>> print(volumes.shape)
    (4, 3, 5)
    >>> print(volumes)
    array([[[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]],
           [[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]],
           [[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]],
           [[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]])
    """
    # x y and z are 1d vectors
    # Calculate the distances between the points in each direction
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    
    # Add an extra element to each distance array to make them the same size as the original arrays
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])
    dz = np.concatenate([dz, [dz[-1]]])
    
    # Create 3D meshgrid of distances
    dx_mesh, dy_mesh, dz_mesh = np.meshgrid(dx, dy, dz, indexing='ij')
    
    # Calculate the cell volumes
    cell_volumes = dx_mesh * dy_mesh * dz_mesh
    
    return cell_volumes

def delete_file(file_path):
    """
    Deletes the specified file from the file system.
    
    This function checks if the file at the given path exists and deletes it if it does.
    If the file does not exist, it prints a message indicating that the file was not found.
    
    Parameters:
    -----------
    file_path : str
        The path to the file that needs to be deleted.
    
    Returns:
    --------
    None
    
    Example:
    --------
    >>> delete_file("example.txt")
    No such file: 'example.txt'
    >>> with open("example.txt", "w") as f:
    ...     f.write("This is a test file.")
    >>> delete_file("example.txt")
    >>> os.path.exists("example.txt")
    False
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        print(f"No such file: '{file_path}'")

def process_file(file_path):
    """
    Read a binary file and convert its contents into a numpy array.

    The function uses numpy's fromfile function to read a binary file and 
    convert its contents into a numpy array.
    The data type of the elements in the output array is set to '<f4', which 
    represents a little-endian single-precision float.

    Parameters:
    -----------
    file_path : str
        The path to the file to be processed.

    Returns:
    --------
        
    numpy.ndarray: 
        The numpy array obtained from the file contents.
    """
    # Placeholder for file processing logic
    return np.fromfile(file_path,dtype='<f4')


def filter_gauss(field,delta, mode='mirror'):
    """
    Apply a Gaussian filter to a 3D numpy array.

    The function checks the input types and dimensions, then applies a 
    Gaussian filter to the input array using scipy's gaussian_filter function.
    The standard deviation of the Gaussian filter is calculated as 
    sqrt(1/12*delta^2), which corresponds to a Gaussian distribution with a 
    variance equal to the square of the filter size divided by 12.

    Parameters:
    -----------
    
        field : numpy.ndarray
            The input 3D array.
            
        delta : int
            The size of the Gaussian filter.
            
        mode : str, optional
            Determines how the input array is extended when the filter overlaps a border. Default is 'mirror'.
            Possible values are 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.

    Raises:
    -------
        
        TypeError: 
            If delta is not an integer or field is not a numpy array.
            
        ValueError: 
            If field is not a 3-dimensional array.

    Returns:
    --------
    
        field_filt : numpy.ndarray 
            The filtered array.
    """
    
    if not isinstance(delta, int):
        raise TypeError("filter_size must be an integer")
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a numpy array")
    if not len(field.shape)==3:
        raise ValueError("field must be a 3 dimensional array")
        
    fieldFilt = gaussian_filter(field, sigma=np.sqrt(1/12*delta**2), mode=mode)

    return fieldFilt

def filter_box(field, delta, mode='mirror'):
    """
    Apply a box filter to a 3D numpy array using scipy's convolve function.

    The function creates a box kernel with the given size, normalizes it so that the sum of its elements is 1,
    and applies it to the input array using scipy's convolve function.

    Note: 
    -----
    When the kernel size is even, the center of the kernel is not a single element but lies between elements.
    In such cases, scipy's convolve function does not shift the kernel to align its center with an element of the input array.
    Instead, it uses the original alignment where the center of the kernel is between elements.
    This means that the output array will be shifted compared to what you might expect if the kernel was centered on an element of the input array.
    If you want to ensure that the kernel is always centered on an element of the input array, you should use an odd-sized kernel.
    If you need to use an even-sized kernel and want to center it on an element, you would need to manually shift the output array to align it as desired.

    Parameters:
    -----------
    field : numpy.ndarray
        The input 3D array.
        
    delta : int
        The size of the box filter.

    mode : str, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border.
        Default is 'mirror'.

    Returns:
    --------
    field_filt : numpy.ndarray
        The filtered array.

    Example:
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import convolve

    >>> # Create a sample 3D array
    >>> field = np.random.rand(5, 5, 5)

    >>> # Apply a box filter with size 3
    >>> delta = 3
    >>> filtered_field = filter_box(field, delta)

    >>> print("Original field:\n", field)
    >>> print("Filtered field:\n", filtered_field)
    """
    if not isinstance(delta, int):
        raise TypeError("filter_size must be an integer")
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a numpy array")
    if not len(field.shape) == 3:
        raise ValueError("field must be a 3 dimensional array")
    
    box = np.ones((delta, delta, delta))
    box /= delta**3
    return convolve(field, box, mode=mode)

def filter_3D(field, filter_size, RHO=None, favre=False, filter_type='Gauss'):
    """
    Apply a 3D filter (Gaussian or box) to a numpy array, with optional Favre filtering.
    
    This function filters a 3D field using either a Gaussian or box filter. When Favre filtering is enabled, the field
    is first multiplied by the density field (RHO) before filtering, and the result is normalized by the filtered density field.
    
    Parameters:
    -----------
    field : numpy.ndarray
        The input 3D array to be filtered.
    
    filter_size : float
        The size of the filter.
    
    RHO : numpy.ndarray, optional
        The density field used for Favre filtering. Required if favre is True.
    
    favre : bool, optional
        If True, apply Favre filtering using the density field (RHO). Default is False.
    
    filter_type : str, optional
        The type of filter to apply. Valid options are 'Gauss' and 'Box'. Default is 'Gauss'.
    
    Returns:
    --------
    field_filt : numpy.ndarray
        The filtered 3D array.
    
    Raises:
    -------
    ValueError
        - If favre is True and RHO is not provided.
        - If field or RHO are not 3-dimensional arrays.
        - If field and RHO do not have the same shape.
        - If an invalid filter_type is provided.
    
    TypeError
        If RHO is not a numpy array.
    
    Example:
    --------
    >>> import numpy as np
    
    >>> # Create a sample 3D array
    >>> field = np.random.rand(5, 5, 5)
    >>> RHO = np.random.rand(5, 5, 5)
    >>> filter_size = 2.0
    
    >>> # Apply Gaussian filter
    >>> filtered_field = filter_3D(field, filter_size, filter_type='Gauss')
    >>> print("Filtered field (Gaussian):\n", filtered_field)
    
    >>> # Apply box filter
    >>> filtered_field = filter_3D(field, filter_size, filter_type='Box')
    >>> print("Filtered field (Box):\n", filtered_field)
    
    >>> # Apply Favre filtering with Gaussian filter
    >>> filtered_field_favre = filter_3D(field, filter_size, RHO=RHO, favre=True, filter_type='Gauss')
    >>> print("Favre filtered field (Gaussian):\n", filtered_field_favre)
    
    >>> # Apply Favre filtering with box filter
    >>> filtered_field_favre = filter_3D(field, filter_size, RHO=RHO, favre=True, filter_type='Box')
    >>> print("Favre filtered field (Box):\n", filtered_field_favre)
    """
    if favre:
        if not isinstance(RHO, np.ndarray):
            raise ValueError("If Favre==True the function needs the density field as an input."
                             "\nRHO must be a numpy array")
        if not len(field.shape) == 3:
            raise ValueError("RHO must be a 3 dimensional array")
        if not RHO.shape==field.shape:
            raise ValueError("field and RHO must have the same shape")
    
    valid_filter_types = ['gauss', 'box']
    check_input_string(filter_type, valid_filter_types, 'filter_type')
    
    if favre:
        field = field*RHO
    
    if filter_type.lower() == 'gauss':
        field_filt = filter_gauss(field, delta=filter_size)
        if favre:
            RHO = filter_gauss(RHO, filter_size)
            field_filt = field_filt/RHO
            
    elif filter_type.lower() == 'box':
        field_filt = filter_box(field, delta=filter_size)
        if favre:
            RHO = filter_gauss(RHO, filter_size)
            field_filt = field_filt/RHO
    else: # handle invalid filter-types
        raise ValueError("Check the filter_type input value.\n"
                        f"Valid entries are: {valid_filter_types}")
        
    return field_filt


def save_file (X, file_name):
    """
    Saves the given array to a file in binary format.

    This function converts the input array to a 32-bit float representation and saves it
    to a file using the specified file name. The file is saved in a binary format.

    Parameters:
    -----------
    X : np.ndarray
        The array to be saved. It will be converted to a 32-bit float array before saving.
    file_name : str
        The name of the file where the array will be saved.

    Returns:
    --------
    None

    Example:
    --------
    >>> import numpy as np
    >>> X = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    >>> save_file(X, "test.bin")
    >>> loaded_X = np.fromfile("test.bin", dtype=np.float32)
    >>> print(loaded_X)
    [1.5 2.5 3.5]
    """
    import numpy as np
    X = X.astype(np.float32)
    X.tofile(file_name)
    
def gradient_x(F, mesh, filter_size=1):
    '''
        Description
        -----------
        
        Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
        used to computed the gradients on all the possible downsampled grids.
        
        Specifically, the parameter filter_size is used to temporarily downsample
        the grid in the x direction. The function considers one point each 
        filter_size points and computes the derivatives on this downsampled grid.
        Does this for every possible downsampled grid, so in the end the output
        field has the same shape as the input field.

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        verbose : bool
            If True, it will output relevant information.

        Returns
        -------
        grad_x : numpy array
            The x component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_x = np.zeros(F._3d.shape)
    X1D = mesh.X1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[start::filter_size, :, :]
        
        grad_x[start::filter_size, :, :] = np.gradient(field, X1D[start::filter_size], axis=0)
        
    return grad_x

def gradient_y(F, mesh, filter_size=1):
    '''
        Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
        used to computed the gradients on all the possible downsampled grids

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        verbose : bool
            If True, it will output relevant information.

        Returns
        -------
        grad_y : numpy array
            The y component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_y = np.zeros(F._3d.shape)
    Y1D = mesh.Y1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[:, start::filter_size, :]
        
        grad_temp = np.gradient(field, Y1D[start::filter_size], axis=1)
        
        grad_y[:, start::filter_size, :] = grad_temp
        
    return grad_y

def gradient_z(F, mesh, filter_size=1):
    '''
        Computes the z component of the gradient of a 3D, non downsampled, filtered field. 
        Numpy is used to computed the gradients on all the possible downsampled grids

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        Returns
        -------
        grad_z : numpy array
            The z component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_z = np.zeros(F._3d.shape)
    Z1D = mesh.Z1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[:, :, start::filter_size]
        
        grad_temp = np.gradient(field, Z1D[start::filter_size], axis=2)
        
        grad_z[:, :, start::filter_size] = grad_temp
        
    return grad_z

def generate_mask(start, shape, delta):
    '''
        Computes the downsampled mask of a 3D field.

        Parameters
        ----------
        
        start : list of int
            Is the a list with the indexes where to start doing the mask.
            
        shape : list of int
            Is the shape of the input field
        
        delta : int
            Is the filter size

        Returns
        -------
        mask : numpy array of bool
            A 3D vector of boolean values.
            
        '''
    import numpy as np
    
    idx_x = np.arange(start[0], shape[0], delta)
    idx_y = np.arange(start[1], shape[1], delta)
    idx_z = np.arange(start[2], shape[2], delta)
    
    # Create mask
    mask = np.zeros((shape[0], shape[1], shape[2]), dtype=bool)
    mask[idx_x[:, None, None], idx_y[None, :, None], idx_z[None, None, :]] = True

    return mask

def check_same_shape(*args):
    '''
        Checks if the shape of the input arguments *args is the same

        Returns
        -------
        bool : bool
            Assumes the value True only if all the inputs have the same shape.
            
        '''
    # Check if there are at least two arguments
    if len(args) < 2:
        raise ValueError("At least two arguments are required")
    
    # Get the shape of the first argument
    reference_shape = args[0].shape
    
    # Check the shape of each argument against the reference shape
    for arg in args[1:]:
        if arg.shape != reference_shape:
            return False
    
    return True

def check_input_string(input_string, valid_strings, input_name):
    '''
        Checks if the value of input_string is contained in the list valid_strings.
        If the result is positive, returns None, if the result is negative
        raises an error
        
        Parameters
        ----------
        
        input_string : string
            Is the string that must be checked
            
        valid_strings : list of strings
            Is the list of valid strings
        
        input_name : string
            Is the name of the parameter that we are checking

        Returns
        -------
        None 
        
        NOTES:
        -------
        Example of output if the function finds an error:
        
        ValueError: Invalid parameter mode 'mode1'. Valid options are: 
         - mode_1
         - mode_2
         - mode_3
        
        '''
    input_lower = input_string.lower()
    valid_strings_lower = [valid_string.lower() for valid_string in valid_strings]

    if input_lower not in valid_strings_lower:
        valid_strings_ = "\n - ".join(valid_strings)
        raise ValueError("Invalid parameter {} '{}'. Valid options are: \n - {}".format(input_name, input_string, valid_strings_))

def plot_power_spectrum(field, C=5):
    """
    Plots the power spectrum of a 3D field.

    This function performs a 3D Fourier Transform on the input field, computes the power spectrum,
    and plots both the power spectrum and its averaged version. It also includes a reference line
    proportional to k^(-5/3).

    Parameters:
    -----------
    field : np.ndarray
        The 3D field for which the power spectrum is to be plotted.
    C : float, optional
        The proportionality constant for the reference line. Default is 5.

    Returns:
    --------
    None

    Example:
    --------
    >>> field = np.random.random((64, 64, 64))
    >>> plot_power_spectrum(field)
    """
    # Perform 3D Fourier Transform
    power_spectrum = np.fft.fftn(field) / np.prod(field.shape)
    # Calculate the power spectrum
    power_spectrum = (np.abs(power_spectrum)) # **(2) should I square this value?
    # Shift the power spectrum
    power_spectrum = np.fft.fftshift(power_spectrum)

    # Calculate the frequencies
    freqs_x = np.fft.fftfreq(field.shape[0])
    freqs_y = np.fft.fftfreq(field.shape[1])
    freqs_z = np.fft.fftfreq(field.shape[2])
    
    # Create a 3D meshgrid of frequencies
    freqs_x_mesh, freqs_y_mesh, freqs_z_mesh = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y), np.fft.fftshift(freqs_z), indexing='ij')
    # Create a mesh with the absolute frequency (omega = sqrt(omega_x^2+omega_y^2+omega_z^2))
    freq_mesh = np.sqrt(freqs_x_mesh**2 + freqs_y_mesh**2 + freqs_z_mesh**2)
    del freqs_x_mesh, freqs_y_mesh, freqs_z_mesh
    
    freq_mesh = freq_mesh.flatten()
    power_spectrum = power_spectrum.flatten()
    
    sort_indices = np.argsort(freq_mesh)
    
    freq_mesh = freq_mesh[sort_indices]
    power_spectrum = power_spectrum[sort_indices]
    # cut the vector at the nyquist frequency
    fmax = np.max(freq_mesh)/2
    power_spectrum = power_spectrum[freq_mesh<fmax]
    freq_mesh = freq_mesh[freq_mesh<fmax] 
    
    f, p = section_and_average(freq_mesh, power_spectrum, n_sections=50)
    
    # Increase the default font size
    plt.rcParams.update({'font.size': 18})
    
    plt.figure(figsize=[10,6])
    plt.plot(freq_mesh, power_spectrum, linewidth=0.4, label='Power Spectrum')
    plt.plot(f, p, '-o', markersize=3, linewidth=3, label='Averaged Power Spectrum')
    plt.plot(f, C*(f)**(-5/3), '--', linewidth=3, label=r'$\propto k^{-5/3}$', c='grey') 
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    
    return

def section_and_average(x, y, n_sections):
    """
    Divides the given array into sections and computes the average for each section.

    Parameters:
    -----------
    x : np.ndarray
        Array of x coordinates.
    y : np.ndarray
        Array of y values.
    n_sections : int
        Number of sections to divide the data into.

    Returns:
    --------
    tuple of np.ndarray
        Tuple containing the averaged x and y values.
    """
    # Define the sections
    sections = np.linspace(x.min(), x.max(), n_sections+1)
    
    # Find the section each x value falls into
    section_indices = np.digitize(x, sections)
    
    # Calculate the section centers
    section_centers = (sections[:-1] + sections[1:]) / 2
    
    # Calculate the mean y value for each section
    section_means = np.array([y[section_indices == i].mean() for i in range(1, len(sections))])
    
    return section_centers, section_means


# %% Process data in chuncks
def read_variable_in_chunks(file_path, chunk_size):
    """
    Generator to read a binary file in chunks.

    Parameters:
    -----------
    file_path : str
        Path to the binary file.
    chunk_size : int
        Number of elements to read per chunk.

    Returns:
    --------
    np.ndarray
        Array of data read from the file.
    """
    with open(file_path, 'rb') as file:
        while True:
            # Read a chunk of data
            data_chunk = np.fromfile(file, dtype='<f4', count=chunk_size)
            # If no more data to read, break the loop
            if len(data_chunk) == 0:
                break
            yield data_chunk

def process_species_in_chunks(file_paths, species_file, chunk_size):
    """
    Reads species mass fractions in chunks from the specified files.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary containing file paths.
    species_file : list
        List of species file names.
    chunk_size : int
        Number of elements to read per chunk.
    
    Returns:
    --------
    list
        List of lists containing chunks of species data.
    """
    species_data_chunks = []
    for specie_file in species_file:
        species_data_chunks.append([])
        for data_chunk in read_variable_in_chunks(file_paths['folder_path'] + file_paths['data_path'] + specie_file, chunk_size):
            species_data_chunks[-1].append(data_chunk)
    return species_data_chunks

