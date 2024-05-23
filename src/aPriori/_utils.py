#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:00:31 2024

@author: lorenzo piu
"""

import os
import re
from ._data_struct import folder_structure

class DataStructureException(Exception):
    pass

def check_data_files(main_folder):
    """
    Check the data files inside the specified main folder.

    Parameters:
    - main_folder (str): The path to the main folder containing the data files.

    Raises:
    - DataStructureException: If any of the following conditions are not met:
        - The data folder does not exist.
        - Any file in the data folder does not have a .dat extension.
        - Any file in the data folder does not have a proper id in the name format.
        - The ids in the file names are not all the same.

    Returns:
    - True: If all checks pass without raising an exception.
    - ids: The id of the files. In this version of the code we just allow 1 id for every file
    
    Note:
        The function is to be asjusted because it uses sets, that are optimal to check repetition
        but not to return values
    """
    print("Checking files inside folder " + main_folder + "...\n")
    data_folder = os.path.join(main_folder, folder_structure["data_path"])
    if not os.path.exists(data_folder) or not os.path.isdir(data_folder):
        raise DataStructureException(f"The folder {data_folder} does not exist")

    ids = set()  # Store unique ids
    for file_name in os.listdir(data_folder):
        if not file_name.startswith('.DS_Store'): # Exclude this file because it's an hidden file inside the mac-OS environment
            if not file_name.endswith(".dat"):
                raise DataStructureException(f"File '{file_name}' does not have a .dat extension.")
            
            # Extracting the id part before .dat extension
            id_part = file_name.split("_")[-1].split(".")[0]
            if not id_part.startswith("id") or not id_part[2:].isdigit():
                raise DataStructureException(f"File '{file_name}' does not have a proper id in the name format.")
            
            ids.add(id_part)

    # Check if all ids are the same
    if len(ids) != 1:
        raise DataStructureException("Not all files have the same id.")

    return True, id_part


def check_folder_structure(folder_path):
    """
    Check if the folder structure is coherent with the blastnet one.

    Raises:
        FileNotFoundError: If 'info.json' file is not found.
        DataStructureException: If the folder structure is not coherent.
        
    Returns:
        True: if all the tests are passed without an exception
    """
    chem_path = folder_structure["chem_path"]
    data_path = folder_structure["data_path"]
    grid_path = folder_structure["grid_path"]
    
    required_folders = [folder_structure["chem_path"], folder_structure["data_path"], folder_structure["grid_path"]]
    my_folder_structure = os.listdir(folder_path)

    for folder in required_folders:
        if folder not in my_folder_structure:
            raise DataStructureException("The folder structure is not coherent. '" + folder + "' folder is missing.\n"
             "Check that the folder structure is coherent with the following:\n"
             "<folder_name>\n"
            f"├──── {chem_path}\n"
             "│     └──── <kinetic_mechanism>.yaml\n"
            f"├──── {data_path}\n"
             "│     ├──── T_K_id000.dat\n"
             "│     └──── ...\n"
            f"├──── {grid_path}/\n"
             "│     ├──── X_m.dat\n"
             "│     ├──── Y_m.dat\n"
             "│     └──── Z_m.dat\n"
             "└──── info.json")
            
            
    if 'info.json' not in my_folder_structure:
        raise FileNotFoundError("The 'info.json' file is missing in the main folder.")

    return True

def extract_species(file_path):
    import yaml
    try:
        with open(file_path, 'r') as file:
            mechanism_data = yaml.safe_load(file)
            species_names = mechanism_data.get('phases', ['species'])
            return species_names[0]['species']
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None   

def find_kinetic_mechanism(folder_path):
    """
    Finds the first YAML file containing a kinetic mechanism in the specified folder.

    Parameters:
    - folder_path (str): The path to the folder to search for the YAML file.

    Returns:
    - str or None: The path to the first YAML file found, or None if no YAML files are found or if the folder does not exist.
    """
    try:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)
        
        # Filter files with .yaml extension
        yaml_files = [file for file in files if file.endswith('.yaml')]
        
        # Sort the filtered list alphabetically
        yaml_files.sort()
        
        if yaml_files:
            # Return the path to the first file in the sorted list
            return os.path.join(folder_path, yaml_files[0])
        else:
            print("No .yaml files found in the folder.")
            return None
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return None


def extract_filter(folder_path):
    """
    Extracts an integer number following the string 'Filter' (case insensitive) from a folder path.

    Args:
        folder_path (str): The folder path to search for the 'Filter' string and integer number.

    Returns:
        int or None: The integer number following the 'Filter' string if found, or None if not found.

    Example:
        >>> folder_path = "Path/To/Some/Filter123/Folder"
        >>> number = extract_filter(folder_path)
        >>> print(number)
        123
    """
    pattern = re.compile(r'filter(\d+)', re.IGNORECASE)

    # Search for the pattern in the folder path
    match = pattern.search(folder_path)

    if match:
        # Extract the integer number
        filter_number = int(match.group(1))
        return filter_number
    else:
        return 1
    

def change_folder_name(folder_path, new_name):
    """
    Change the name of the last folder in a given folder path.

    Parameters:
        folder_path (str): The original folder path.
        new_name (str): The new name for the last folder.

    Returns:
        str: The updated folder path with the new folder name.
    
    Example:
        >>> change_folder_name('../data/DNS_DATA_TEST', 'new_folder_name')
        '../data/new_folder_name'
    """
    # Keep splitting the folder path until we get to the last folder name
    while True:
        directory, folder_name = os.path.split(folder_path)
        if folder_name:
            break
        folder_path = directory
    
    # Replace the last folder name with the new name
    new_folder_path = os.path.join(directory, new_name)
    
    return new_folder_path
 

def check_mass_fractions(attr_list, bool_list, folder_path):
    """
    Check if all the species mass fractions are present in the given folder path.

    This function iterates through a list of attributes and their corresponding 
    presence indicators. It checks if an attribute represents a species mass fraction 
    by inspecting its name. Currently, it simply checks if the attribute name starts 
    with 'Y', but this method can be improved in the future. If any mass fraction 
    attribute is found to be missing, it raises a DataStructureException. Otherwise, 
    it returns True.

    Parameters:
        attr_list (list): A list of attribute names.
        bool_list (list): A list of boolean indicators representing the presence of attributes.
        folder_path (str): The path of the folder containing the data.

    Raises:
        DataStructureException: If any species mass fraction is missing in the folder.

    Returns:
        bool: True if all species mass fractions are present, otherwise False.
    """
    for attribute, is_present in zip(attr_list, bool_list):
        if attribute[0] == 'Y' :  #TODO: now I am using a simple method, 
        # but it must be improved. Now I just check if the attribute
        # starts with 'Y' to understand if it's a specie mass fraction,
        # but we can do better.
            if not is_present:
                raise DataStructureException("Not all the species mass fractions are present in the folder '{folder_path}'. \n")
            else:
                return True
            
def check_reaction_rates(attr_list, bool_list, folder_path):
    """
    Check if reaction rates are present in the given folder path.

    This function performs the following steps:
    1. Determine whether to compute DNS (Direct Numerical Simulation) or 
       LFR (Laminar Finite Rate) rates based on the filter size extracted 
       from the folder path.
    2. Check if any of the reaction rates that will be computed in the main 
       function are already present in the folder.

    Parameters:
        attr_list (list): A list of attribute names.
        bool_list (list): A list of boolean indicators representing the presence of attributes.
        folder_path (str): The path of the folder containing the data.

    Returns:
        bool: True if reaction rates are present and match the computation mode, otherwise False.

    Notes:
        - The function currently identifies reaction rates by checking if the attribute name starts with 'R'. This method may be improved in the future.
        - The mode ('DNS' or 'LFR') is determined based on the filter size extracted from the folder path.

    Example:
        >>> attr_list = ['R1', 'R2', 'R3', 'S1']
        >>> bool_list = [True, True, False, True]
        >>> folder_path = '/path/to/data'
        >>> check_reaction_rates(attr_list, bool_list, folder_path)
        True
    """
    # Step 1:
    # Understand if we are going to compute the DNS or LFR rates
    filter_size = extract_filter(folder_path)
    if filter_size == 1:
        mode = 'DNS'
    else:
        mode = 'LFR'
    
    # Step 2:
    # check if some of the Reaction Rates that will be computed in the
    # main function are already present in the folder
    for attribute, is_present in zip(attr_list, bool_list):
        if attribute[0] == 'R' :  #TODO: now I am using a simple method, 
        # but it must be improved. Now I just check if the attribute
        # starts with 'R' to understand if it's a Reactio Rate,
        # but we can do better.
            if is_present and mode in attribute:
                return True
            else:
                return False

