#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A collection of training data building utilities  
"""

__authors__ = "Lorenzo Piu"
__copyright__ = "Copyright (c) 2024-2025, Lorenzo Piu, Heinz Pitsch, and Alessandro Parente"
__credits__ = ["Aero-Thermo-Mechanics laboratories - Universite Libre de Bruxelles, Brussels, Belgium"
               "Institut für Technische Verbrennung (ITV) - RWTH Aachen University, Aachen, Germany"]
__license__ = "GPL 3.0"
__version__ = "1.2.3"
__maintainer__ = ["Lorenzo Piu"]
__email__ = ["lorenzo.piu@ulb.be"]
__status__ = "Production"


import json
import os
from .DNS import Field
import numpy as np
  
class TrainingBuilder(dict):
    """
    Description:
    ------------
    
    A dictionary-like class that builds and manages a collection of VectorScaler instances for training purposes.

    This class includes methods to add, configure, and save/load multiple VectorScaler objects, 
    and allows batch scaling of data based on a specified Field input.
    
    Attributes:
    -----------
        - state_dict (dict): A dictionary containing the state of all scalers in the TrainingBuilder.
        
    Methods:
    --------
        - __init__(*args, **kwargs):
            Initialize the TrainingBuilder with optional dictionary arguments.

        - get_subset(keys):
            Return a subset of the TrainingBuilder based on the specified keys.

        - __setitem__(key, value):
            Set an item in the TrainingBuilder with key and value type validation.

        - add(variable, *args, **kwargs):
            Add a new VectorScaler to the TrainingBuilder.

        - build_x(field):
            Construct a feature matrix by transforming data from a Field instance.

        - fit(field):
            Fit each VectorScaler in the TrainingBuilder to the corresponding data in the Field.

        - load(file_path):
            Load the state of each scaler from a JSON file and reinitialize the TrainingBuilder.

        - save(file_path):
            Save the current state of the TrainingBuilder to a JSON file.

        - state_dict:
            Return the current state of the TrainingBuilder, including all scaler states.

        - __str__():
            Return a string representation of the TrainingBuilder.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the TrainingBuilder with optional dictionary arguments.

        Parameters:
        -----------
            - *args: 
                Variable length argument list. Must be coherent with the input arguments used
                in the initialization of a dictionary.
                
            - **kwargs: 
                Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def get_subset(self, keys):
        """
        Return a subset of the TrainingBuilder based on the specified keys.

        Parameters:
        -----------
            - keys (list): 
                The keys to include in the subset.

        Returns:
        --------
            - TrainingBuilder: 
                A new TrainingBuilder containing only the specified keys.
        """
        return TrainingBuilder({k: self[k] for k in keys if k in self})

    def __setitem__(self, key, value):
        """
        Set an item in the TrainingBuilder with key validation.

        Parameters:
        -----------
            - key (str): 
                The key for the item.
                
            - value (VectorScaler): 
                The VectorScaler object to add.

        Raises:
        -------
            - KeyError: If the key is not a string.
            - ValueError: If the value is not an instance of VectorScaler.
        """
        if not isinstance(key, str):
            raise KeyError("Keys must be strings.")
        if not isinstance(value, VectorScaler):  # Example type restriction
            raise ValueError("Values must be int or str.")
        super().__setitem__(key, value)

    def add(self, variable, *args, **kwargs):
        """
        Add a new VectorScaler to the TrainingBuilder.

        Parameters:
        -----------
            - variable (str): 
                The key to associate with the VectorScaler.
                
            - *args: 
                Arguments to pass to the VectorScaler initializer.
            
            - **kwargs: 
                Keyword arguments to pass to the VectorScaler initializer.

        Raises:
        -------
            - KeyError: 
                If the specified variable already exists in the TrainingBuilder.
        """
        if variable in self:
            raise KeyError(f"Key '{variable}' already exists.")
        
        value = VectorScaler(*args,**kwargs)
        self[variable] = value  # Calls __setitem__
        
    def build_x(self, field):
        """
        Construct a feature matrix by transforming data from a Field instance.

        Parameters:
        -----------
            - field (Field): The input field containing data to scale.

        Returns:
        --------
            - np.ndarray: A 2D array containing the scaled data for all variables in the TrainingBuilder.

        Raises:
        -------
            - KeyError: If the field is not an instance of Field.
        """
        def append_column(array_2d, column):
            column = np.array(column).reshape(-1, 1)  # Reshape to a 2D column
            return np.hstack((array_2d, column)) if array_2d.size else column  # Append column or return the new column

        # Check type
        if not isinstance(field, Field):
            raise KeyError("Keys must be strings.")
            
        # Start looping through the variables
        X = np.empty((0,0))
        for variable_name, scaler in self.items():
            scalar = getattr(field, variable_name).value
            scaler.fit(scalar)
            x = scaler.transform(scalar)
            X = append_column(X, x)
            
        return X
    
    # def fit(self, field):
    #     """
    #     Old function, only able to handle one field at a time.
    #     Delete after testing the new utility.
    #     Fit each VectorScaler in the TrainingBuilder to the corresponding data in the Field.

    #     Parameters:
    #     -----------
    #         - field (Field): The field with data to fit each scaler.
    #     """
    #     for variable_name in self.keys():
    #         self[variable_name].fit(getattr(field, variable_name).value)
    
    def fit(self, field):
        """
        Fit each VectorScaler in the TrainingBuilder to the corresponding data in the Field or list of Field objects.

        Parameters:
        -----------
            - field (Field or list of Field): The field(s) with data to fit each scaler.
        """
        # Handle single field case
        if isinstance(field, Field):
            fields = [field]
        # Handle list of fields case
        elif isinstance(field, list) and all(isinstance(f,Field) for f in field):
            fields = field
        else:
            raise TypeError("Input must be a Field object or a list of Field objects")
        
        # Collect data from all fields for each variable
        for variable_name in self.keys():
            # Gather data from all fields for this variable
            all_data = []
            for f in fields:
                variable_data = getattr(f, variable_name).value
                all_data.append(variable_data.flatten())  # Flatten to 1D array
            
            # Concatenate all data for this variable
            combined_data = np.concatenate(all_data)
            
            # Fit the scaler to the combined data
            self[variable_name].fit(combined_data)
            
    def load(self, file_path):
        """
        Load the state of a previously saved object from a JSON file and reinitialize the TrainingBuilder.

        Parameters:
        -----------
            - file_path (str): The path to the JSON file to load.

        Raises:
        -------
            - ValueError: If the file is not a valid JSON file.
        """
        # Check that the file is a json
        if not is_json_file(file_path):
            raise ValueError(f"The file {file_path} is not a valid json file.")
            
        # Read the dictionary
        with open(file_path, 'r') as file:
            state_dict = json.load(file)
        
        # reinitialize object
        self.clear()
        for variable, state_dict_scaler in state_dict.items():
            scaler = VectorScaler()
            print(state_dict_scaler)
            scaler.load(state_dict_scaler)
            self[variable] = scaler
            
    def save(self, file_path):
        """
        Save the current state of the TrainingBuilder to a JSON file.

        Parameters:
        -----------
            - file_path (str): The path to save the JSON file.
        """
        with open(file_path, 'w') as file:
            json.dump(self.state_dict, file, indent=4)
        
    @property
    def state_dict(self):
        """
        Return the current state of the TrainingBuilder, including all scaler states.

        Returns:
        --------
            - dict: A dictionary representing the state of all scalers in the TrainingBuilder.
        """
        state_dict = dict()
        for key in self.keys():
            state_dict[key] = self[key].state_dict
        return state_dict

    def __str__(self):
        """
        Return a string representation of the TrainingBuilder.

        Returns:
        --------
            - str: The string representation.
        """
        return f"TrainingBuilder({super().__str__()})"
    

class VectorScaler():
    """
    Description:
    ------------
    A class for scaling and transforming vector data. This class supports various scaling
    modes, such as min-max scaling, standard scaling, and mean scaling, with additional options
    for modulus transformation, logarithmic scaling, and value clipping.
    
    The operations are performed in the following order:
        1. Modulus transformation
        2. Clipping
        3. Logarithmic transformation (with an automatic clipping at 1e-20 to avoid negative or zero values)
        4. Scaling based on the specified mode
    
    Attributes:
    -----------
        - mode (str): 
            The scaling mode. Options are 'minmax', 'standard', 'mean', or None.
            
        - modulus (bool): 
            Whether to apply the modulus (absolute value) operation.
        
        - log (bool): 
            Whether to apply logarithmic transformation.
        
        - vmin (float, optional): 
            Minimum value for clipping.
        
        - vmax (float, optional): 
            Maximum value for clipping.
        
        - copy (bool): 
            Whether to create a copy of the input array.
        
        - max, min, mean, std, ptp (float): 
            calculated scaling parameters, depending on mode.
        
    Methods:
    --------
        - __init__(mode='minmax', modulus=False, log=False, vmin=None, vmax=None, copy=True):
            Initialize the VectorScaler with scaling mode, transformation options, and optional clipping bounds.

        - _preprocess_input(x):
            Preprocess the input array by applying modulus, clipping, and logarithmic transformations.

        - _reset():
            Reset the scaler attributes (max, min, mean, std, ptp) to None. Used before recalculating parameters.

        - fit(x):
            Fit the VectorScaler to the data by calculating the necessary statistics based on the specified mode.

        - transform(x):
            Transform the input data based on the fitted scaling parameters and mode.

        - load(state_dict):
            Load saved state values into the VectorScaler.

        - state_dict:
            Return the current state of the VectorScaler, including mode and calculated parameters.
    """
    _lower_limit = 1e-30
    
    _args = ['mode', 'modulus', 'log', 'vmin', 'vmax', 'copy']
    _scalers = ['max', 'min', 'mean', 'std', 'ptp']
    
    def __init__(self, mode='minmax', modulus=False, log=False, vmin=None, vmax=None, copy=True):
        """
        Initialize the VectorScaler with scaling mode, transformation options, and optional clipping bounds.

        Parameters:
        -----------
            - mode (str): 
                The scaling mode. Options are 'minmax', 'standard', 'mean', or None.
                
            - modulus (bool): 
                Whether to apply modulus transformation.
                
            - log (bool): 
                Whether to apply logarithmic transformation.
                
            - vmin (float, optional): 
                Minimum value for clipping.
                
            - vmax (float, optional): 
                Maximum value for clipping.
                
            - copy (bool): 
                Whether to create a copy of the input array.
        """
        self.mode    = mode
        self.modulus = modulus
        self.log     = log
        self.vmin    = vmin
        self.vmax    = vmax
        self.copy    = copy
        
        self.max = None
        self.min = None
        self.mean = None
        self.std = None
        self.ptp = None
        
    def _preprocess_input(self, x):
        """
        Preprocess the input array by applying modulus, clipping, and logarithmic transformations.

        Parameters:
        -----------
            - x (array-like): 
                The input data to preprocess.

        Returns:
        --------
            - np.ndarray: 
                The preprocessed input as a 2D column vector.
        """
        # Copy if needed
        if self.copy:
            x = x.copy()
            
        # Convert to float array and reshape as a column vector
        x = np.array(x).flatten().astype(float)
        x = np.reshape(x, [len(x), 1])

        # Apply modulus if necessary
        if self.modulus:
            x = np.abs(x)

        # Clip values based on vmin and vmax
        if self.vmin is not None:
            x[x < self.vmin] = self.vmin
        if self.vmax is not None:
            x[x > self.vmax] = self.vmax

        # Apply logarithmic transformation if enabled
        if self.log:
            if any(x < VectorScaler._lower_limit):
                x[x < VectorScaler._lower_limit] = VectorScaler._lower_limit
            x = np.log10(x)
            
        return x
    
    def _reset(self):
        """
        Reset the scaler attributes (max, min, mean, std, ptp) to None. Used before recalculating parameters.
        """
        self.max = None
        self.min = None
        self.mean = None
        self.std = None
        self.ptp = None
    
    def fit(self, x):
        """
        Fit the VectorScaler to the data by calculating the necessary statistics based on the specified mode.

        Parameters:
        -----------
            - x (array-like): 
                The data to fit the scaler to.

        Raises:
        -------
            - ValueError: 
                If the mode is unrecognized.
        """
        self._reset()
        
        x = self._preprocess_input(x)
        
        if self.mode is None:
            pass
    
        elif self.mode.lower()=='minmax':
            self.max = np.max(x)
            self.min = np.min(x)
            
        elif self.mode.lower() == 'standard':
            self.mean = np.mean(x)
            self.std = np.std(x)
            
        elif self.mode.lower() == 'mean':
            self.mean = np.mean(x)
            self.ptp = np.ptp(x)
        
        else:
            raise ValueError(f"Unknown scaling mode {self.mode}")
            
    
    def transform(self, x):
        """
        Transform the input data based on the fitted scaling parameters and mode.

        Parameters:
        -----------
            - x (array-like): 
                The data to transform.

        Returns:
        --------
            - np.ndarray: 
                The scaled and transformed data.
        
        Raises:
        -------
            - ValueError: 
                If the mode is unrecognized.
        """
        x = self._preprocess_input(x)
        
        if self.mode is None:
            pass

        elif self.mode.lower()=='minmax':
            x = (x-self.min)/(self.max-self.min)
            
        elif self.mode.lower() == 'standard':
            x = (x - self.mean) / self.std
            
        elif self.mode.lower() == 'mean':
            # Scale between -1 and 1 using mean and range
            x = (x - self.mean) / (self.ptp / 2)
        
        else:
            raise ValueError(f"Unknown scaling mode {self.mode}")

        return x
    
    def load(self, state_dict):
        """
        Load saved state values into the VectorScaler.

        Parameters:
        -----------
            - state_dict (dict): 
                Dictionary containing scaler attributes to load.
        """
        for arg in state_dict.keys():
            setattr(self, arg, state_dict[arg])
    
    @property
    def state_dict(self):
        """
        Return the current state of the VectorScaler, including mode and calculated parameters.

        Returns:
        --------
            - dict: 
                A dictionary representation of the scaler's current state.
        """
        state_dict = dict()
        for arg in VectorScaler._args:
            state_dict[arg] = getattr(self, arg)
        for arg in VectorScaler._scalers:
            if getattr(self, arg) is not None:
                state_dict[arg] = getattr(self, arg)
        return state_dict
    

def is_json_file(file_path):
    """
    Check if the given file_path is a JSON file and if it contains valid JSON data.

    Parameters:
    -----------
        - file_path (str): The path to the file.

    Returns:
    --------
        - bool: True if the file is a valid JSON file, False otherwise.
    """
    # Check for .json file extension
    if not file_path.endswith('.json'):
        return False

    # Check if the file exists
    if not os.path.isfile(file_path):
        return False

    # Try to open and load the JSON content
    try:
        with open(file_path, 'r') as file:
            json.load(file)  # This will raise an error if the file is not valid JSON
    except (json.JSONDecodeError, OSError):
        return False

    return True
