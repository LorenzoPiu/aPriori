aPriori.NN.TrainingBuilder
==========================

.. py:class:: aPriori.NN.TrainingBuilder(*args, **kwargs)

   Bases: :py:obj:`dict`


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


   .. py:method:: get_subset(keys)

      Return a subset of the TrainingBuilder based on the specified keys.

      Parameters:
      -----------
          - keys (list):
              The keys to include in the subset.

      Returns:
      --------
      :
          - TrainingBuilder:
              A new TrainingBuilder containing only the specified keys.



   .. py:method:: add(variable, *args, **kwargs)

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



   .. py:method:: build_x(field)

      Construct a feature matrix by transforming data from a Field instance.

      Parameters:
      -----------
          - field (Field): The input field containing data to scale.

      Returns:
      --------
      :
          - np.ndarray: A 2D array containing the scaled data for all variables in the TrainingBuilder.

      Raises:
      -------
          - KeyError: If the field is not an instance of Field.



   .. py:method:: fit(field)

      Fit each VectorScaler in the TrainingBuilder to the corresponding data in the Field or list of Field objects.

      Parameters:
      -----------
          - field (Field or list of Field): The field(s) with data to fit each scaler.



   .. py:method:: load(file_path)

      Load the state of a previously saved object from a JSON file and reinitialize the TrainingBuilder.

      Parameters:
      -----------
          - file_path (str): The path to the JSON file to load.

      Raises:
      -------
          - ValueError: If the file is not a valid JSON file.



   .. py:method:: save(file_path)

      Save the current state of the TrainingBuilder to a JSON file.

      Parameters:
      -----------
          - file_path (str): The path to save the JSON file.



   .. py:property:: state_dict

      Return the current state of the TrainingBuilder, including all scaler states.

      Returns:
      --------
          - dict: A dictionary representing the state of all scalers in the TrainingBuilder.

