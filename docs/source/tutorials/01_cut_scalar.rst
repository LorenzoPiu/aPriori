Tutorial 1: Scalar class basic usage
=======================================

.. note::

   The complete code associated with this tutorial is available
   `here <https://github.com/LorenzoPiu/aPriori/blob/main/tutorials/01-cut_scalar.py>`_.


The goal of this tutorial is to demonstrate how the ``Scalar`` class works in
both its *standard mode* and *light mode*. Understanding these two approaches
will make it easier to work with large 3D arrays efficiently.

.. note::

   To run the full tutorial (including the DNS example), make sure you have
   downloaded the ``data`` folder from the GitHub repository.

Import the Module and Define an Array
-------------------------------------

First, import the required modules. Ensure that you have installed the package
in your environment, otherwise go back to the :doc:`installation guide <installation>`

Then:

.. code-block:: python

   from aPriori.DNS import Scalar
   import numpy as np

Define a generic 3D array that we will use as a scalar field:

.. code-block:: python

   shape = [30, 20, 15]
   array = np.random.rand(*shape)

Define a ``Scalar`` Object
----------------------------

You can initialize a ``Scalar`` object directly from a NumPy array:

.. code-block:: python

   scalar = Scalar(shape=shape, value=array)
   print(f"Initial shape:\n{scalar.shape}")

This prints:

.. code-block:: text

   Initial shape:
   (30, 20, 15)

Cutting the Scalar Field
------------------------

Cut the field using the ``equal`` mode. This is useful after filtering
operations or any situation where boundary values should be removed.

Example: cut 2 cells from each side along every axis:

.. code-block:: python

   scalar.cut(n_cut=2, mode="equal")
   print(f"Shape after 'equal' cut:\n{scalar.shape}")

Result:

.. code-block:: text

   Shape after 'equal' cut:
   (26, 16, 11)

Using ``Scalar`` in Light Mode
--------------------------------

We now use a real DNS dataset to demonstrate the *light mode*.  
In this mode, the object stores only a pointer to the file on disk and loads the
data **only when needed**, which prevents memory overload.

This mode activates when you **omit the ``value`` argument** and provide only
the ``path`` to a data file.

Set up the path to the dataset:

.. code-block:: python

   import os

   # Change this path according to where you placed the data folder
   directory = os.path.join("..", "data", "Lifted_H2_subdomain")
   T_path = os.path.join(directory, "data", "T_K_id000.dat")

   print(f"Checking the path '{T_path}' is correct...")

   if not os.path.exists(T_path):
       raise ValueError(
           f"The path '{T_path}' does not exist. "
           "Please verify the location of your data folder."
       )

The BlastNet DNS data include information about the global shape inside
``info.json``:

.. code-block:: python

   import json

   with open(os.path.join(directory, "info.json"), "r") as file:
       info = json.load(file)

   DNS_shape = info["global"]["Nxyz"]

Now we can define the ``Scalar`` object in light mode:

.. code-block:: python

   T = Scalar(shape=DNS_shape, path=T_path)

Access some values:

.. code-block:: python

   T_values = T._3D   # returns the full 3D numpy array
   print("Temperature at (10, 10, 10):")
   print(T_values[10, 10, 10])

Example output:

.. code-block:: text

   Temperature at (10, 10, 10):
   905.779

Comparing Memory Usage
----------------------

Since ``Scalar`` stores only a pointer to the file, its memory usage is
minimal—even for very large DNS fields.

Compare with a full NumPy array of the same size:

.. code-block:: python

   T_numpy = np.random.rand(*DNS_shape)

   import sys
   T_size = sys.getsizeof(T)
   T_numpy_size = sys.getsizeof(T_numpy)

   print(
       f"\nSize of the numpy array:     {T_numpy_size} bytes\n"
       f"Size of the Scalar object: {T_size} bytes"
   )

Typical output:

.. code-block:: text

   Size of the numpy array:     32000144 bytes
   Size of the Scalar object: 56 bytes

Cutting DNS Fields in Light Mode
--------------------------------

You can also cut a field in light mode.  
Use ``mode="xyz"`` to cut a *different number of cells in each direction*.  
Each cut is applied symmetrically at the beginning and end of the axis.

Example: cut 30 cells in ``x``, 0 in ``y``, and 10 in ``z``:

.. code-block:: python

   T_cut = T.cut(n_cut=[30, 0, 10], mode="xyz")

   print(f"Original shape: {T.shape}")
   print(f"After 'xyz' cut: {T_cut.shape}")

Typical output:

.. code-block:: text

   Original shape: (200, 200, 100)
   After 'xyz' cut: (140, 200, 80)