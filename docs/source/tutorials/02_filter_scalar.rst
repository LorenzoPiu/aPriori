Tutorial 2: Filtering numpy arrays
==================================

.. note::

   The complete code associated with this tutorial is available
   `here <https://github.com/LorenzoPiu/aPriori/blob/main/tutorials/02-filter_scalar.py>`_.


In this tutorial you will learn how to apply spatial filters to a 3D field
using the :class:`Scalar3D` class.  
Make sure you have downloaded the data folder from the GitHub repository
before starting.

Introduction
------------

This example shows two use cases:

1. Filtering a **synthetic 3D NumPy array**  
2. Filtering a **DNS scalar field** loaded in ``light_mode`` using pointers

The goal is to demonstrate how filtering works, and how ``Scalar3D`` simplifies
operations on large DNS fields.

Import modules
--------------

First, import the modules required for this tutorial:

.. code-block:: python

    import numpy as np
    import aPriori.DNS as DNS
    from aPriori.DNS import Scalar3D

Filtering a synthetic array
---------------------------

Although ``Scalar3D`` is mainly used with DNS data on disk, it can also store
arrays directly for convenience.

Define a small test array:

.. code-block:: python

    shape = [4, 2, 4]
    array = np.ones(shape)
    array[2:3, :, 2:3] = 2
    print(array)

You should see:

.. code-block:: text

    [[[1. 1. 1. 1.]
      [1. 1. 1. 1.]]

     [[1. 1. 1. 1.]
      [1. 1. 1. 1.]]

     [[1. 1. 2. 1.]
      [1. 1. 2. 1.]]

     [[1. 1. 1. 1.]
      [1. 1. 1. 1.]]]

Apply a Gaussian filter:

.. code-block:: python

    filter_size = 2
    filtered_field = DNS.filter_3D(array, filter_size,
                                   favre=False, filter_type='Gauss')
    print(filtered_field)

Filtering DNS temperature data
------------------------------

To make a more realistic example, let's filter a temperature field loaded from
the DNS dataset using ``Scalar3D`` in *light mode*.

Locate the data directory and temperature file:

.. code-block:: python

    import os

    # Change this to the appropriate path on your machine
    directory = os.path.join("..", "data", "Lifted_H2_subdomain")
    T_path = os.path.join(directory, "data", "T_K_id000.dat")

    print(f"\nChecking the path '{T_path}' is correct...")

    if not os.path.exists(T_path):
        raise ValueError(
            f"The path '{T_path}' does not exist. "
            "Ensure you have downloaded the DNS dataset."
        )

The DNS folder contains an ``info.json`` file that stores the grid shape:

.. code-block:: python

    import json
    with open(os.path.join(directory, "info.json"), "r") as file:
        info = json.load(file)

    DNS_shape = info["global"]["Nxyz"]

Create the ``Scalar3D`` object:

.. code-block:: python

    T = Scalar3D(shape=DNS_shape, path=T_path)

Access some values:

.. code-block:: python

    print("Temperature values in the cells 5:8, 5:8, 5:8")
    print(T._3D[5:8, 5:8, 5:8])

Filter the scalar field using a Gaussian kernel:

.. code-block:: python

    filter_size = 20
    T_filt = DNS.filter_3D(T._3D, filter_size,
                           favre=False, filter_type='Gauss')

    print("Filtered temperature values in the cells 5:8, 5:8, 5:8")
    print(T_filt[5:8, 5:8, 5:8])

You can also use a **box filter**:

.. code-block:: python

    T_filt_box = DNS.filter_3D(T._3D, filter_size,
                               favre=False, filter_type='Box')

    print("Filtered temperature values in the cells 5:8, 5:8, 5:8")
    print(T_filt_box[5:8, 5:8, 5:8])

Conclusion
----------

This example showed how to:

- Apply 3D Gaussian and box filters to NumPy arrays
- Load DNS temperature data using ``Scalar3D`` in lightweight mode
- Filter DNS fields without loading the entire dataset into memory

To better visualize the effects of filtering, consider completing the next two
tutorials, where filtering is applied and plotted on real DNS fields.