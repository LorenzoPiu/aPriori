Tutorial 3: Read a DNS dataset
===============================

.. note::

   The complete code associated with this tutorial is available
   `here <https://github.com/LorenzoPiu/aPriori/blob/main/tutorials/03-initialize_DNS_field.py>`_.

This exercise introduces the :class:`Field` class, which highlights the main
purpose of **aPrioriDNS**: providing a clean, memory-efficient interface for
working with formatted DNS datasets.

From this point onward, the tutorials assume you are working with properly
formatted DNS data (e.g., BlastNet).  
If you want to run the tutorial locally, download the dataset from the
aPrioriDNS GitHub repository using the function ``ap.download()``.  
If you already downloaded the data, simply comment out that line.

Import modules and define data path
-----------------------------------

.. code-block:: python
   :caption: Script header

   import os
   import aPriori as ap
   import json

   # Download data (comment this if already downloaded)
   ap.download(dataset='h2_lifted')

   # Adjust this path depending on where the DNS dataset is stored
   directory = os.path.join('.', 'Lifted_H2_subdomain')

   # BlastNet stores the global shape in info.json.
   # Field will automatically parse everything, so we don't need to load it manually.
   # The following lines check that the dataset is present in your system
   with open(os.path.join(directory, 'info.json'), 'r') as file:
       info = json.load(file)
   DNS_shape = info['global']['Nxyz']

Initialize the Field object
-----------------------------

If your dataset follows the **BlastNet format**, you only need to provide the
correct folder path. ``Field`` will automatically:

- read the mesh,
- load the chemistry mechanism,
- detect all scalar/vector fields available,
- create memory-light pointers to the data files.

.. code-block:: python

   DNS_field = ap.Field(directory)

Example console output
----------------------

.. code-block:: text

   ---------------------------------------------------------------
   Initializing 3D Field

   Checking files inside folder ../data/Lifted_H2_subdomain...

   Folder structure OK

   ---------------------------------------------------------------
   Building mesh attribute...
   Mesh fields read correctly

   ---------------------------------------------------------------
   Reading kinetic mechanism...
   Kinetic mechanism file found: ../data/Lifted_H2_subdomain/chem_thermo_tran/li_h2.yaml
   Species:
   ['H2', 'O2', 'H2O', 'H', 'O', 'OH', 'HO2', 'H2O2', 'N2']

   ---------------------------------------------------------------
   Building scalar attributes...
   Field attributes:
   +-----------+------------------------------------------------------+
   | Attribute |                         Path                         |
   +-----------+------------------------------------------------------+
   |     P     |   ../data/Lifted_H2_subdomain/data/P_Pa_id000.dat    |
   |    RHO    | ../data/Lifted_H2_subdomain/data/RHO_kgm-3_id000.dat |
   |     T     |    ../data/Lifted_H2_subdomain/data/T_K_id000.dat    |
   |    U_X    |  ../data/Lifted_H2_subdomain/data/UX_ms-1_id000.dat  |
   |    U_Y    |  ../data/Lifted_H2_subdomain/data/UY_ms-1_id000.dat  |
   |    U_Z    |  ../data/Lifted_H2_subdomain/data/UZ_ms-1_id000.dat  |
   |    YH2    |    ../data/Lifted_H2_subdomain/data/YH2_id000.dat    |
   |    YO2    |    ../data/Lifted_H2_subdomain/data/YO2_id000.dat    |
   |   YH2O    |   ../data/Lifted_H2_subdomain/data/YH2O_id000.dat    |
   |    YH     |    ../data/Lifted_H2_subdomain/data/YH_id000.dat     |
   |    YO     |    ../data/Lifted_H2_subdomain/data/YO_id000.dat     |
   |    YOH    |    ../data/Lifted_H2_subdomain/data/YOH_id000.dat    |
   |   YHO2    |   ../data/Lifted_H2_subdomain/data/YHO2_id000.dat    |
   |   YH2O2   |   ../data/Lifted_H2_subdomain/data/YH2O2_id000.dat   |
   |    YN2    |    ../data/Lifted_H2_subdomain/data/YN2_id000.dat    |
   +-----------+------------------------------------------------------+

Using the Field object
------------------------

Once initialized, the ``Field`` object exposes all field variables as
``Scalar`` objects.

Access a scalar field:
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DNS_field.T

which returns a flattened numpy array. This form is generally useful when computing
statistics and is easy to manage, hence was chosen as the default value to be returned 
when accessing objects of the `Scalar` class.

.. note::

   The command `DNS_field.T` by default accesses the attribute `T` of the object,
   which belongs to the class `Scalar`. When calling it, the `value` attribute of
   the object is returned, so that the commands `DNS_field.T`, and `DNS_field.T.value`
   return the same array.

To access the temperature values reshaped as a column vector with shape [N_elements, 1]:

.. code-block:: python

   DNS_field.T.reshape_column()

or reshaped as a line vector with shape [1, N_elements]:

.. code-block:: python

   DNS_field.T.reshape_line()

Access the reshaped 3D data:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DNS_field.U_Y._3D

This returns the NumPy array of the *Y* velocity component reshaped in 3D.

To extract only the x midplane of the *Y* velocity componene:

.. code-block:: python

   DNS_field.U_Y.x_midplane

or for the y and z midplanes:

.. code-block:: python

   DNS_field.U_Y.y_midplane
   DNS_field.U_Y.z_midplane

.. tip::
   It's been a while I want to add a method that extracts a different midplane
   (not necessarily the mid one). It's a relatively fast update so I hope I'll find the
   time to add ths soon. However, if while you read this the method is not available yet,
   and you want to help this project, you're very welcome to :doc:`contribute <contribute>` 🫶

Filtering a field
~~~~~~~~~~~~~~~~~

You may apply any filtering operation (e.g., Gaussian, box, or Favre filtering):

.. code-block:: python

   filt_YO2 = DNS.filter_3D(DNS_field.YO2._3D, 8)

Plotting a midplane slice
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DNS_field.plot_z_midplane('YH2O2')

3D visualisation and further examples will be covered in the next tutorials.