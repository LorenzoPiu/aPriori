Tutorial 5: Cut DNS field
==============================

.. note::

   The complete code associated with this tutorial is available
   `here <https://github.com/LorenzoPiu/aPriori/blob/main/tutorials/05-cut_DNS_field.py>`_.

This tutorial shows how to extract a sub-domain from a DNS dataset using the
:py:meth:`~aPrioriDNS.DNS.Field.cut` method. After cutting the field, we load
the resulting dataset as a new :py:class:`~aPrioriDNS.DNS.Field` object and
compare a representative midplane of a variable (here :math:`U_x`) between the
original and the cut domain.

Import modules and define data path
-----------------------------------

.. code-block:: python
   :caption: Imports and paths definition

   import os
   import aPriori as ap
   from aPriori.DNS import Field
   from aPriori import DNS
   import json
   
   directory = os.path.join('Lifted_H2_subdomain') # change this with your path to the data folder
   T_path = os.path.join(directory,'data', 'T_K_id000.dat')
   print(f"\nChecking the path \'{T_path}\' is correct...")
   if not os.path.exists(T_path):
      print(f"The path '{T_path}' does not exist in your system. Downloading the dataset from Github...")
      ap.download(dataset='h2_lifted')
      

Cut the DNS field and compare midplanes
---------------------------------------

.. container:: demo

   .. code-block:: python
      :caption: Cut the field and visualize original vs cut domain

      # Load the full DNS field
      DNS_field = Field(directory)

      # Cut a sub-domain (example indices along x, y, z)
      cut_field_name = DNS_field.cut([20, 20, 10])

      # Load the cut field as a new Field object
      DNS_field_cut = Field(cut_field_name)

      # Compare the same variable on the z midplane
      DNS_field.plot_z_midplane('U_X', vmin=100, vmax=280)
      DNS_field_cut.plot_z_midplane('U_X', vmin=100, vmax=280)

   .. container:: demo-result

      .. figure:: /_static/figures/fundamentals_and_usage/tutorial_05_fig1.png
         :width: 50%
         :align: center

         Figure 1 - :math:`U_x` on the z midplane for the original DNS field.

      .. figure:: /_static/figures/fundamentals_and_usage/tutorial_05_fig2.png
         :width: 50%
         :align: center

         Figure 2 - :math:`U_x` on the z midplane for the cut DNS field.