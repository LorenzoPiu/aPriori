Tutorial 6: Favre filtering for variable density flows
======================================================

.. note::

   The complete code associated with this tutorial is available
   `here <https://github.com/LorenzoPiu/aPriori/blob/main/tutorials/06-filter_DNS_field.py>`_.

This tutorial introduces the Favre filtering utilities available in the library.
Favre filtering is a density-weighted spatial filtering operation that is
widely used in large-eddy simulation (LES) of compressible and reacting flows
to separate resolved-scale quantities from sub-filter-scale contributions.

For a generic scalar quantity :math:`\phi`, the Favre-filtered field
:math:`\widetilde{\phi}` is defined as

.. math::

   \widetilde{\phi} = \frac{\overline{\rho \phi}}{\overline{\rho}},

where :math:`\rho` is the density field and the overbar denotes a spatial
filtering operation. This definition ensures that the filtering operation
is consistent with the conservative form of the governing equations in
variable-density flows.

In the following example, Favre filtering is applied to a DNS dataset using
different filter kernels and filter sizes, generating new filtered fields
that can be analyzed or visualized in the same way as the original DNS data.

Import modules and define data path
-----------------------------------

.. code-block:: python
   :caption: Imports and paths definition

   import os
   from aPriori.DNS import Field
   from aPriori import DNS
   import json

   directory = os.path.join('Lifted_H2_subdomain') # change this with your path to the data folder
   T_path = os.path.join(directory,'data', 'T_K_id000.dat')
   print(f"\nChecking the path {T_path} is correct...")
   if not os.path.exists(T_path):
      print(f"The path '{T_path}' does not exist in your system. Downloading the dataset from Github...")
      ap.download(dataset='h2_lifted')
      

Apply Favre filtering
---------------------

.. container:: demo

   .. code-block:: python
      :caption: Favre filtering with Gaussian and box kernels

      # Load the DNS field
      DNS_field = Field(directory)

      # Define the filter size (in grid points)
      filter_size = 32

      # Apply Favre filtering using a Gaussian kernel
      field_filt_name = DNS_field.filter_favre(
          filter_size,
          filter_type='Gauss'
      )
      DNS_field_filt_gauss = Field(field_filt_name)

      # Apply Favre filtering using a box kernel
      # The box filter can take a long time as the convolution operation is not
      # optimized. If you have any suggestion on how to improve it, please
      # open an issue on the Github page!
      DNS_field_filt_box = Field(
          DNS_field.filter_favre(filter_size, filter_type='Box')
      )

   .. container:: demo-result

      .. figure:: /_static/figures/fundamentals_and_usage/tutorial_06_fig1.png
         :width: 90%
         :align: center

         Figure 1 - Example of a Favre-filtered field using a box and a Gaussian kernel.
