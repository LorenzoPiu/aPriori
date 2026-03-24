aPriori.DNS.gradient_x
======================

.. py:function:: aPriori.DNS.gradient_x(F, mesh, filter_size=1)

   Description
   -----------

   Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
   used to compute the gradients on all the possible downsampled grids.

   Specifically, the parameter filter_size is used to temporarily downsample
   the grid in the x direction. The function considers one point each
   filter_size points and computes the derivatives on this downsampled grid.
   Does this for every possible downsampled grid, so in the end the output
   field has the same shape as the input field.

   :param Ur: Is the field to filter.
   :type Ur: Scalar object
   :param mesh: Is the mesh object used to compute the derivatives.
   :type mesh: Mesh object
   :param filter_size: Is the filter size used to filter the field.
   :type filter_size: int
   :param verbose: If True, it will output relevant information.
   :type verbose: bool

   :returns: **grad_x** -- The x component of the gradient
   :rtype: numpy array

