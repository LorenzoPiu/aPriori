aPriori.DNS.gradient_y
======================

.. py:function:: aPriori.DNS.gradient_y(F, mesh, filter_size=1)

   Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
   used to computed the gradients on all the possible downsampled grids

   :param Ur: Is the field to filter.
   :type Ur: Scalar object
   :param mesh: Is the mesh object used to compute the derivatives.
   :type mesh: Mesh object
   :param filter_size: Is the filter size used to filter the field.
   :type filter_size: int
   :param verbose: If True, it will output relevant information.
   :type verbose: bool

   :returns: **grad_y** -- The y component of the gradient
   :rtype: numpy array

