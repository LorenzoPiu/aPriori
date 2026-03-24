aPriori.DNS.gradient_z
======================

.. py:function:: aPriori.DNS.gradient_z(F, mesh, filter_size=1)

   Computes the z component of the gradient of a 3D, non downsampled, filtered field.
   Numpy is used to computed the gradients on all the possible downsampled grids

   :param Ur: Is the field to filter.
   :type Ur: Scalar object
   :param mesh: Is the mesh object used to compute the derivatives.
   :type mesh: Mesh object
   :param filter_size: Is the filter size used to filter the field.
   :type filter_size: int

   :returns: **grad_z** -- The z component of the gradient
   :rtype: numpy array

