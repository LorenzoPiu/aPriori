aPriori.derivatives.gradient_x
==============================

.. py:function:: aPriori.derivatives.gradient_x(F, mesh, filter_size=1, acc=8, reduce_acc=False)

   Description
   -----------
   Computes the x-component of the gradient using findiff with specified accuracy.
   Uses striding (filter_size) to compute derivatives on downsampled grids.

   :param F: The field to differentiate.
   :type F: Scalar object
   :param mesh: The mesh object containing coordinates (X1D).
   :type mesh: Mesh object
   :param filter_size: The stride used to filter the field (default is 1).
   :type filter_size: int, optional
   :param acc: The accuracy order for the finite difference scheme (default is 2).
   :type acc: int, optional

   :returns: **grad_x** -- The x component of the gradient.
   :rtype: numpy array

