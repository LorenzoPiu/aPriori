aPriori.derivatives.laplacian
=============================

.. py:function:: aPriori.derivatives.laplacian(F, mesh, filter_size=1, acc=4, reduce_acc=False)

   Description
   -----------
   Computes the Laplacian (d2/dx2 + d2/dy2 + d2/dz2) of the field.

   :param F:
   :type F: Scalar object
   :param mesh:
   :type mesh: Mesh object
   :param filter_size:
   :type filter_size: int, optional
   :param acc:
   :type acc: int, optional

   :returns: **lap** -- The Laplacian of the input field.
   :rtype: numpy array

