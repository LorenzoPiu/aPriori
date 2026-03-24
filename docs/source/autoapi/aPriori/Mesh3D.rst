aPriori.Mesh
==============

.. py:class:: aPriori.Mesh(X, Y, Z)

   A class used to represent a 3D mesh.

   This class takes three Scalar objects representing the X, Y, and Z coordinates of a 3D mesh.
   It checks that the input objects are instances of the Scalar class and have the same shape.
   The shape of the mesh is expected to be a list of three integers.
   The class also provides properties to access the unique values of the X, Y, and Z coordinates and their 3D representations.
   It also provides properties to access the X, Y, and Z coordinates at the midpoints along each axis.

   Attributes:
   -----------
   shape : list
       The shape of the 3D mesh.

   Nx : int
       The size of the mesh along the X axis.

   Ny : int
       The size of the mesh along the Y axis.

   Nz : int
       The size of the mesh along the Z axis.

   X : Scalar
       The X coordinates of the mesh.

   Y : Scalar
       The Y coordinates of the mesh.

   Z : Scalar
       The Z coordinates of the mesh.

   Methods:
   --------
   X1D:
       Returns the unique values of the X coordinates.

   Y1D:
       Returns the unique values of the Y coordinates.

   Z1D:
       Returns the unique values of the Z coordinates.

   X3D:
       Returns the 3D representation of the X coordinates.

   Y3D:
       Returns the 3D representation of the Y coordinates.

   Z3D:
       Returns the 3D representation of the Z coordinates.

   X_midY:
       Returns the X coordinates at the midpoint along the Y axis.

   X_midZ:
       Returns the X coordinates at the midpoint along the Z axis.

   Y_midX:
       Returns the Y coordinates at the midpoint along the X axis.

   Y_midZ:
       Returns the Y coordinates at the midpoint along the Z axis.

   Z_midX:
       Returns the Z coordinates at the midpoint along the X axis.

   Z_midY:
       Returns the Z coordinates at the midpoint along the Y axis.


   .. py:attribute:: VALID_DIMENSIONS
      :value: [3, 1]



   .. py:attribute:: shape


   .. py:attribute:: Nx


   .. py:attribute:: Ny


   .. py:attribute:: Nz


   .. py:attribute:: X


   .. py:attribute:: Y


   .. py:attribute:: Z


   .. py:attribute:: l


   .. py:property:: X1D


   .. py:property:: Y1D


   .. py:property:: Z1D


   .. py:property:: X3D


   .. py:property:: Y3D


   .. py:property:: Z3D


   .. py:property:: X_midY


   .. py:property:: X_midZ


   .. py:property:: Y_midX


   .. py:property:: Y_midZ


   .. py:property:: Z_midX


   .. py:property:: Z_midY

