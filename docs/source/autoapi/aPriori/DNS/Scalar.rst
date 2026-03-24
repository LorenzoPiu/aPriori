aPriori.DNS.Scalar
==================

.. py:class:: aPriori.DNS.Scalar(shape, value=None, path='')

   A class representing a 3D scalar field.

   Attributes:
   -----------
   __scalar_dimension : int
       The dimension of the scalar field.
   test_phase : bool
       A flag indicating whether the test phase is active.
   VALID_DIMENSIONS : list
       A list of valid dimensions for the scalar field.

   Methods:
   --------
   __init__(shape, value=None, path=''):
       Initializes a Scalar object.
   value
       Getter and setter for the value attribute.
   shape
       Getter and setter for the shape attribute.
   path
       Getter and setter for the path attribute.
   Nx
       Getter and setter for the Nx attribute.
   Ny
       Getter and setter for the Ny attribute.
   Nz
       Getter and setter for the Nz attribute.
   file_name
       Getter and setter for the file_name attribute.
   file_id
       Getter and setter for the file_id attribute.
   filter_size
       Getter and setter for the filter_size attribute.
   _3d
       Getter for the 3D reshaped scalar field.
   is_light_mode():
       Checks if the scalar field is in light mode.

   reshape_3d():
       Reshapes the scalar field to 3D.

   reshape_column():
       Reshapes the scalar field to a column vector.

   reshape_line():
       Reshapes the scalar field to a row vector.

   cut(n_cut=1, mode='equal'):
       Cuts the scalar field.

   filter_gauss(delta, n_cut=0, mute=False):
       Filters the scalar field with a Gaussian function.

   plot_x_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
       Plots the x midplane of a 3D field.

   plot_y_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
       Plots the y midplane of a 3D field.

   plot_z_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)
       Plots the z midplane of a 3D field.


   .. py:attribute:: test_phase
      :value: True



   .. py:attribute:: VALID_DIMENSIONS
      :value: [3, 1]



   .. py:property:: shape


   .. py:property:: value


   .. py:property:: path


   .. py:property:: Nx


   .. py:property:: Ny


   .. py:property:: Nz


   .. py:property:: file_name


   .. py:property:: file_id


   .. py:property:: filter_size


   .. py:property:: x_midplane


   .. py:property:: y_midplane


   .. py:property:: z_midplane


   .. py:method:: is_light_mode()

      Checks if the scalar field is in light mode, that is when only the
      path is specified without specifying the value as an array.

      Returns:
      --------
      :
      bool
          True if the scalar field is in light mode, False otherwise.

      Raises:
      -------
      ValueError
          If neither the value nor the path is specified.




   .. py:method:: lenght()


   .. py:method:: reshape_3d()

      Reshapes the scalar field to 3D.

      Returns:
      --------
      :
      array-like
          The scalar field reshaped to 3D.




   .. py:method:: reshape_column()

      Reshapes the scalar field to a column vector.

      Returns:
      --------
      :
      array-like
          The scalar field reshaped to a column vector.




   .. py:method:: reshape_line()

      Reshapes the scalar field to a row vector.

      Returns:
      --------
      :
      array-like
          The scalar field reshaped to a row vector.




   .. py:method:: cut(n_cut=1, mode='equal')

              Cuts the scalar field.

              Parameters:
              -----------
              n_cut : int or tuple, optional
                  The number of samples to cut at the extrema. If mode is 'equal', n_cut is an integer
                  specifying the number of samples to cut from each side. If mode is 'xyz', n_cut is a
                  tuple specifying the number of samples to cut for each dimension. Default is 1.
              mode : {'equal', 'xyz'}, optional
                  The mode of cutting. 'equal' cuts the same number of samples from each side,
                  'xyz' allows specifying the number of samples to cut for each dimension. Default is 'equal'.

              Returns:
              --------
              array-like
                  The cut scalar field.

              Example:
              --------
              >>> # Example of cutting a scalar field
              >>> field = Scalar(shape=[10, 10, 10], value=np.random.rand(10, 10, 10))
              >>> # Cut the field with equal number of samples removed from each side
              >>> cut_field_equal = field.cut(n_cut=2, mode='equal')
              >>> print("Cut field with equal mode:")
              >>> print(cut_field_equal)
              >>> # Cut the field with specified number of samples removed for each dimension
              >>> cut_field_xyz = field.cut(n_cut=(1, 2, 3), mode='xyz')
              >>> print("
      Cut field with xyz mode:")
              >>> print(cut_field_xyz)




   .. py:method:: filter_gauss(delta, n_cut=0, mute=False)

      Filters the scalar field with a Gaussian function.
      The variance sigma is considered equal to:
          sigma = sqrt(1/12*delta**2)
      where delta is the filter size (in this case specified as
      the number of cells and not in meters)

      Parameters:
      -----------
      delta : float
          The amplitude of the Gaussian filter.
      n_cut : int, optional
          The number of samples to cut at the extrema. Default is 0.
      mute : bool, optional
          A flag indicating whether to mute the output. Default is False.

      Returns:
      --------
      :
      array-like
          The filtered scalar field.




   .. py:method:: plot_x_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)

      Plots the x midplane of a 3D field.

      Description:
      ------------
      This method plots the x midplane of a 3D field using the provided mesh. It uses the midpoint
      of the x-axis to generate a 2D plot of the field values at that plane.

      Parameters:
      -----------
      mesh : object
          The mesh object containing the coordinates.
      title : str, optional
          The title of the plot. Default is an empty string.
      colormap : str, optional
          The colormap to use for the plot. Default is 'viridis'.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_y_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)

      Plots the y midplane of a 3D field.

      Description:
      ------------
      This method plots the y midplane of a 3D field using the provided mesh. It uses the midpoint
      of the x-axis to generate a 2D plot of the field values at that plane.

      Parameters:
      -----------
      mesh : object
          The mesh object containing the coordinates.
      title : str, optional
          The title of the plot. Default is an empty string.
      colormap : str, optional
          The colormap to use for the plot. Default is 'viridis'.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None




   .. py:method:: plot_z_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)

      Plots the x midplane of a 3D field.

      Description:
      ------------
      This method plots the x midplane of a 3D field using the provided mesh. It uses the midpoint
      of the x-axis to generate a 2D plot of the field values at that plane.

      Parameters:
      -----------
      mesh : object
          The mesh object containing the coordinates.
      title : str, optional
          The title of the plot. Default is an empty string.
      colormap : str, optional
          The colormap to use for the plot. Default is 'viridis'.
      vmin : float, optional
          The minimum value for the color scale. Default is None.
      vmax : float, optional
          The maximum value for the color scale. Default is None.

      Returns:
      --------
      :
      None



