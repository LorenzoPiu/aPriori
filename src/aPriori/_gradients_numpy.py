# def gradient_x(F, mesh, filter_size=1):
#     '''
#         Description
#         -----------
        
#         Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
#         used to compute the gradients on all the possible downsampled grids.
        
#         Specifically, the parameter filter_size is used to temporarily downsample
#         the grid in the x direction. The function considers one point each 
#         filter_size points and computes the derivatives on this downsampled grid.
#         Does this for every possible downsampled grid, so in the end the output
#         field has the same shape as the input field.

#         Parameters
#         ----------
#         Ur : Scalar object
#             Is the field to filter.
        
#         mesh : Mesh object
#             Is the mesh object used to compute the derivatives.
            
#         filter_size : int
#             Is the filter size used to filter the field.
        
#         verbose : bool
#             If True, it will output relevant information.

#         Returns
#         -------
#         grad_x : numpy array
#             The x component of the gradient 
#         '''
#     import time
#     # Check the data types of the input
#     if not isinstance(mesh, Mesh):
#         raise TypeError("mesh must be an object of the class Mesh")
#     if not isinstance(F, Scalar):
#         raise TypeError("F must be an object of the class Scalar")
#     if not isinstance(filter_size, int):
#         raise TypeError("filter_size must be an integer")
#     Nx = F.Nx
#     Ny = F.Ny
#     Nz = F.Nz
    
#     grad_x = np.zeros(F._3d.shape)
#     X1D = mesh.X1D
#     for i in range(filter_size):
#         start = i
        
#         field = F._3d[start::filter_size, :, :]
        
#         grad_x[start::filter_size, :, :] = np.gradient(field, X1D[start::filter_size], axis=0)
        
#     return grad_x

# def gradient_y(F, mesh, filter_size=1):
#     '''
#         Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
#         used to computed the gradients on all the possible downsampled grids

#         Parameters
#         ----------
#         Ur : Scalar object
#             Is the field to filter.
        
#         mesh : Mesh object
#             Is the mesh object used to compute the derivatives.
            
#         filter_size : int
#             Is the filter size used to filter the field.
        
#         verbose : bool
#             If True, it will output relevant information.

#         Returns
#         -------
#         grad_y : numpy array
#             The y component of the gradient            
#         '''
#     import time
#     # Check the data types of the input
#     if not isinstance(mesh, Mesh):
#         raise TypeError("mesh must be an object of the class Mesh")
#     if not isinstance(F, Scalar):
#         raise TypeError("F must be an object of the class Scalar")
#     if not isinstance(filter_size, int):
#         raise TypeError("filter_size must be an integer")
#     Nx = F.Nx
#     Ny = F.Ny
#     Nz = F.Nz
    
#     grad_y = np.zeros(F._3d.shape)
#     Y1D = mesh.Y1D
#     for i in range(filter_size):
#         start = i
        
#         field = F._3d[:, start::filter_size, :]
        
#         grad_temp = np.gradient(field, Y1D[start::filter_size], axis=1)
        
#         grad_y[:, start::filter_size, :] = grad_temp
        
#     return grad_y

# def gradient_z(F, mesh, filter_size=1):
#     '''
#         Computes the z component of the gradient of a 3D, non downsampled, filtered field. 
#         Numpy is used to computed the gradients on all the possible downsampled grids

#         Parameters
#         ----------
#         Ur : Scalar object
#             Is the field to filter.
        
#         mesh : Mesh object
#             Is the mesh object used to compute the derivatives.
            
#         filter_size : int
#             Is the filter size used to filter the field.
        
#         Returns
#         -------
#         grad_z : numpy array
#             The z component of the gradient            
#         '''
#     import time
#     # Check the data types of the input
#     if not isinstance(mesh, Mesh):
#         raise TypeError("mesh must be an object of the class Mesh")
#     if not isinstance(F, Scalar):
#         raise TypeError("F must be an object of the class Scalar")
#     if not isinstance(filter_size, int):
#         raise TypeError("filter_size must be an integer")
#     Nx = F.Nx
#     Ny = F.Ny
#     Nz = F.Nz
    
#     grad_z = np.zeros(F._3d.shape)
#     Z1D = mesh.Z1D
#     for i in range(filter_size):
#         start = i
        
#         field = F._3d[:, :, start::filter_size]
        
#         grad_temp = np.gradient(field, Z1D[start::filter_size], axis=2)
        
#         grad_z[:, :, start::filter_size] = grad_temp
        
#     return grad_z