#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:17:23 2024

@author: Lorenzo Piu
"""

###########################################################
#                       Field3d
###########################################################
class Field3D:
    # I want to try to use a getter for the fields value that every time that you
    # call the field (e.g. if you write DNS_FIELD.T) loads the value in the variable,
    # removing it when not necessary.
    # or also I could build a loader function, es DNS_field.load("T")
    # and a release function, e.g. 
    
    _field_dimension = 3
    test_phase = True  # I only need it to be True when I debug the code
    variables_names = {
                        "dUX_dX"        : "dUX_dX_s-1",
                        "dUX_dY"        : "dUX_dY_s-1",
                        "dUX_dZ"        : "dUX_dZ_s-1",
                        "dUY_dX"        : "dUY_dX_s-1",
                        "dUY_dY"        : "dUY_dY_s-1",
                        "dUY_dZ"        : "dUY_dZ_s-1",
                        "dUZ_dX"        : "dUZ_dX_s-1",
                        "dUZ_dY"        : "dUZ_dY_s-1",
                        "dUZ_dZ"        : "dUZ_dZ_s-1",
                        "strain rate"   : "S_s-1"
                        
                       }
    
    def __init__(self, shape, n_cut=0):
        self._shape = shape
    
    @property
    def T(self):
        if Field3D.test_phase is True:
            print("Getting Temperature")
        return self._temperature
    @T.setter
    def T(self, value):
        if Field3D.test_phase is True:
            print("Setting Temperature field...")
        if value.any() < 0:
            raise ValueError("The temperature should assume positive values")
        self._T = value
        
    @property
    def RHO(self):
        return self._RHO
    @T.setter
    def RHO(self, value):
        if value.any() < 0:
            raise ValueError("The density should assume positive values")
        self._RHO = Scalar3D(self.shape, value)
        
    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, shape):
        # Check that the shape has the correct format
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Scalar3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        # assign the values
        self._Nx = shape[0]
        self._Ny = shape[1]
        self._Nz = shape[2]
        self._shape = (self.Nx, self.Ny, self.Nz)
        
    @property
    def Nx(self):
        return self._Nx
    @Nx.setter
    def Nx(self, Nx):
        self._Nx = Nx
        
    @property
    def Ny(self):
        return self._Ny
    @Ny.setter
    def Ny(self, Ny):
        self._Ny = Ny
    
    @property
    def Nz(self):
        return self._Nz
    @Nz.setter
    def Nz(self, Nz):
        self._Nz = Nz
    
    
###############################################################################
#                                Scalar3D
###############################################################################
class Scalar3D:
    
    __scalar_dimension = 3
    test_phase = True  # I only need it to be True when I debug the code
    
    VALID_DIMENSIONS = [3, 1] # I am still not using it but I want to insert a check that allows to input the scalar field also as a 1D vector
    
    def __init__(self, shape, value=None, path=''):
        import numpy as np
        # check that the shape of the field is a list of 3 integers
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Scalar3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        # setting the shape, Nx, Ny, Nz
        self.shape = shape
        
        # assign the value to the field and reshape it if it was initialized
        self._value = value
        if value is not None and np.ndim(value)!=3:
            self.reshape_3d()
        
        #assign the path
        self.path = path
    
    # The value attribute contains the array with the values of the field.
    # By default it is reshaped in a 3d array
    @property
    def value(self):
        # if Scalar3D.test_phase is True:
        #     print("Getting scalar field...")
        if self._value is not None:
            # print("value assigned to the variable. returning the field in memory")
            return self._value
        else:
            if self.path != '':
                # print("Value not assigned, reading the file")
                return process_file(self.path)
            else:
                raise ValueError("To call the value of a Scalar3D object, you must specify either the value or the file path")    
    @value.setter
    def value(self, value):
        import numpy as np
        # assign value
        self._value = value
        # reshape to 3d field by default
        if np.ndim(value)!=3:
            self.reshape_3d()
    
    # shape getter and setter. The setter also set Nx, Ny, Nz. These variables
    # can be considered redundant but help coding. only set them using the shape setter.
    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, shape):
        # Check that the shape has the correct format
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Scalar3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        # assign the values
        self._Nx = shape[0]
        self._Ny = shape[1]
        self._Nz = shape[2]
        self._shape = (self.Nx, self.Ny, self.Nz)
        
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, path):
        import re
        if isinstance(path, str):
            self._path = path
            self._file_name = path.split('/')[-1]
            pattern = r'id\d+'
            match = re.search(pattern, self._file_name, re.IGNORECASE)
            if match:
                self._file_id = match.group()
            
            folder_name = path.split('/')[0]
            pattern = r'filter(\d+)'
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                self._filter_size = int(match.group(1))
            else:
                self._filter_size = None

        else:
            TypeError("The file path must be a string")
        
    @property
    def Nx(self):
        return self._Nx
    @Nx.setter
    def Nx(self, Nx):
        self._Nx = Nx
        
    @property
    def Ny(self):
        return self._Ny
    @Ny.setter
    def Ny(self, Ny):
        self._Ny = Ny
    
    @property
    def Nz(self):
        return self._Nz
    @Nz.setter
    def Nz(self, Nz):
        self._Nz = Nz
    
    @property
    def file_name(self):
        return self._file_name
    @file_name.setter
    def file_name(self, file_name):
        if isinstance(file_name, str):
            self._file_name = file_name
        else:
            TypeError("The file name must be a string")
            
    @property
    def file_id(self):
        return self._file_id
    @file_id.setter
    def file_id(self, file_id):
        if isinstance(file_id, str):
            self._file_id = file_id
        else:
            TypeError("The file name must be a string")
            
    @property
    def filter_size(self):
        return self._filter_size
    @filter_size.setter
    def filter_size(self, filter_size):
        if isinstance(filter_size, int):
            self._filter_size = filter_size
        else:
            TypeError("The filter size must be an integer")
    
    @property
    def _3d(self):
        import numpy as np
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(self.Nx, self.Ny, self.Nz)        

    
    # Class Methods
    def is_light_mode(self):
        if self._value is not None:
            return False
        else:
            if self.path != '':
                return True
            else:
                raise ValueError("To call the value of a Scalar3D object, you must specify either the value or the file path")    
    
    def reshape_3d(self):
        import numpy as np
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(self.Nx, self.Ny, self.Nz)
        
    def reshape_column(self):
        import numpy as np
        new_shape = [self.Nx*self.Ny*self.Nz, 1]
        if self.is_light_mode():
            return np.reshape(self.value, self.shape)
        else:
            return self.value.reshape(new_shape)
        
    def reshape_line(self):
        self.value.reshape(1, self.Nx*self.Ny*self.Nz)
        
    def cut(self, n_cut=1, mode='equal'):
        if not self.is_light_mode():
            # TODO: update this function to handle the xyz mode also in this branch
            self.reshape_3d()
            if mode=='equal':
                self.value = self.value[n_cut:self.Nx-n_cut,n_cut:self.Ny-n_cut,n_cut:self.Nz-n_cut]
            self.shape = self.value.shape #update the shape of the field
        else:
            field_cut = self.reshape_3d()
            if mode=='equal':
                field_cut = field_cut[n_cut:self.Nx-n_cut,n_cut:self.Ny-n_cut,n_cut:self.Nz-n_cut]
            elif mode=='xyz':
                n_cut_x = n_cut[0]
                n_cut_y = n_cut[1]
                n_cut_z = n_cut[2]
                field_cut = field_cut[n_cut_x:self.Nx-n_cut_x,n_cut_y:self.Ny-n_cut_y,n_cut_z:self.Nz-n_cut_z]
                return field_cut
        
    
    def filterGaussDS(self, delta,n_cut=0,mute=False):
        # field must be a 3d array, with the same size as the grid points in X,Y,Z.
        # delta is the amplitude of the gaussian filter
        # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.signal import decimate
        
        # filter the field with a constant amplitude gaussian function. notice that in this case we consider that the points are equispaced. It should not be a problem also for non equispaced points.    
        fieldFilt = gaussian_filter(self.value, sigma=1/12*delta**2, mode='constant')

        # Downsample the filtered field
        fieldDS = downsampleAndCut(fieldFilt, delta, n_cut, mute)
        
        return fieldDS
    
    def filter_gauss(self, delta,n_cut=0,mute=False):
        # field must be a 3d array, with the same size as the grid points in X,Y,Z.
        # delta is the amplitude of the gaussian filter
        # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.signal import decimate
        
        # filter the field with a constant amplitude gaussian function. notice that in this case we consider that the points are equispaced. It should not be a problem also for non equispaced points.    
        field_filt = gaussian_filter(self.value, sigma=1/12*delta**2, mode='constant')

        return field_filt
    
    def filterBoxDS(self,delta,n_cut=0):
        # field must be a 3d array, with the same size as the grid points in X,Y,Z.
        # delta is the amplitude of the box to consider for the variance
        # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
        
        import numpy as np
        
        # Get field dimensions
        Nx, Ny, Nz = self.value.shape

        # define the cells to extract. they will be th center the boxes
        i_idx = np.arange(0, Nx-1, delta)
        j_idx = np.arange(0, Ny-1, delta)
        k_idx = np.arange(0, Nz-1, delta)
        
        deltaPos = delta//2
        # take into account that if the filter size is pari the box will not be centered in the node
        if delta%2==0:
            deltaNeg = delta//2 -1
        else:
            deltaNeg = delta//2
        
        average = np.zeros([len(i_idx), len(j_idx), len(k_idx)])
        
        counter_i = 0
        for i in i_idx:
            counter_j = 0
            for j in j_idx:
                counter_k = 0
                for k in j_idx:
                    # build the indexes to define the box
                    x_idx_box = range(i-deltaNeg, i+deltaPos)
                    x_idx_box = [x for x in x_idx_box if (x >= 0 and x<Nx)]
                    y_idx_box = range(j-deltaNeg, j+deltaPos)
                    y_idx_box = [x for x in y_idx_box if (x >= 0 and x<Ny)]
                    z_idx_box = range(k-deltaNeg, k+deltaPos)
                    z_idx_box = [x for x in z_idx_box if (x >= 0 and x<Nz)]
                    
                    idx_box = np.ix_(x_idx_box, y_idx_box, z_idx_box)
                    
                    box = self.value[idx_box]
                    
                    box_ = box.flatten()
                    
                    average[counter_i, counter_j, counter_k] = np.average(box_)
                    
                    counter_k = counter_k+1
                counter_j = counter_j+1
            counter_i = counter_i+1
        
        # cut estrema
        averageCut = average[n_cut:average.shape[0]-n_cut,n_cut:average.shape[1]-n_cut,n_cut:average.shape[2]-n_cut]
        
        return averageCut
    
    def varianceDS(self,delta,n_cut=0):
        # field must be a 3d array, with the same size as the grid points in X,Y,Z.
        # delta is the amplitude of the box to consider for the variance
        # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
        
        import numpy as np
        
        # Get field dimensions
        Nx, Ny, Nz = self.value.shape

        # define the cells to extract. they will be th center the boxes
        i_idx = np.arange(0, Nx-1, delta)
        j_idx = np.arange(0, Ny-1, delta)
        k_idx = np.arange(0, Nz-1, delta)
        
        deltaNeg = delta//2
        # take into account that if the filter size is pari the box will not be centered in the node
        if delta%2==0:
            deltaPos = delta//2 + 1
        else:
            deltaPos = delta//2
        
        variance = np.zeros([len(i_idx), len(j_idx), len(k_idx)])
            
        counter_i = 0
        for i in i_idx:
            counter_j = 0
            for j in j_idx:
                counter_k = 0
                for k in j_idx:
                    # build the indexes to define the box
                    x_idx_box = range(i-deltaNeg, i+deltaPos)
                    x_idx_box = [x for x in x_idx_box if (x >= 0 and x<Nx)]
                    y_idx_box = range(j-deltaNeg, j+deltaPos)
                    y_idx_box = [x for x in y_idx_box if (x >= 0 and x<Ny)]
                    z_idx_box = range(k-deltaNeg, k+deltaPos)
                    z_idx_box = [x for x in z_idx_box if (x >= 0 and x<Nz)]
                    
                    idx_box = np.ix_(x_idx_box, y_idx_box, z_idx_box)
                    
                    box = self.value[idx_box]
                    
                    box_ = box.flatten()
                    
                    variance[counter_i, counter_j, counter_k] = np.sum((box_-np.average(box_))**2)/len(box_)
                    
                    counter_k = counter_k+1
                counter_j = counter_j+1
            counter_i = counter_i+1
        
        # cut estrema
        varianceCut = variance[n_cut:variance.shape[0]-n_cut,n_cut:variance.shape[1]-n_cut,n_cut:variance.shape[2]-n_cut]

        return varianceCut
    
    def gradient_module(self, mesh):
        import numpy as np
        F_x,F_y,F_z = np.gradient(self.reshape_3d(), mesh.X1D, mesh.Y1D, mesh.Z1D)
        grad_C = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        return grad_C
    
    def gradient(self,mesh):
        import numpy as np
        F_x,F_y,F_z = np.gradient(self.reshape_3d(), mesh.X1D, mesh.Y1D, mesh.Z1D)
        return F_x, F_y, F_z
    
    def gradient_x(self,mesh):
        import numpy as np
        F_x,_,_ = np.gradient(self.reshape_3d(), mesh.X1D, mesh.Y1D, mesh.Z1D)
        return F_x
    
    def gradient_y(self,mesh):
        import numpy as np
        _,F_y,_ = np.gradient(self.reshape_3d(), mesh.X1D, mesh.Y1D, mesh.Z1D)
        return F_y
    
    def gradient_z(self,mesh):
        import numpy as np
        _,_,F_z = np.gradient(self.reshape_3d(), mesh.X1D, mesh.Y1D, mesh.Z1D)
        return F_z

    def plot_x_midplane(self, mesh, title='', colormap='viridis'):
        import matplotlib.pyplot as plt
        import numpy as np

        Y,Z = 1e3*mesh.Y3D, 1e3*mesh.Z3D
        f = self.reshape_3d()
        
        # Calculate the midplane index
        x_mid = Y.shape[0] // 2

        # Plot the x midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(Y[x_mid, :, :], Z[x_mid, :, :], f[x_mid, :, :], shading='auto', cmap = colormap)
        ax.set_xlabel('y (mm)', fontsize=18)
        ax.set_ylabel('z (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()
        
    def plot_y_midplane(self, mesh, title='', colormap='viridis'):
        import matplotlib.pyplot as plt
        import numpy as np

        if not isinstance(mesh, Mesh3D):
            raise ValueError("mesh must be an object of the class Mesh3D")
        
        y_mid = mesh.shape[2]//2

        # Plot the y midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(1e3*mesh.X_midY, 1e3*mesh.Z_midY, self._3d[:, y_mid, :], shading='auto', cmap = colormap)
        ax.set_xlabel('x (mm)', fontsize=18)
        ax.set_ylabel('z (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()        
        
    def plot_z_midplane(self, mesh, title='', colormap='viridis'):
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not isinstance(mesh, Mesh3D):
            raise ValueError("mesh must be an object of the class Mesh3D")
        
        z_mid = mesh.shape[2]//2
        
        # Plot the z midplane
        fig, ax = plt.subplots()
        im = ax.pcolormesh(1e3*mesh.X_midZ, 1e3*mesh.Y_midZ, self._3d[:, :, z_mid], shading='auto', cmap = colormap)
        ax.set_xlabel('x (mm)', fontsize=18)
        ax.set_ylabel('y (mm)', fontsize=18)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(labelsize=16)
        plt.title(title, fontsize=22)
        plt.show()
    
####################################################################
#                       Mesh3D
# class Mesh3D:
#     __scalar_dimension = 3

#     VALID_DIMENSIONS = [3, 1]

#     def __init__(self, X, Y, Z, shape):
#         import numpy as np

#         valid_shape = all(isinstance(item, int) for item in shape) and len(shape) == Mesh3D.__scalar_dimension
#         if not valid_shape:
#             raise ValueError("The shape of the 3d field must be a list of 3 integers")

#         self.shape = shape
#         self.Nx, self.Ny, self.Nz = shape

#         if np.ndim(X) == Mesh3D.__scalar_dimension:
#             if X.shape != shape:
#                 raise ValueError("The shape of the mesh must be the same as the input shape variable")
#             X, Y, Z = np.unique(X), np.unique(Y), np.unique(Z)
#         else:
#             if len(X) != np.prod(shape):
#                 X = np.unique(X.reshape(shape))
#                 Y = np.unique(Y.reshape(shape))
#                 Z = np.unique(Z.reshape(shape))
#             else:
#                 X, Y, Z = np.unique(X), np.unique(Y), np.unique(Z)

#         self.X, self.Y, self.Z = X, Y, Z
#         self.X3D, self.Y3D, self.Z3D = None, None, None

#     @property
#     def X3D(self):
#         import numpy as np
#         x3D, _, _ = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return x3D

#     @property
#     def Y3D(self):
#         import numpy as np
#         _, y3D, _ = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return y3D

#     @property
#     def Z3D(self):
#         import numpy as np
#         _, _, z3D = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return z3D


# class Mesh3D:
# old class for the mesh

#     __scalar_dimension = 3

#     VALID_DIMENSIONS = [3, 1] # I am still not using it but I want to insert a check that allows to input the scalar field also as a 1D vector
      
#     def __init__(self, X, Y, Z, shape):
#         import numpy as np
#         # check that the shape of the field is a list of 3 integers
#         valid_shape =  False
#         if isinstance(shape, list) and len(shape)==Mesh3D.__scalar_dimension:
#             for item in shape:
#                 if not isinstance(item, int):
#                     valid_shape =  True
#         if valid_shape is False:
#             ValueError("The shape of the 3d field must be a list of 3 integers")
#         # setting the shape, Nx, Ny, Nz
#         self.shape = shape
#         self.Nx = shape[0]
#         self.Ny = shape[1]
#         self.Nz = shape[2]
        
#         # store the 1d vectors of the mesh because it does not occupy a lot of memory
#         # check that X, Y and Z are Scalar3d objects
#         is1d = False
#         is3d = False
#         isunique = False # True if the vectors are already in the correct form
#         if np.ndim(X) == Mesh3D.__scalar_dimension:
#             is3d = True
#             if X.shape != shape:
#                 ValueError("The shape of the mesh must be the same as the input shape variable")
#             else :
#                 X = np.unique(X)
#                 Y = np.unique(Y)
#                 Z = np.unique(Z)
#         else:
#             is1d = True
#             if len(X) != shape[0]*shape[1]*shape[2]:
#                 isunique = True
#             else:
#                 X = np.unique(X.reshape(shape[0],shape[1],shape[2]))
#                 Y = np.unique(Y.reshape(shape[0],shape[1],shape[2]))
#                 Z = np.unique(Z.reshape(shape[0],shape[1],shape[2]))
                
#         self.X = X
#         self.Y = Y
#         self.Z = Z
#         self.X3D = None
#         self.Y3D = None
#         self.Z3D = None

#     # The value attribute contains the array with the values of the field.
#     # By default it is reshaped in a 3d array
    
#     @property
#     def X3D(self):
#         return self.get_X3D()
#     @X3D.setter
#     def X3D(self, value):
#         self._X3D = value
    
#     @property
#     def Y3D(self):
#         return self.get_Y3D()
#     @Y3D.setter
#     def Y3D(self, value):
#         self._Y3D = value
    
#     @property
#     def Z3D(self):
#         return self.get_Z3D()
#     @Z3D.setter
#     def Z3D(self, value):
#         self._Z3D = value
        
#     def get_X3D(self):
#         import numpy as np
#         x3D, _, _ = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return x3D
    
#     def get_Y3D(self):
#         import numpy as np
#         _, y3D, _ = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return y3D
    
#     def get_Z3D(self):
#         import numpy as np
#         _, _, z3D = np.meshgrid(self.X, self.Y, self.Z, indexing='ij')
#         return z3D
    
    
    
class Mesh3D:


    __scalar_dimension = 3

    VALID_DIMENSIONS = [3, 1] # I am still not using it but I want to insert a check that allows to input the scalar field also as a 1D vector
    
    def __init__(self, X, Y, Z, shape):
        import numpy as np
        
        # check that X, Y and Z are Scalar3D objects
        if not isinstance(X, Scalar3D):
            raise ValueError("X must be an object of the class Scalar3D")
        if not isinstance(Y, Scalar3D):
            raise ValueError("X must be an object of the class Scalar3D")
        if not isinstance(Z, Scalar3D):
            raise ValueError("Z must be an object of the class Scalar3D")
        
        # check that the shape of the field is a list of 3 integers
        valid_shape =  False
        if isinstance(shape, list) and len(shape)==Mesh3D.__scalar_dimension:
            for item in shape:
                if not isinstance(item, int):
                    valid_shape =  True
        if valid_shape is False:
            ValueError("The shape of the 3d field must be a list of 3 integers")
        
        # setting the shape, Nx, Ny, Nz
        self.shape = shape
        self.Nx = shape[0]
        self.Ny = shape[1]
        self.Nz = shape[2]
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        x_mid = self.shape[0]//2
        y_mid = self.shape[1]//2
        z_mid = self.shape[2]//2
        
        self._X_midZ = X._3d[:, :, z_mid]
        self._Y_midZ = Y._3d[:, :, z_mid]
        
        self._X_midY = X._3d[:, y_mid, :]
        self._Z_midY = Z._3d[:, y_mid, :]
        
        self._Y_midX = Y._3d[x_mid, :, :]
        self._Z_midX = Y._3d[x_mid, :, :]

    # The value attribute contains the array with the values of the field.
    # By default it is reshaped in a 3d array
    
    @property
    def X1D(self):
        self._X1D = np.unique(self.X.value)
        return self._X1D

    @property
    def Y1D(self):
        self._Y1D = np.unique(self.Y.value)
        return self._Y1D    

    @property
    def Z1D(self):
        self._Z1D = np.unique(self.Z.value)
        return self._Z1D    
    
    @property
    def X3D(self):
        return self.X._3d

    @property
    def Y3D(self):
        return self.Y._3d    

    @property
    def Z3D(self):
        return self.Z._3d
    
    @property
    def X_midY(self):
        return self._X_midY
    
    @property
    def X_midZ(self):
        return self._X_midZ
    
    @property
    def Y_midX(self):
        return self._Y_midX
    
    @property
    def Y_midZ(self):
        return self._Y_midZ
    
    @property
    def Z_midX(self):
        return self._Z_midX
    
    @property
    def Z_midY(self):
        return self._Z_midY
    
    
###############################################################################
#                               Functions
###############################################################################
def process_file(file_path):
    import numpy as np
    # Placeholder for file processing logic
    return np.fromfile(file_path,dtype='<f4')

def filterGaussDS(field,delta,n_cut=0,mute=False):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amplitude of the gaussian filter
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import decimate
    
    # filter the field with a constant amplitude gaussian function. notice that in this case we consider that the points are equispaced. It should not be a problem also for non equispaced points.    
    fieldFilt = gaussian_filter(field, sigma=1/12*delta**2, mode='constant')

    # Downsample the filtered field
    fieldDS = downsampleAndCut(fieldFilt, delta, n_cut, mute)
    
    return fieldDS

def filter_gauss(field,delta,n_cut=0,mute=False):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amplitude of the gaussian filter
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import decimate
    
    # filter the field with a constant amplitude gaussian function. notice that in this case we consider that the points are equispaced. It should not be a problem also for non equispaced points.    
    fieldFilt = gaussian_filter(field, sigma=1/12*delta**2, mode='constant')

    return fieldFilt

def varianceDS(field,delta,n_cut=0):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amplitude of the box to consider for the variance
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
    
    import numpy as np
    
    # Get field dimensions
    Nx, Ny, Nz = field.shape

    # define the cells to extract. they will be th center the boxes
    i_idx = np.arange(0, Nx-1, delta)
    j_idx = np.arange(0, Ny-1, delta)
    k_idx = np.arange(0, Nz-1, delta)
    
    deltaNeg = delta//2
    # take into account that if the filter size is pari the box will not be centered in the node
    if delta%2==0:
        deltaPos = delta//2 + 1
    else:
        deltaPos = delta//2
    
    variance = np.zeros([len(i_idx), len(j_idx), len(k_idx)])
        
    counter_i = 0
    for i in i_idx:
        counter_j = 0
        for j in j_idx:
            counter_k = 0
            for k in j_idx:
                # build the indexes to define the box
                x_idx_box = range(i-deltaNeg, i+deltaPos)
                x_idx_box = [x for x in x_idx_box if (x >= 0 and x<Nx)]
                y_idx_box = range(j-deltaNeg, j+deltaPos)
                y_idx_box = [x for x in y_idx_box if (x >= 0 and x<Ny)]
                z_idx_box = range(k-deltaNeg, k+deltaPos)
                z_idx_box = [x for x in z_idx_box if (x >= 0 and x<Nz)]
                
                idx_box = np.ix_(x_idx_box, y_idx_box, z_idx_box)
                
                box = field[idx_box]
                
                box_ = box.flatten()
                
                variance[counter_i, counter_j, counter_k] = np.sum((box_-np.average(box_))**2)/len(box_)
                
                counter_k = counter_k+1
            counter_j = counter_j+1
        counter_i = counter_i+1
    
    # cut estrema
    varianceCut = variance[n_cut:variance.shape[0]-n_cut,n_cut:variance.shape[1]-n_cut,n_cut:variance.shape[2]-n_cut]
    
    return varianceCut

def filterBoxDS(field,delta,n_cut=0):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amplitude of the box to consider for the variance
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
    
    import numpy as np
    
    # Get field dimensions
    Nx, Ny, Nz = field.shape

    # define the cells to extract. they will be th center the boxes
    i_idx = np.arange(0, Nx-1, delta)
    j_idx = np.arange(0, Ny-1, delta)
    k_idx = np.arange(0, Nz-1, delta)
    
    deltaPos = delta//2
    # take into account that if the filter size is pari the box will not be centered in the node
    if delta%2==0:
        deltaNeg = delta//2 -1
    else:
        deltaNeg = delta//2
    
    average = np.zeros([len(i_idx), len(j_idx), len(k_idx)])
        
    counter_i = 0
    for i in i_idx:
        counter_j = 0
        for j in j_idx:
            counter_k = 0
            for k in j_idx:
                # build the indexes to define the box
                x_idx_box = range(i-deltaNeg, i+deltaPos)
                x_idx_box = [x for x in x_idx_box if (x >= 0 and x<Nx)]
                y_idx_box = range(j-deltaNeg, j+deltaPos)
                y_idx_box = [x for x in y_idx_box if (x >= 0 and x<Ny)]
                z_idx_box = range(k-deltaNeg, k+deltaPos)
                z_idx_box = [x for x in z_idx_box if (x >= 0 and x<Nz)]
                
                idx_box = np.ix_(x_idx_box, y_idx_box, z_idx_box)
                
                box = field[idx_box]
                
                box_ = box.flatten()
                
                average[counter_i, counter_j, counter_k] = np.average(box_)
                
                counter_k = counter_k+1
            counter_j = counter_j+1
        counter_i = counter_i+1
    
    # cut estrema
    averageCut = average[n_cut:average.shape[0]-n_cut,n_cut:average.shape[1]-n_cut,n_cut:average.shape[2]-n_cut]
    
    return averageCut

def myFilterGaussDS(field,xDS,yDS,zDS,delta,n_cut=0):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amplitude of the filter (in DNS cells) 
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
    
    import numpy as np
    
    # Get field dimensions
    Nx, Ny, Nz = field.shape

    # define the cells to extract. they will be th center the boxes
    i_idx = np.arange(0, Nx-1, delta)
    j_idx = np.arange(0, Ny-1, delta)
    k_idx = np.arange(0, Nz-1, delta)
    
    deltaNeg = delta//2
    # take into account that if the filter size is pari the box will not be centered in the node
    if delta%2==0:
        deltaPos = delta//2 + 1
    else:
        deltaPos = delta//2
    
    average = np.zeros([len(i_idx), len(j_idx), len(k_idx)])
    
    counter_i = 0
    for i in i_idx:
        counter_j = 0
        for j in j_idx:
            counter_k = 0
            for k in j_idx:
                # build the indexes to define the box
                x_idx_box = range(i-deltaNeg, i+deltaPos)
                x_idx_box = [x for x in x_idx_box if (x >= 0 and x<Nx)]
                y_idx_box = range(j-deltaNeg, j+deltaPos)
                y_idx_box = [x for x in y_idx_box if (x >= 0 and x<Ny)]
                z_idx_box = range(k-deltaNeg, k+deltaPos)
                z_idx_box = [x for x in z_idx_box if (x >= 0 and x<Nz)]
                
                idx_box = np.ix_(x_idx_box, y_idx_box, z_idx_box)
                
                box = field[idx_box]
                
                box_ = box.flatten()
                
                average[counter_i, counter_j, counter_k] = np.average(box_)
                
                counter_k = counter_k+1
            counter_j = counter_j+1
        counter_i = counter_i+1
    
    # cut estrema
    averageCut = average[n_cut:average.shape[0]-n_cut,n_cut:average.shape[1]-n_cut,n_cut:average.shape[2]-n_cut]
    
    return averageCut

def downsampleAndCut(field,delta,n_cut=0, mute=False):
    # field must be a 3d array, with the same size as the grid points in X,Y,Z.
    # delta is the amount of points to consider (delta = 3 means we keep only 1 point every 3)
    # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
    

    # Downsample the filtered field
    fieldDS = field[::delta, ::delta, ::delta]
    
    # cut the extrema because the filtering operation creates nonsense values at the Boundaries:
    # n_cut = #amount of data to cut from both sides, both in the x, y and z direction
    fieldDS = fieldDS[n_cut:fieldDS.shape[0]-n_cut,n_cut:fieldDS.shape[1]-n_cut,n_cut:fieldDS.shape[2]-n_cut]

    if mute == False:
    # Print the original and filtered field shapes
        print("\nOriginal field shape:", field.shape)
        print("Downsampled field shape:", fieldDS.shape)
    
    return fieldDS

def plot_midplanes(X, Y, Z, f, min_f=None, max_f=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if max_f==None:
        max_f = np.max(f)
    if min_f==None:
        min_f = np.min(f)
        
    # Calculate the midplane indices
    x_mid = X.shape[0] // 2
    y_mid = Y.shape[1] // 2
    z_mid = Z.shape[2] // 2

    X,Y,Z = 1e3*X, 1e3*Y, 1e3*Z
    # Create a 3-figure subplot
    fig, axs = plt.subplots(3, 1, figsize=(15, 7), gridspec_kw={'height_ratios': [1, 1, 1]})

    # Plot the x midplane
    im1=axs[0].pcolormesh(Y[x_mid, :, :], Z[x_mid, :, :], f[x_mid, :, :], shading='auto', vmin=min_f, vmax=max_f)
    cbar1=fig.colorbar(im1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=22)
    axs[0].set_xlabel('y (mm)', fontsize=22)
    axs[0].set_ylabel('z (mm)', fontsize=22)
    axs[0].set_aspect('equal')

    # Plot the y midplane
    im2 = axs[1].pcolormesh(X[:, y_mid, :], Z[:, y_mid, :], f[:, y_mid, :], shading='auto', vmin=min_f, vmax=max_f)
    cbar2 = fig.colorbar(im2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=22)
    axs[1].set_xlabel('x (mm)', fontsize=22)
    axs[1].set_ylabel('z (mm)', fontsize=22)
    axs[1].set_aspect('equal')

    # Plot the z midplane
    im3 = axs[2].pcolormesh(X[:, :, z_mid], Y[:, :, z_mid], f[:, :, z_mid], shading='auto', vmin=min_f, vmax=max_f)
    cbar3 = fig.colorbar(im3, orientation='vertical')
    cbar3.ax.tick_params(labelsize=22)
    axs[2].set_xlabel('x (mm)', fontsize=22)
    axs[2].set_ylabel('y (mm)', fontsize=22)
    axs[2].set_aspect('equal')
    
    axs.tick_params(labelsize=22)
      
    plt.tight_layout()
    plt.show()
    
def plot_z_midplane(mesh, f, title='', colormap='viridis'):

    import matplotlib.pyplot as plt
    import numpy as np

    X,Y,Z = 1e3*mesh.X3D, 1e3*mesh.Y3D, 1e3*mesh.Z3D

    # Calculate the midplane index
    z_mid = Z.shape[2] // 2

    # Plot the z midplane
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X[:, :, z_mid], Y[:, :, z_mid], f[:, :, z_mid], shading='auto', cmap = colormap)
    ax.set_xlabel('x (mm)', fontsize=18)
    ax.set_ylabel('y (mm)', fontsize=18)
    ax.set_aspect('equal')
    cbar = fig.colorbar(im, orientation='vertical', shrink=0.3, aspect=7)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16)
    plt.title(title, fontsize=22)
    plt.show()

def computeSourceTerms(P,T,Y_sp, reshape=True):
    from tqdm import tqdm
    import cantera as ct
    import numpy as np
    
    gas = ct.Solution('ch4Coffee.yaml')
    
    Nx,Ny,Nz = P.shape
    nSp = Y_sp.shape[3]
    
    P = P.reshape(Nx*Ny*Nz,1)
    T = T.reshape(Nx*Ny*Nz,1)
    Y_sp = Y_sp.reshape(Nx*Ny*Nz, nSp)
    
    R = np.zeros(Y_sp.shape)
    HRR = np.zeros(T.shape)
    D = np.zeros(T.shape)
    
    R = R.astype(np.float32)
    HRR = HRR.astype(np.float32)
    D = D.astype(np.float32)
    
    print('Computing Heat Release Rates, Net Production Rates and Diffusivity:\n')
    
    for i in tqdm(range(len(T)), desc='Progress', unit='%', ascii=True):
    #for i in range(3):
        gas.TPY = T[i], P[i], Y_sp[i,:]
        R[i,:] = gas.net_production_rates*gas.molecular_weights
        HRR[i] = gas.heat_release_rate
        D[i] = np.dot(Y_sp[i,:], gas.mix_diff_coeffs_mass)
    
    if reshape == True:
        R = R.reshape(Nx,Ny,Nz,nSp)
        HRR = HRR.reshape(Nx,Ny,Nz)
        D = D.reshape(Nx,Ny,Nz)
        
    return R, HRR, D

def computeProgressVariable(Y):
    Yburnt = Y.min()
    Yunburnt = Y.max()

    C = 1 - ( (Y - Yburnt)/(Yunburnt - Yburnt))
    
    return C

def gradientModule(F,X,Y,Z):
    import numpy as np
    F_x,F_y,F_z = np.gradient(F, np.unique(X), np.unique(Y), np.unique(Z))
    grad_C = np.sqrt(F_x**2 + F_y**2 + F_z**2)
    return grad_C
    
def filterGaussDS_reactionRates(R, delta, n_cut=0, mute=True):
    
    import numpy as np
    from tqdm import tqdm
    
    Nx,Ny,Nz,nSp = R.shape
    
    R_DNS = [filterGaussDS(R[:,:,:,i], delta, n_cut, mute=True) for i in range(nSp)] # This returns a list
    
    R_DNS = np.stack(R_DNS, axis=-1) # this joins the list of 3d arrays along a fourth axis (axis = -1)
    
    return R_DNS

def compute_strain_rate(U, V, W, mesh, verbose=False, mode='strain_rate'):
    import os
    # types check
    if (not isinstance(U, Scalar3D)) or (not isinstance(V, Scalar3D)) or (not isinstance(W, Scalar3D)):
        raise TypeError("U,V, and W must be an object of the class Scalar3D")
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    # shape check
    if not check_same_shape(U, V, W, mesh):
        raise ValueError("The velocity components and the mesh must have the same shape")
    # check that mode is in the valid ones
    valid_modes = ['strain_rate', 'derivatives']
    check_input_string(mode, valid_modes, 'mode')
    
    
    shape = U.shape
    
    # define the list to use to change the file name
    path_list = U.path.split('/')
    if hasattr(U, 'file_id'):
        file_id = U.file_id
    else:
        file_id = ''
    
    #------------------------ Compute dU/dx ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUX_dX"] + '_' + file_id + '.dat' 
    file_name = '/'.join(path_list)                                
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dU/dx...")
        dU_dx = gradient_x(U, mesh, U.filter_size)        
        save_file(dU_dx, file_name)
        del dU_dx
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dU_dx = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dU/dy ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUX_dY"] + '_' + file_id + '.dat'
    file_name = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dU/dy...")
        dU_dy = gradient_y(U, mesh, U.filter_size)
        save_file(dU_dy, file_name)
        del dU_dy
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dU_dy = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dU/dz ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUX_dZ"] + '_' + file_id + '.dat'
    file_name = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dU/dz...")
        dU_dz         = gradient_z(U, mesh, U.filter_size)
        save_file(dU_dz, file_name)
        del dU_dz
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dU_dz         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dV/dx ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUY_dX"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dV/dx...")
        dV_dx         = gradient_x(V, mesh, U.filter_size)
        save_file(dV_dx, file_name)
        del dV_dx
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dV_dx         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dV/dy ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUY_dY"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dV/dy...")
        dV_dy         = gradient_y(V, mesh, U.filter_size)
        save_file(dV_dy, file_name)
        del dV_dy
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dV_dy         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dV/dz ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUY_dZ"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dV/dz...")
        dV_dz         = gradient_z(V, mesh, U.filter_size)
        save_file(dV_dz, file_name)
        del dV_dz
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dV_dz         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dW/dx ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUZ_dX"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dW/dx...")
        dW_dx         = gradient_x(W, mesh, U.filter_size)
        save_file(dW_dx, file_name)
        del dW_dx
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dW_dx         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dW/dy ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUZ_dY"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dW/dy...")
        dW_dy         = gradient_y(W, mesh, U.filter_size)
        save_file(dW_dy, file_name)
        del dW_dy
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dW_dy         = Scalar3D(shape, path=file_name)
    
    #------------------------ Compute dW/dz ----------------------------------#
    path_list[-1] = Field3D.variables_names["dUZ_dZ"] + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    # Check if the file already exists
    if not os.path.exists(file_name):
        if verbose:
            print("Computing dW/dz...")
        dW_dz         = gradient_z(W, mesh, U.filter_size)
        save_file(dW_dz, file_name)
        del dW_dz
    else:
        if verbose:
            print("File {} already exists".format(file_name))
    dW_dz         = Scalar3D(shape, path=file_name)
    
    
    #------------------------ Compute Strain Rate ----------------------------#
    
    if mode=='derivatives':
        return dU_dx, dU_dy, dU_dz, dV_dx, dV_dy, dV_dz, dW_dx, dW_dy, dW_dz

    S = np.zeros(shape)
    S += np.sqrt( 2*((dU_dx._3d)**2) + 2*((dV_dy._3d)**2) + 2*((dW_dz._3d)**2) + (dU_dy._3d+dV_dx._3d)**2 + (dU_dz._3d+dW_dx._3d)**2 + (dV_dz._3d+dW_dy._3d)**2 )
    
    # Save file
    LES = ''
    if hasattr(U, 'filter_size'):
        LES = '_LES'
    path_list[-1] = Field3D.variables_names["strain rate"] + LES + '_' + file_id + '.dat'
    file_name     = '/'.join(path_list)
    save_file(S, file_name)
    
    return S

def computeStrainRateOld (uFilt, vFilt, wFilt, xDS, yDS, zDS):
    import numpy as np
    # compute the velocity derivatives
    uFilt_x,uFilt_y,uFilt_z = np.gradient(uFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))
    vFilt_x,vFilt_y,vFilt_z = np.gradient(vFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))
    wFilt_x,wFilt_y,wFilt_z = np.gradient(wFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))
    
    S = np.zeros_like(uFilt)
    # compute the filtered rate-of-strain tensor
    for i in range(uFilt.shape[0]):
        for j in range(uFilt.shape[1]):
            for k in range(uFilt.shape[2]):
                S_tensor = np.zeros([3,3])
                S_tensor[0,0] = 0.5*(uFilt_x[i,j,k] + uFilt_x[i,j,k])
                S_tensor[0,1] = 0.5*(uFilt_y[i,j,k] + vFilt_x[i,j,k])
                S_tensor[0,2] = 0.5*(uFilt_z[i,j,k] + wFilt_x[i,j,k])
                S_tensor[1,0] = 0.5*(vFilt_x[i,j,k] + uFilt_y[i,j,k])
                S_tensor[1,1] = 0.5*(vFilt_y[i,j,k] + vFilt_y[i,j,k])
                S_tensor[1,2] = 0.5*(vFilt_z[i,j,k] + wFilt_y[i,j,k])
                S_tensor[2,0] = 0.5*(wFilt_x[i,j,k] + uFilt_z[i,j,k])
                S_tensor[2,1] = 0.5*(wFilt_y[i,j,k] + vFilt_z[i,j,k])
                S_tensor[2,2] = 0.5*(wFilt_z[i,j,k] + wFilt_z[i,j,k])
                
                S[i,j,k] = np.sqrt(np.sum(2*S_tensor**2))
                # S[i,j,k] = np.sum(2*S_tensor)
    
    return S

def computeDissipationRate (uFilt, vFilt, wFilt, xDS, yDS, zDS, mu):
    import numpy as np
    # compute the velocity derivatives
    u_x,u_y,u_z = np.gradient(uFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))
    v_x,v_y,v_z = np.gradient(vFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))
    w_x,w_y,w_z = np.gradient(wFilt, np.unique(xDS), np.unique(yDS), np.unique(zDS))

    epsilon = 2*(u_x**2) + u_y**2 + u_z**2 + v_x**2 + 2*(v_y**2) + v_z**2 + w_x**2 + w_y**2  + 2*(w_z**2) + 2*(u_y*v_x + u_z*w_x + v_z*w_y)
    epsilon = mu*epsilon
    
    return epsilon

def save_file (X, filename):
    import numpy as np
    X = X.astype(np.float32)
    X.tofile(filename)
    
def computeChemicalTimescale (R,Y,RHO):
    import numpy as np
    Nx,Ny,Nz,nSp = R.shape
    R = R.reshape(Nx*Ny*Nz,nSp)
    Y = Y.reshape(Nx*Ny*Nz,nSp)
    RHO = RHO.reshape(Nx*Ny*Nz,1)
    # tau_c = [(np.max(np.abs(R[i,:])/(Y[i,:]))*RHO[i]) for i in range(Nx*Ny*Nz)]
    tau_c = [(np.max(np.abs(R[i,:])/(Y[i,:]))*RHO) for i in range(Nx*Ny*Nz)]

    tau_c = np.hstack(tau_c).reshape(Nx,Ny,Nz)
    
    tau_c[tau_c==0] = 1e30
    
    tau_c = 1/tau_c
    
    return tau_c

def gradient_x(F, mesh, filter_size):
    '''
        Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
        used to computed the gradients on all the possible downsampled grids

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        verbose : bool
            If True, it will output relevant information.

        Returns
        -------
        grad_x : numpy array
            The x component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_x = np.zeros(F._3d.shape)
    X1D = mesh.X1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[start::filter_size, :, :]
        
        grad_x[start::filter_size, :, :] = np.gradient(field, X1D[start::filter_size], axis=0)
        
    return grad_x

def gradient_y(F, mesh, filter_size):
    '''
        Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
        used to computed the gradients on all the possible downsampled grids

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        verbose : bool
            If True, it will output relevant information.

        Returns
        -------
        grad_y : numpy array
            The y component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_y = np.zeros(F._3d.shape)
    Y1D = mesh.Y1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[:, start::filter_size, :]
        
        grad_temp = np.gradient(field, Y1D[start::filter_size], axis=1)
        
        grad_y[:, start::filter_size, :] = grad_temp
        
    return grad_y

def gradient_z(F, mesh, filter_size):
    '''
        Computes the gradient of a 3D, non downsampled, filtered field. Numpy is
        used to computed the gradients on all the possible downsampled grids

        Parameters
        ----------
        Ur : Scalar3D object
            Is the field to filter.
        
        mesh : Mesh3D object
            Is the mesh object used to compute the derivatives.
            
        filter_size : int
            Is the filter size used to filter the field.
        
        Returns
        -------
        grad_z : numpy array
            The z component of the gradient            
        '''
    import time
    # Check the data types of the input
    if not isinstance(mesh, Mesh3D):
        raise TypeError("mesh must be an object of the class Mesh3D")
    if not isinstance(F, Scalar3D):
        raise TypeError("F must be an object of the class Scalar3D")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    Nx = F.Nx
    Ny = F.Ny
    Nz = F.Nz
    
    grad_z = np.zeros(F._3d.shape)
    Z1D = mesh.Z1D
    for i in range(filter_size):
        start = i
        
        field = F._3d[:, :, start::filter_size]
        
        grad_temp = np.gradient(field, Z1D[start::filter_size], axis=2)
        
        grad_z[:, :, start::filter_size] = grad_temp
        
    return grad_z

def generate_mask(start, shape, delta):
    '''
        Computes the downsampled mask of a 3D field.

        Parameters
        ----------
        
        start : list of int
            Is the a list with the indexes where to start doing the mask.
            
        shape : list of int
            Is the shape of the input field
        
        delta : int
            Is the filter size

        Returns
        -------
        mask : numpy array of bool
            A 3D vector of boolean values.
            
        '''
    import numpy as np
    
    idx_x = np.arange(start[0], shape[0], delta)
    idx_y = np.arange(start[1], shape[1], delta)
    idx_z = np.arange(start[2], shape[2], delta)
    
    # Create mask
    mask = np.zeros((shape[0], shape[1], shape[2]), dtype=bool)
    mask[idx_x[:, None, None], idx_y[None, :, None], idx_z[None, None, :]] = True

    return mask

def check_same_shape(*args):
    '''
        Checks if the shape of the input arguments *args is the same

        Returns
        -------
        bool : bool
            Assumes the value True only if all the inputs have the same shape.
            
        '''
    # Check if there are at least two arguments
    if len(args) < 2:
        raise ValueError("At least two arguments are required")
    
    # Get the shape of the first argument
    reference_shape = args[0].shape
    
    # Check the shape of each argument against the reference shape
    for arg in args[1:]:
        if arg.shape != reference_shape:
            return False
    
    return True

def check_input_string(input_string, valid_strings, input_name):
    '''
        Checks if the value of input_string is contained in the list valid_strings.
        If the result is positive, returns None, if the result is negative
        raises an error
        
        Parameters
        ----------
        
        input_string : string
            Is the string that must be checked
            
        valid_strings : list of strings
            Is the list of valid strings
        
        input_name : string
            Is the name of the parameter that we are checking

        Returns
        -------
        None 
        
        RNOTES:
        -------
        Example of output if the function finds an error:
        
        ValueError: Invalid parameter mode 'mode1'. Valid options are: 
         - mode_1
         - mode_2
         - mode_3
        
        '''
    input_lower = input_string.lower()
    valid_strings_lower = [valid_string.lower() for valid_string in valid_strings]

    if input_lower not in valid_strings_lower:
        valid_strings_ = "\n - ".join(valid_strings)
        raise ValueError("Invalid parameter {} '{}'. Valid options are: \n - {}".format(input_name, input_string, valid_strings_))


# %% Process data in chuncks
import os
import numpy as np
import cantera as ct
def read_variable_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        while True:
            # Read a chunk of data
            data_chunk = np.fromfile(file, dtype='<f4', count=chunk_size)
            # If no more data to read, break the loop
            if len(data_chunk) == 0:
                break
            yield data_chunk

def process_species_in_chunks(file_paths, species_file, chunk_size):
    species_data_chunks = []
    for specie_file in species_file:
        species_data_chunks.append([])
        for data_chunk in read_variable_in_chunks(file_paths['folder_path'] + file_paths['data_path'] + specie_file, chunk_size):
            species_data_chunks[-1].append(data_chunk)
    return species_data_chunks

def process_and_save(file_paths, chunk_size):
    # Load gas mechanism
    gas = ct.Solution(file_paths['chem_path'] + file_paths['chem_mech'])
    
    # Read temperature data in chunks
    for T_chunk in read_variable_in_chunks(file_paths['folder_path'] + file_paths['data_path'] + file_paths['T_filename'], chunk_size):
        # Read pressure data in chunks
        for P_chunk in read_variable_in_chunks(file_paths['folder_path'] + file_paths['data_path'] + file_paths['P_filename'], chunk_size):
            # Read species mass fractions data in chunks
            species_data_chunks = process_species_in_chunks(file_paths, file_paths['species_filename'], chunk_size)
            
            # Initialize arrays for results
            R_chunk = np.zeros((chunk_size, len(file_paths['species'])))
            HRR_chunk = np.zeros(chunk_size)
            D_chunk = np.zeros(chunk_size)
            
            for i in range(chunk_size):
                gas.TPY = T_chunk[i], P_chunk[i], np.concatenate([specie_data[i] for specie_data in species_data_chunks], axis=0)
                R_chunk[i, :] = gas.net_production_rates * gas.molecular_weights
                HRR_chunk[i] = gas.heat_release_rate
                D_chunk[i] = np.dot(np.concatenate([specie_data[i] for specie_data in species_data_chunks], axis=0), gas.mix_diff_coeffs_mass)
            
            # Save processed data
            save_folder = file_paths['folder_path'] + file_paths['data_path'] + "processed_data/"
            os.makedirs(save_folder, exist_ok=True)
            np.save(save_folder + f"T_chunk_{chunk_size}", T_chunk)
            np.save(save_folder + f"P_chunk_{chunk_size}", P_chunk)
            np.save(save_folder + f"R_chunk_{chunk_size}", R_chunk)
            np.save(save_folder + f"HRR_chunk_{chunk_size}", HRR_chunk)
            np.save(save_folder + f"D_chunk_{chunk_size}", D_chunk)

def merge_chunks(file_paths, chunk_size):
    save_folder = file_paths['folder_path'] + file_paths['data_path'] + "processed_data/"
    # Initialize arrays for merged results
    T_merged = np.empty((0))
    P_merged = np.empty((0))
    R_merged = np.empty((0, len(file_paths['species'])))
    HRR_merged = np.empty((0))
    D_merged = np.empty((0))

    # Read and concatenate data from each chunk
    for i in range(20):
        T_chunk = np.load(save_folder + f"T_chunk_{chunk_size}.npy")
        P_chunk = np.load(save_folder + f"P_chunk_{chunk_size}.npy")
        R_chunk = np.load(save_folder + f"R_chunk_{chunk_size}.npy")
        HRR_chunk = np.load(save_folder + f"HRR_chunk_{chunk_size}.npy")
        D_chunk = np.load(save_folder + f"D_chunk_{chunk_size}.npy")
        
        T_merged = np.concatenate((T_merged, T_chunk))
        P_merged = np.concatenate((P_merged, P_chunk))
        R_merged = np.concatenate((R_merged, R_chunk))
        HRR_merged = np.concatenate((HRR_merged, HRR_chunk))
        D_merged = np.concatenate((D_merged, D_chunk))
        
    return T_merged, P_merged, R_merged, HRR_merged, D_merged


# def filterGaussDS_reactionRates(R, delta, n_cut=0, mute=True):
#     # takes as input the Reaction Rates matrix (Nx, Ny, Nz, nSp)
#     # returns the Filtered Reaction Rates
#     # you can use this function both with the Reaction Rates and the Mass Fractions
#     import numpy as np
#     from tqdm import tqdm
    
#     Nx,Ny,Nz,nSp = R.shape
    
#     R = R.reshape(Nx*Ny*Nz, nSp)
#     for i in tqdm(range(nSp), desc='Progress', unit='%', ascii=True):
#         if i == 0:
#             R_DNS = filterGaussDS(R[:,i].reshape(Nx,Ny,Nz), delta, n_cut)
#             NxFilt,NyFilt,NzFilt = R_DNS.shape
#             R_DNS = R_DNS.reshape(-1)
#         else:
#             R_DNS = np.hstack([R_DNS, filterGaussDS(R[:,i].reshape(Nx,Ny,Nz), delta, n_cut).reshape(-1)])

#     R_DNS = R_DNS.reshape(NxFilt, NyFilt, NzFilt, nSp)
    
#     return R_DNS


# def filterGaussDS(field,X,Y,Z,delta,n_cut):
#     # field must be a 3d array, with the same size as the grid points in X,Y,Z.
#     # delta is the amplitude of the gaussian filter
#     # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema)
#     import numpy as np
#     from scipy.ndimage import gaussian_filter
#     from scipy.signal import decimate
    
#     # Define the dimensions of the 3D field
#     Nx, Ny, Nz = np.shape(field)
    
#     fieldFilt = gaussian_filter(field, sigma=1/12*delta**2, mode='constant')

#     # Downsample the filtered field
#     fieldDS = fieldFilt[::delta, ::delta, ::delta]
    
#     xDS = X[::delta, ::delta, ::delta]
#     yDS = Y[::delta, ::delta, ::delta]
#     zDS = Z[::delta, ::delta, ::delta]
    
#     # cut the extrema because the filtering operation creates nonsense values at the Boundaries:
#     n_cut = 2 #amount of data to cut from both sides, both in the y and z direction
#     # cutIndexes = np.hstack[range(0,Nx-1),range(n_cut,Ny-1-n_cut),range(n_cut,Ny-1-n_cut)] #this line doesnt work

#     fieldDS = fieldDS[n_cut:fieldDS.shape[0]-n_cut,n_cut:fieldDS.shape[1]-n_cut,n_cut:fieldDS.shape[2]-n_cut]
#     xDS = xDS[n_cut:xDS.shape[0]-n_cut,n_cut:xDS.shape[1]-n_cut,n_cut:xDS.shape[2]-n_cut]
#     yDS = yDS[n_cut:yDS.shape[0]-n_cut,n_cut:yDS.shape[1]-n_cut,n_cut:yDS.shape[2]-n_cut]
#     zDS = zDS[n_cut:zDS.shape[0]-n_cut,n_cut:zDS.shape[1]-n_cut,n_cut:zDS.shape[2]-n_cut]
    
#     # Print the original and filtered field shapes
#     print("Original field shape:", field.shape)
#     print("Filtered field shape:", fieldDS.shape)
    
#     return fieldDS, xDS, yDS, zDS

# def varianceDSold(field,delta,n_cut):
#     # field must be a 3d array, with the same size as the grid points in X,Y,Z.
#     # delta is the amplitude of the box to consider for the variance
#     # n_cut is the number of samples to cut at the extrema (because with the filtering operation we obtain strange values at the extrema. Even if this function just downsamples, we keep this to be consistent with the arrays' dimensions)
    
#     import numpy as np
#     from scipy.ndimage import generic_filter
    
#     # compute the variance in every box of size (delta x delta x delta) centered in each cell
#     def compute_variance(x):
#         return np.var(x)
#     # Compute the variance of the 3D array in every cell
#     fieldVar = generic_filter(field, compute_variance, size=filterSize)
    
#     # Downsample and cut the filtered field
#     fieldDS = downsampleAndCut(fieldVar, delta, n_cut)
    
#     return fieldDS