# Scalar3D class

[![Static Badge](https://img.shields.io/badge/Source%20Code-06D40C?style=plastic\&logo=github)](../../DNS.py#L104-L537)

![Static Badge](https://img.shields.io/badge/Source%20Code-06D40C?style=plastic\&logo=github\&color=FFFFF)

![Static Badge](https://img.shields.io/badge/Source%20Code-06D40C?style=plastic\&logo=github\&color=FFFFF)

### **aPriori.DNS.**<mark style="color:red;">**Scalar3D**</mark>**(self, shape, value=None, path=''):**

A class used to represent a 3D scalar field.

### Attributes

* `shape` : list
  * A list of 3 integers representing the shape of the 3D field.
* `value` : ndarray
  * The values of the field, reshaped into a 3D array by default.
* `path` : str
  * The file path where the field data is stored.
* `Nx, Ny, Nz` : int
  * The dimensions of the field along the x, y, and z axes, respectively.
* `file_name` : str
  * The name of the file where the field data is stored.
* `file_id` : str
  * The ID of the file where the field data is stored.
* `filter_size` : int
  * The size of the filter used for processing the field data.

### Methods

* `is_light_mode()`
  * Checks if the field data is stored in memory or in a file.
* `reshape_3d()`
  * Reshapes the field data into a 3D array.
* `reshape_column()`
  * Reshapes the field data into a column vector.
* `reshape_line()`
  * Reshapes the field data into a row vector.
* `cut(n_cut=1, mode='equal')`
  * Cuts the field data along the edges.
* `filter_gauss(delta, n_cut=0, mute=False)`
  * Filters the field data using a Gaussian filter.
* `plot_x_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)`
  * Plots the field data on the x midplane.
* `plot_y_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)`
  * Plots the field data on the y midplane.
* `plot_z_midplane(mesh, title='', colormap='viridis', vmin=None, vmax=None)`
  * Plots the field data on the z midplane.

### Notes

The field data can be either in memory or in a file. If it's in a file, the file path, name, and ID are required. The reshape methods are used to change the shape of the field data for different purposes. The cut method is used to remove the edges of the field data. The filter\_gauss method is used to smooth the field data. The plot methods are used to visualize the field data on different planes.

### Examples

```renpy
import numpy as np
field = Scalar3D(shape=[10, 10, 10], value=np.random.rand(1000))
print(field.is_light_mode())  # True
field.reshape_column()
print(field.value.shape)  # (1000, 1)
field.cut(n_cut=2)
print(field.value.shape)  # (8, 8, 8)
field.filter_gauss(delta=1)
field.plot_x_midplane(mesh=np.meshgrid(np.arange(8), np.arange(8), np.arange(8)))
```

***

\
