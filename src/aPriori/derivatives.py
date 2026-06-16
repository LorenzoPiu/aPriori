#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""derivatives.py: the module that handles derivatives (gradients and laplacians).

"""

__authors__ = "Lorenzo Piu"
__copyright__ = "Copyright (c) 2024-2026, Lorenzo Piu, Heinz Pitsch, and Alessandro Parente"
__credits__ = ["Aero-Thermo-Mechanics laboratories - Universite Libre de Bruxelles, Brussels, Belgium"
               "Institut für Technische Verbrennung (ITV) - RWTH Aachen University, Aachen, Germany"]
__license__ = "GPL 3.0"
__version__ = "1.3.2"
__maintainer__ = ["Lorenzo Piu"]
__email__ = ["lorenzo.piu@ulb.be"]
__status__ = "Production"


import numpy as np
from findiff import FinDiff
from .DNS import Scalar
from .DNS import Mesh


_DEFAULT_DERIVATIVES_ORDER = 4

def set_gradients_order(order: int) -> None:
    global _DEFAULT_DERIVATIVES_ORDER
    _DEFAULT_DERIVATIVES_ORDER = order

def print_gradients_order(order=None):
    if order is None:
        order = _DEFAULT_DERIVATIVES_ORDER
    print(f"Using order {order}")



def _apply_findiff_on_strided_grid(data, coords_1d, axis, filter_size, acc, deriv_order):
    """
    Internal helper function to compute derivatives on a potentially 
    strided (downsampled) grid using findiff.

    Parameters
    ----------
    data : numpy.ndarray
        The 3D array on which to compute the derivative.
    coords_1d : numpy.ndarray
        The 1D coordinates corresponding to the 'axis' of interest.
    axis : int
        The axis (0, 1, or 2) along which to compute the derivative.
    filter_size : int
        The stride/downsampling factor.
    acc : int
        Accuracy order for findiff.
    deriv_order : int
        The order of the derivative (1 for gradient, 2 for Laplacian).

    Returns
    -------
    result : numpy.ndarray
        The computed derivative with the same shape as 'data'.
    """
    result = np.zeros_like(data)

    # Loop over the stride offsets
    for i in range(filter_size):
        # 1. Create the slice object for the 3D data
        # We want equivalent of data[i::filter_size, :, :] if axis=0
        slices = [slice(None)] * 3
        slices[axis] = slice(i, None, filter_size)
        tuple_slices = tuple(slices)

        # 2. Extract the strided data
        sub_field = data[tuple_slices]

        # 3. Extract the strided coordinates for this specific axis
        sub_coords = coords_1d[i::filter_size]

        # 4. Check if we have enough points for the requested accuracy
        # Findiff generally requires roughly 'acc' points. 
        if len(sub_coords) <= acc:
             print(f"Warning: Grid too coarse (N={len(sub_coords)}) for requested accuracy ({acc}) on stride {i}.")

        # 5. Define and apply the FinDiff operator
        # FinDiff(axis, spacing/coords, derivative_order, accuracy)
        fd = FinDiff(axis, sub_coords, deriv_order, acc=acc)
        
        # 6. Compute and place back into the result array
        result[tuple_slices] = fd(sub_field)

    return result


def gradient_x(F, mesh, filter_size=1, acc=8, reduce_acc=False):
    '''
    Description
    -----------
    Computes the x-component of the gradient using findiff with specified accuracy.
    Uses striding (filter_size) to compute derivatives on downsampled grids.

    Parameters
    ----------
    F : Scalar object
        The field to differentiate.
    mesh : Mesh object
        The mesh object containing coordinates (X1D).
    filter_size : int, optional
        The stride used to filter the field (default is 1).
    acc : int, optional
        The accuracy order for the finite difference scheme (default is 2).

    Returns
    -------
    grad_x : numpy array
        The x component of the gradient.
    '''
    # 1. Type Checking
    # Note: Assuming Scalar/Mesh names are available in the namespace. 
    # If not imported, use F.__class__.__name__ == 'Scalar' or similar.
    if type(F).__name__ != 'Scalar':
        raise TypeError("F must be an object of the class Scalar")
    if type(mesh).__name__ != 'Mesh':
        raise TypeError("mesh must be an object of the class Mesh")
    if not isinstance(filter_size, int):
        raise TypeError("filter_size must be an integer")
    if not isinstance(acc, int):
        raise TypeError("acc must be an integer")
    
    if reduce_acc: # Switch to decrease gradients accuracy (used for example on LES grids)
        acc=2

    # 2. Call Internal Helper
    return _apply_findiff_on_strided_grid(
        data=F._3d,
        coords_1d=mesh.X1D,
        axis=0,
        filter_size=filter_size,
        acc=acc,
        deriv_order=1
    )


def gradient_y(F, mesh, filter_size=1, acc=8, reduce_acc=False):
    '''
    Description
    -----------
    Computes the y-component of the gradient using findiff.
    '''
    if type(F).__name__ != 'Scalar':
        raise TypeError("F must be an object of the class Scalar")
    if type(mesh).__name__ != 'Mesh':
        raise TypeError("mesh must be an object of the class Mesh")
    if not isinstance(acc, int):
        raise TypeError("acc must be an integer")
    
    if reduce_acc:
        acc = 2

    return _apply_findiff_on_strided_grid(
        data=F._3d,
        coords_1d=mesh.Y1D,
        axis=1,
        filter_size=filter_size,
        acc=acc,
        deriv_order=1
    )


def gradient_z(F, mesh, filter_size=1, acc=8, reduce_acc=False):
    '''
    Description
    -----------
    Computes the z-component of the gradient using findiff.
    '''
    if type(F).__name__ != 'Scalar':
        raise TypeError("F must be an object of the class Scalar")
    if type(mesh).__name__ != 'Mesh':
        raise TypeError("mesh must be an object of the class Mesh")
    if not isinstance(acc, int):
        raise TypeError("acc must be an integer")
    
    if reduce_acc:
        acc = 2

    return _apply_findiff_on_strided_grid(
        data=F._3d,
        coords_1d=mesh.Z1D,
        axis=2,
        filter_size=filter_size,
        acc=acc,
        deriv_order=1
    )


def laplacian(F, mesh, filter_size=1, acc=4, reduce_acc=False):
    '''
    Description
    -----------
    Computes the Laplacian (d2/dx2 + d2/dy2 + d2/dz2) of the field.
    
    Parameters
    ----------
    F : Scalar object
    mesh : Mesh object
    filter_size : int, optional
    acc : int, optional

    Returns
    -------
    lap : numpy array
        The Laplacian of the input field.
    '''
    if type(F).__name__ != 'Scalar':
        raise TypeError("F must be an object of the class Scalar")
    if type(mesh).__name__ != 'Mesh':
        raise TypeError("mesh must be an object of the class Mesh")
    
    if reduce_acc:
        acc = 2

    # Compute second derivatives for each axis
    laplacian_value  = _apply_findiff_on_strided_grid(F._3d, mesh.X1D, 0, filter_size, acc, deriv_order=2)
    laplacian_value += _apply_findiff_on_strided_grid(F._3d, mesh.Y1D, 1, filter_size, acc, deriv_order=2)
    laplacian_value += _apply_findiff_on_strided_grid(F._3d, mesh.Z1D, 2, filter_size, acc, deriv_order=2)

    return laplacian_value