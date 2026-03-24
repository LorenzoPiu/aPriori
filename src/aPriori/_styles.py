#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference plotting parameters and styles that will be used in the plot_utils module.
"""
__authors__ = "Lorenzo Piu"
__copyright__ = "Copyright (c) 2024-2026, Lorenzo Piu, Heinz Pitsch, and Alessandro Parente"
__credits__ = ["Aero-Thermo-Mechanics laboratories - Universite Libre de Bruxelles, Brussels, Belgium"
               "Institut für Technische Verbrennung (ITV) - RWTH Aachen University, Aachen, Germany"]
__license__ = "GPL 3.0"
__version__ = "1.3.0"
__maintainer__ = ["Lorenzo Piu"]
__email__ = ["lorenzo.piu@ulb.be"]
__status__ = "Production"


class ParityPlot():
    fig_size      = [10, 8]
    dpi           = 400
    fontsize      = 48
    border_width  = 2
    
class ContourPlot():
    fig_size     = 4 # Will be adjusted based on the aspect ratio
    dpi          = 800
    fontsize     = 32
    
class CondMeanPlot():
    fig_size     = [8, 6]
    dpi          = 400
    fontsize     = 48
    border_width = 2

class ScatterPlot():
    fig_size     = [8, 6]
    dpi          = 400
    fontsize     = 48
    aspect_ratio = 1.2
    

