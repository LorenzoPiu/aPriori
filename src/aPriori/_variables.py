#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List of attributes available for data processing
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

variables_list = {
           # Attribute Name   :           File Name             :  Species  :         Models          :      Tensor      :     Reactions      :     Variable Description
             "dU{}_dX{}_{}"   : ["dU{}_dX{}_{}_s-1_{}.dat"      ,   False   ,    ['DNS', 'LES']       ,      True        ,       False        , "Derivative with respect to j of the ith velocity component"],
             "S_{}"           : ["S_{}_s-1_{}.dat"              ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Module of the Strain Rate tensor"],
             "S{}{}_{}"       : ["S{}{}_{}_s-1_{}.dat"          ,   False   ,    ['DNS', 'LES']       ,   'Symmetric'    ,       False        , "Module of the Strain Rate tensor"],
             "P"              : ["P_Pa_{}.dat"                  ,   False   ,       None              ,      False       ,       False        , "Pressure"],
             "RHO"            : ["RHO_kgm-3_{}.dat"             ,   False   ,       None              ,      False       ,       False        , "Density"],
             "T"              : ["T_K_{}.dat"                   ,   False   ,       None              ,      False       ,       False        , "Temperature"],
             "U_{}"           : ["U_{}_ms-1_{}.dat"             ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Module of the velocity vector"],
             "U_X"            : ["UX_ms-1_{}.dat"               ,   False   ,       None              ,      False       ,       False        , "X component of the velocity"],
             "U_Y"            : ["UY_ms-1_{}.dat"               ,   False   ,       None              ,      False       ,       False        , "Y component of the velocity"],
             "U_Z"            : ["UZ_ms-1_{}.dat"               ,   False   ,       None              ,      False       ,       False        , "Z component of the velocity"],
             "Y{}"            : ["Y{}_{}.dat"                   ,   True    ,       None              ,      False       ,       False        , "Species' mass fraction; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_LFR"        : ["R{}_LFR_kgm-3s-1_{}.dat"      ,   True    ,       None              ,      False       ,       False        , "Species' Reaction Rates, computed as the Arrhenius rates on the filtered values of T, Y; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_DNS"        : ["R{}_DNS_kgm-3s-1_{}.dat"      ,   True    ,       None              ,      False       ,       False        , "Species' Reaction Rates, computed as the Arrhenius rates on the DNS values of T, Y, and eventually filtered; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_Batch"      : ["R{}_Batch_kgm-3s-1_{}.dat"    ,   True    ,       None              ,      False       ,       False        , "Species' Reaction Rates, computed as the PaSR rates on the DNS values of T, Y, and eventually filtered; expected one scalar field for every specie in the chemical mechanism"],
             "Tau_c_{}"       : ["Tau_c_{}_s_{}.dat"            ,   False   , ['Ch', 'SFR', 'FFR']    ,      False       ,       False        , "Chemical timescale. Models available: Chomiak ('Ch'), Slowest Formation Rate ('SFR'), Fastest Formation Rate ('FFR')"],
             "Tau_m_{}"       : ["Tau_m_{}_s_{}.dat"            ,   False   , ['Kolmo', 'Int', 'Sub'] ,      False       ,       False        , "Mixing timescale. Models available: Kolmogorov ('kolmo'), Integral ('int'), Subgrid velocity stretch ('sub')"],
             "TAU_r_{}{}_{}"  : ["TAU_r_{}{}_{}_m2s-2_{}.dat"   ,   False   , ['DNS', 'Smag']         ,   'Symmetric'    ,       False        , "Anisotropic part of the Residual stress tensor (filt(Ui_DNS*Uj_DNS)-(Ui_LES*Uj_LES) - 2/3*K_res. Models available: DNS ('DNS'), Smagorinsky ('Smag')"],
             "TAU_r_{}"       : ["TAU_r_{}_m2s-2_{}.dat"        ,   False   , ['DNS', 'Smag']         ,      False       ,       False        , "MODULE of the Anisotropic part of the Residual stress tensor. Models available: DNS ('DNS'), Smagorinsky ('Smag')"],
             "Epsilon"        : ["Epsilon_m2s-3_{}.dat"         ,   False   ,       None              ,      False       ,       False        , "Turbulence Dissipation Rate"],
             "Epsilon_r_{}"   : ["Epsilon_m2s-3_{}_{}.dat"      ,   False   , ['DNS', 'Smag']         ,      False       ,       False        , "Turbulence Dissipation Rate"],
             "K_{}"           : ["K_{}_m2s-2_{}.dat"            ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Turbulence Kinetic Energy"],
             "K_r_{}"         : ["K_r_{}_m2s-2_{}.dat"          ,   False   ,    ['DNS', 'Yosh']      ,      False       ,       False        , "Residual Turbulence Kinetic Energy"],
             "HRR_{}"         : ["HRR_{}_kgm2s-3_DNS_{}.dat"    ,   False   , ['DNS', 'LFR','Batch']  ,      False       ,       False        , "Heat Release Rate, computed on unfiltered DNS data"],
             "Mu"             : ["Mu_kgm-1s-1_{}.dat"           ,   False   ,       None              ,      False       ,       False        , "Dynamic Viscosity"],
             "Cp"             : ["Cp_W_m-1_K-1_{}.dat"          ,   False   ,       None              ,      False       ,       False        , "Specific heat"],
             "Lambda"         : ["Lambda_W_m-1_K-1_{}.dat"      ,   False   ,       None              ,      False       ,       False        , "Thermal conductivity"],
             "C"              : ["C_{}.dat"                     ,   False   ,       None              ,      False       ,       False        , "Progress variable"],
             "C_grad"         : ["C_grad_m-1_{}.dat"            ,   False   ,       None              ,      False       ,       False        , "Progress variable gradient (module)"],
             "C_grad_X"       : ["C_grad_X_m-1_{}.dat"          ,   False   ,       None              ,      False       ,       False        , "Progress variable gradient (x component)"],
             "C_grad_Y"       : ["C_grad_Y_m-1_{}.dat"          ,   False   ,       None              ,      False       ,       False        , "Progress variable gradient (y component)"],
             "C_grad_Z"       : ["C_grad_Z_m-1_{}.dat"          ,   False   ,       None              ,      False       ,       False        , "Progress variable gradient (z component)"],
             "C_laplacian"    : ["C_laplacian_m-2_{}.dat"       ,   False   ,       None              ,      False       ,       False        , "Progress variable laplacian"],
             "Z"              : ["Z_{}.dat"                     ,   False   ,       None              ,      False       ,       False        , "Mixture fraction"],
             "Z_grad"         : ["Z_grad_m-1_{}.dat"            ,   False   ,       None              ,      False       ,       False        , "Mixture fraction gradient modulus"],
             "Chi_Z"          : ["Chi_Z_ms-1_{}.dat"            ,   False   ,       None              ,      False       ,       False        , "Mixture fraction Dissipation rate"],
             "M"              : ["M_{}.dat"                     ,   False   ,       None              ,      False       ,       False        , "Fraction of resolved kinetic energy"],
             "PHI_T_X_{}"     : ["PHI_T_X_{}_kgKm-2s-1_{}.dat"  ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Temperature convective flux in the X direction"],
             "PHI_T_Y_{}"     : ["PHI_T_Y_{}_kgKm-2s-1_{}.dat"  ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Temperature convective flux in the Y direction"],
             "PHI_T_Z_{}"     : ["PHI_T_Z_{}_kgKm-2s-1_{}.dat"  ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Temperature convective flux in the Z direction"],
             "TAU_T_X"        : ["TAU_T_X_kgKm-2s-1_{}.dat"     ,   False   ,       None              ,      False       ,       False        , "Sub-filter temperature flux in the X direction"  ],
             "TAU_T_Y"        : ["TAU_T_Y_kgKm-2s-1_{}.dat"     ,   False   ,       None              ,      False       ,       False        , "Sub-filter temperature flux in the Y direction"  ],
             "TAU_T_Z"        : ["TAU_T_Z_kgKm-2s-1_{}.dat"     ,   False   ,       None              ,      False       ,       False        , "Sub-filter temperature flux in the Z direction"  ],
             "PHI_C_X_{}"     : ["PHI_C_X_{}_kgm-2s-1_{}.dat"   ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "PV convective flux in the X direction"],
             "PHI_C_Y_{}"     : ["PHI_C_Y_{}_kgm-2s-1_{}.dat"   ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "PV convective flux in the Y direction"],
             "PHI_C_Z_{}"     : ["PHI_C_Z_{}_kgm-2s-1_{}.dat"   ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "PV convective flux in the Z direction"],
             "TAU_C_X"        : ["TAU_C_X_kgm-2s-1_{}.dat"      ,   False   ,       None              ,      False       ,       False        , "Sub-filter PV flux in the X direction"  ],
             "TAU_C_Y"        : ["TAU_C_Y_kgm-2s-1_{}.dat"      ,   False   ,       None              ,      False       ,       False        , "Sub-filter PV flux in the Y direction"  ],
             "TAU_C_Z"        : ["TAU_C_Z_kgm-2s-1_{}.dat"      ,   False   ,       None              ,      False       ,       False        , "Sub-filter PV flux in the Y direction"  ],
             "TSR_{}"         : ["TSR_{}_{}.dat"                ,   False   ,    ['DNS', 'LES']       ,      False       ,       False        , "Tangential Stretching Rate"  ],
             "Y"              : ["Y_{}.dat"                     ,   False   ,       None              ,      False       ,       False        , "Generic transported scalar"  ],
             "API_TSR_{}"     : ["API_TSR_{}_{}.dat"            ,   False   ,       None              ,      False       ,       True         , "Participation indexes to the TSR"],
             "API_{}_{}"      : ["API_{}_{}_{}.dat"             ,   True    ,       None              ,      False       ,       True         , "Participation indexes on species production rates"],
            }

mesh_list  = {
             "X_mesh"         : ["X_m.dat"                    ,   False   ,       None              , ""],
             "Y_mesh"         : ["Y_m.dat"                    ,   False   ,       None              , ""],
             "Z_mesh"         : ["Z_m.dat"                    ,   False   ,       None              , ""]
             }