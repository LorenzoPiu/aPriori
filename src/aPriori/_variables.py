#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:16:15 2024

@author: lorenzo piu
"""
variables_list = {
           # Attribute Name   :           File Name           :  Species  :         Models          :      Tensor      :     Variable Description
             "dU{}_dX{}_{}"   : ["dU{}_dX{}_{}_s-1_{}.dat"    ,   False   ,    ['DNS', 'LES']       ,      True        , "Derivative with respect to j of the ith velocity component"],
             "S_{}"           : ["S_{}_s-1_{}.dat"            ,   False   ,    ['DNS', 'LES']       ,      False       , "Module of the Strain Rate tensor"],
             "S{}{}_{}"       : ["S{}{}_{}_s-1_{}.dat"        ,   False   ,    ['DNS', 'LES']       ,   'Symmetric'    , "Module of the Strain Rate tensor"],
             "P"              : ["P_Pa_{}.dat"                ,   False   ,       None              ,      False       , "Pressure"],
             "RHO"            : ["RHO_kgm-3_{}.dat"           ,   False   ,       None              ,      False       , "Density"],
             "T"              : ["T_K_{}.dat"                 ,   False   ,       None              ,      False       , "Temperature"],
             "U_{}"           : ["U_{}_ms-1_{}.dat"           ,   False   ,    ['DNS', 'LES']       ,      False       , "Module of the velocity vector"],
             "U_X"            : ["UX_ms-1_{}.dat"             ,   False   ,       None              ,      False       , "X component of the velocity"],
             "U_Y"            : ["UY_ms-1_{}.dat"             ,   False   ,       None              ,      False       , "Y component of the velocity"],
             "U_Z"            : ["UZ_ms-1_{}.dat"             ,   False   ,       None              ,      False       , "Z component of the velocity"],
             "Y{}"            : ["Y{}_{}.dat"                 ,   True    ,       None              ,      False       , "Species' mass fraction; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_LFR"        : ["R{}_LFR_kgm-3s-1_{}.dat"    ,   True    ,       None              ,      False       , "Species' Reaction Rates, computed as the Arrhenius rates on the filtered values of T, Y; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_DNS"        : ["R{}_DNS_kgm-3s-1_{}.dat"    ,   True    ,       None              ,      False       , "Species' Reaction Rates, computed as the Arrhenius rates on the DNS values of T, Y, and eventually filtered; expected one scalar field for every specie in the chemical mechanism"],
             "R{}_Batch"      : ["R{}_Batch_kgm-3s-1_{}.dat"    ,   True    ,       None              ,      False       , "Species' Reaction Rates, computed as the PaSR rates on the DNS values of T, Y, and eventually filtered; expected one scalar field for every specie in the chemical mechanism"],
             "Tau_c_{}"       : ["Tau_c_{}_s_{}.dat"          ,   False   , ['Ch', 'SFR', 'FFR']    ,      False       , "Chemical timescale. Models available: Chomiak ('Ch'), Slowest Formation Rate ('SFR'), Fastest Formation Rate ('FFR')"],
             "Tau_m_{}"       : ["Tau_m_{}_s_{}.dat"          ,   False   , ['Kolmo', 'Int']        ,      False       , "Mixing timescale. Models available: Kolmogorov ('kolmo'), Integral ('int')"],
             "TAU_r_{}{}_{}"  : ["TAU_r_{}{}_{}_m2s-2_{}.dat" ,   False   , ['DNS', 'Smag']         ,   'Symmetric'    , "Anisotropic part of the Residual stress tensor (filt(Ui_DNS*Uj_DNS)-(Ui_LES*Uj_LES) - 2/3*K_res. Models available: DNS ('DNS'), Smagorinsky ('Smag')"],
             "TAU_r_{}"       : ["TAU_r_{}_m2s-2_{}.dat"      ,   False   , ['DNS', 'Smag']         ,      False       , "MODULE of the Anisotropic part of the Residual stress tensor. Models available: DNS ('DNS'), Smagorinsky ('Smag')"],
             "Epsilon"        : ["Epsilon_m2s-3_{}.dat"       ,   False   ,       None              ,      False       , "Turbulence Dissipation Rate"],
             "Epsilon_r_{}"   : ["Epsilon_m2s-3_{}_{}.dat"    ,   False   , ['DNS', 'Smag']         ,      False       , "Turbulence Dissipation Rate"],
             "K_{}"           : ["K_{}_m2s-2_{}.dat"          ,   False   ,    ['DNS', 'LES']       ,      False       , "Turbulence Kinetic Energy"],
             "K_r_{}"         : ["K_r_{}_m2s-2_{}.dat"        ,   False   ,    ['DNS', 'Yosh']      ,      False       , "Residual Turbulence Kinetic Energy"],
             "HRR_{}"         : ["HRR_{}_kgm2s-3_DNS_{}.dat"  ,   False   , ['DNS', 'LFR','Batch']  ,      False       , "Heat Release Rate, computed on unfiltered DNS data"],
             "Mu"             : ["Mu_kgm-1s-1_{}.dat"         ,   False   ,       None              ,      False       , "Dynamic Viscosity"],
             "C"              : ["C_{}.dat"                   ,   False   ,       None              ,      False       , "Progress Variable"],
             "Z"              : ["Z_{}.dat"                   ,   False   ,       None              ,      False       , "Mixture Fraction"]

            }

mesh_list  = {
             "X_mesh"         : ["X_m.dat"                    ,   False   ,       None              , ""],
             "Y_mesh"         : ["Y_m.dat"                    ,   False   ,       None              , ""],
             "Z_mesh"         : ["Z_m.dat"                    ,   False   ,       None              , ""]
             }