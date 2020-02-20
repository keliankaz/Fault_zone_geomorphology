#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:03:44 2019

@author: kdascher
"""

def default_landscape_parameters():
    mdl_dict = {model_domain: {xmax:2000.,                  # length of the domain [meters]
                               ymax:1000.,                  # width of the domain [meters]
                               loopBuff:0.5,                # added buffer to loop around kdc
                               Nxy:100},                    # number of nodes in the y direction
                landscape:    {m:0.5,                       # exponent on drainage area [non dimensional]
                               n:1,                         # exponent on river slope [non dimensional]
                               Do:0.2,                      # Hillslope linear diffusivity [meters sqared per kiloyear]
                               Dm:0.2,                      # Fault Hillslope linear diffusivity [meters sqared per kiloyear]
                               ko:0.1,                      # value for the "erodability" of the landscape bedrock [kyr^-1]
                               km:0.2},                     # Fault value for the "erodability" of the landscape bedrock [kyr^-1]
                tectonics:    {uplift_rate:1.8,             # uplft rate of the landscape [meters per kiloyear]
                               vmax:6.0,                    # maximum off fault deformation rate [meters per kiloyear]
                               v_star:50.},                 # e-folding lengthscale for the deformation profile [meters]
                duration:     {total_time: 1500,            # Maximum time the simulation can run for [kyr]
                               shear_start: 1000,           # Time to start the off-fault deformation [kyr]
                               dt: 1.0,                     # time_step of the for loop [kyr]
                               dt_during_shear: 0.8},       # time steps during shear    
                fault:        {width:500,                   # width of the damage zone (meters)
                               fault_pos: 'midway',         # fault location in the y direction (meters) - or None, or otherwise specify y coordinate
                               slip_profile:'arctan',       # shape of the desired slip function (opt: arctan, boxcar, exp, none)
                               damage_prof:'exp'},          # shape of the desired damage pattern (opt: exp, boxcar, none)    
                save_opt:     {save_out: True,              # save output or not (opt: True, 'temp', False)
                               save_format: 'Default'}}
    
    return mdl_dict
    




################################
    

def default_lanscape_parameters():
    mdl_dict = {xmax:2000.,                 # length of the domain [meters]
                ymax:1000.,                 # width of the domain [meters]
                loopBuff = 0.5;              # added buffer to loop around kdc
    dxy = ymax/100               # pixel size as the length of one side [meters]
    
    
    # Erosion variables #
    #K = 0.08 # value for the "erodability" of the landscape bedrock [kyr^-1]
    m = 0.5                 # exponent on drainage area [non dimensional]
    n = 1                   # exponent on river slope [non dimensional]
    
    Do = 0.2 # Hillslope linear diffusivity [meters sqared per kiloyear]
    Dm = 0.2 # Fault Hillslope linear diffusivity [meters sqared per kiloyear]
    ko = 0.1 # value for the "erodability" of the landscape bedrock [kyr^-1]
    km = 0.2 # Fault value for the "erodability" of the landscape bedrock [kyr^-1]
    
    
    # Tectonic variables #
    uplift_rate = 1.8       # uplft rate of the landscape [meters per kiloyear]
    vmax        = 6.0       # maximum off fault deformation rate [meters per kiloyear]
    v_star      = 50.      # e-folding lengthscale for the deformation profile [meters]
    
    # Model miscellaneous variables ##
    total_time  = 1500      # Maximum time the simulation can run for [kyr]
    shear_start = 1000      # Time to start the off-fault deformation [kyr]
    dt          = 1.0       #time_step of the for loop [kyr]
    dt_during_shear = 0.8  #time steps during shear
    
    current_time= 0         # time tracking variable [kyr]
    i           = 0         # iteration tracker [integer]
    plot_num    = 2       # number of iterations to run before each new plot
    calculate_BR= False     # calculate the metric BR used in the main paper
    num_frames  = ((total_time-shear_start)/dt_during_shear)/plot_num +1
    
    # fault variables # 
    width       = 500        # width of the damage zone (meters)
    fault_pos   = ymax/2    # fault location in the y direction (meters)
    
    slip_profile= 'arctan' # shape of the desired slip function (opt: arctan, boxcar, exp, none)
    damage_prof = 'exp'    # shape of the desired damage pattern (opt: exp, boxcar, none)
    
    save_out    = True     # save output or not (opt: True, 'temp', False)
    
    # informative output file: e.g. arctan_exp_5m_fault
    d = datetime.datetime.today()
    dirname     = slip_profile + '_' + damage_prof + '_' + str(width) + 'm_fault_' + d.strftime('%d-%m-%Y')
    file_name   = slip_profile + '_' + damage_prof + '_' + str(width) + 'm_fault'
    
    return mdl_dict