#!/usr/bin/env python
# coding: utf-8

"""
Created on Tue Jun 21 15:14:15 2016

@author: harrison

Code used to perform the research described in Gray et al. (2017).
In this code, we use the Landlab modeling toolkit to simulates a landscape
undergoing continuum off-fault deformation.

The landscape is created by uplifting a uniform block 600 meters wide by 1000
meters long and then eroding the landscape using linear diffusion to simulate
hillslope sediment transport and a "fastscape" eroder to simulate sediment
transport by rivers.

The tectonic deformation is performed using a exponential shaped velocity
profile. This is performed in the code by "accumulating" tectonic displacement
until the offset is greater than a pixel. At this point, the row of pixels is
shifted over by 1 pixel and any pixels moved outside of the domain are
duplicated on the other end.

Running this code will generate the landscape and output a plot of the
landscape at regular time intervals. Each parameter in the code is commented
with its physical analog and units are given in square brackets. The main weird
unit is I have time units in kiloyears (kyr) instead of just years. This is
just because it removes a lot of excess zeros. It also feels like landlab runs
faster with less zeros in the numbers but I think that is psychosomatic.

Gray, H. J., Shobe, C. M., Hobley, D. E., Tucker, G. E., Duvall, A. R.,
Harbert, S. A., & Owen, L. A. (2017). Off-fault deformation rate along the
southern San Andreas fault at Mecca Hills, southern California, inferred from
landscape modeling of curved drainages. Geology.

EDIT: Feb 1 2019

@author: Kelian Dascher-Cousineau
This code adds a fault damage zone component to the model

@author: Kelian Dascher-Cousineau
Limited the size of the buffer to improve computation time

"""
################################################################################
## First, we initialize the code by importing python libraries. ###############
################################################################################

import numpy as np                                          # numerical python
import xarray as xr
import matplotlib.pyplot as plt                             # matplotlib plotting functions


from landlab import RasterModelGrid                         #landlab raster grid
from landlab.plot import imshow_grid                        #landlab plotting function
import holoviews as hv
hv.notebook_extension('matplotlib')

import dill                                                 # to save the workspace variables for reproducibility
import datetime
import os

from landlab.components import (LinearDiffuser,             #Hillslopes
                                FlowAccumulator,            # Flow accumulation
                                FastscapeEroder,            #river erosion
                                DepressionFinderAndRouter,  # lake filling
                                Lithology)                  # Diffirent lithologies (damage zone)

from landlab.testing.tools import cdtemp
from landlab.io.esri_ascii import write_esri_ascii          # export to ascii (to analyse in matlab)

from tqdm import tqdm                                       # progress bar

import csv                                                  # to record model log



# In[]
################################################################################
################# Second, let's define model inputs. ###########################
################################################################################

def default_landscape_parameters():
    mdl_dict ={'model_domain':{'xmax':          1500.,          # length of the domain [meters]
                               'ymax':          1000.,            # width of the domain [meters]
                               'loopBuff':      0.5,              # added buffer to loop around kdc
                               'Nxy':           100},             # number of nodes in the y direction
               'landscape':  {'m':              0.5,            # exponent on drainage area [non dimensional]
                               'n':             1,                # exponent on river slope [non dimensional]
                               'Do':            0.1,              # Hillslope linear diffusivity [meters sqared per kiloyear]
                               'Dm':            0.1,              # Fault Hillslope linear diffusivity [meters sqared per kiloyear]
                               'ko':            0.005,            # value for the "erodability" of the landscape bedrock [kyr^-1]
                               'km':            0.02},            # Fault value for the "erodability" of the landscape bedrock [kyr^-1]
               'tectonics':  {'uplift_rate':    1.8,              # uplft rate of the landscape [meters per kiloyear]
                               'boundaries':    [False,True,False,False],# boundaries (closed: true or false) - left, top, right, bottom
                               'track_uplift':  False},         # display the grown of the landscape in the initial preperation of the model

               'duration':   {'total_time':     3000,             # Maximum time the simulation can run for [kyr]
                               'shear_start':   2000,             # Time to start the off-fault deformation [kyr]
                               'dt':            1.0,              # time_step of the for loop [kyr]
                               'dt_during_shear':0.8},            # time steps during shear
               'fault_opt':  {'width':          100.,           # width of the damage zone (meters)
                               'fault_pos':     'midway',         # fault location in the y direction (meters) - or None, or otherwise specify y coordinate
                               'slip_profile':  'arctan',         # shape of the desired slip function (opt: arctan, boxcar, exp, none)
                               'localization':  1,                # degree of localization 1 -> same width as the damage zone, <<1 -> localized fault
                               'damage_prof':   'exp',            # shape of the desired damage pattern (opt: exp, boxcar, heaviside_up, heaviside_down, none)
                               'vmax':          6.0,              # maximum off fault deformation rate [meters per kiloyear]
                               'v_star':        50.},             # e-folding lengthscale for the deformation profile [meters],
               'save_opt':   {'save_out':       'temp',           # save output or not (opt: True, 'temp', False)
                               'save_format':   'Default',        # alt: a dict with 'Directory' and 'Filename' keys
                               'save_ascii':    False,
                               'out_fields':    ['topographic__elevation',
                                                 'drainage_area'], 
                               'append_to_log': True}}         
    return mdl_dict

################################################################################
###########################  we now define the model ###########################
################################################################################

def run_model(mdl_input = None):
    if mdl_input == None:
        mdl_input = default_landscape_parameters()

    # for ease, seperate the input into its main comonents
    domain      = mdl_input['model_domain']
    landscape   = mdl_input['landscape']
    tectonics   = mdl_input['tectonics']
    fault_opt   = mdl_input['fault_opt']
    duration    = mdl_input['duration']
    save_opt    = mdl_input['save_opt']

    # set the pointspacing
    dxy = domain['ymax']/domain['Nxy']

    # set the location of the fault
    if fault_opt['fault_pos'] == 'midway':
        fault_pos = domain['ymax']/2

    # time bookeeping
    current_time= 0         # time tracking variable [kyr]
    i           = 0         # iteration tracker [integer]
    plot_num    = 2         # number of iterations to run before each new plot
    #calculate_BR= False     # calculate the metric BR used in the main paper
    num_frames  = ((duration['total_time']-duration['shear_start'])/duration['dt_during_shear'])/plot_num +1

    # informative output file: e.g. arctan_exp_5m_fault
    if save_opt['save_format'] == 'Default':
        d = datetime.datetime.today()
        fn = '{slip}{localization:.2e}_{damage}{intensity:.2e}_fault'.format(
                slip            = fault_opt['slip_profile'],
                localization    = fault_opt['localization'],
                damage          = fault_opt['damage_prof'],
                intensity       = landscape['km'])
        
        save_opt['save_format'] = {'file_name' : fn,
                                   'dirname':    fn + d.strftime('%d-%m-%Y')}

    print('Starting model run: ' + save_opt['save_format']['file_name'])

    ###########################################################################
    ###########################################################################
    ## Define domain (e.g. number of columns and rows)
    ncols = int(domain['xmax']*(1.+2.*domain['loopBuff'])/dxy) # number of columns, tripled for looped boundaries kdc
    nrows = int(domain['ymax']/dxy)       # number of rows
    e1 = ncols*dxy * domain['loopBuff']/(2.*domain['loopBuff']+1.)      # edge1
    e2 = ncols*dxy - e1                             # edge2

    ## Build the landlab grid #
    rmg = RasterModelGrid((nrows,ncols),dxy) # build a landlab grid of nrows x ncols

    #set boundary conditions to have top closed and all others open
    rmg.set_closed_boundaries_at_grid_edges(tectonics['boundaries'][0],
                                            tectonics['boundaries'][1],
                                            tectonics['boundaries'][2],
                                            tectonics['boundaries'][3])

    # Add an elevation field + slope and noise to get the flow router started
    rmg.add_zeros('node','topographic__elevation')
    rmg['node']['topographic__elevation'] += (rmg.node_y*0.1 +
                                              np.random.rand(nrows*ncols))


    ################################################################################
    ## Fourth, the fun part, set up the off-fault deformation profile #############
    ################################################################################


    # define how a fault will work
    class fault:
        def __init__(self,rmg,fault_pos=None, fault_width=None):
            self.rmg        = rmg
            self.nrows      = rmg.shape[0]
            self.ncols      = rmg.shape[1]
            self.fault_pos  = fault_pos
            self.fault_width= fault_width


        def slip_profile(self,fault_prof,width=None):
            '''
            Define a slip profile relative to the dimensions of the model.
            '''

            yLocation = np.arange(nrows)*self.rmg.dx

            if width == None:
                width = self.fault_width

            if      fault_prof == 'arctan':
                profile = 0.5 + 1/np.pi * np.arctan((yLocation-self.fault_pos)*np.pi/width) # deal with magic number
            elif    fault_prof == 'heaviside':
                profile = np.heaviside(yLocation-self.fault_pos,1)
            elif    fault_prof == 'exp':
                profile = np.exp(-yLocation/(fault_opt['v_star']/dxy))
            else:
                raise('fault_opt must be one of: arctan, heavyside, exp')

            v_profile = profile * fault_opt['vmax']
            accum_disp= profile * self.rmg.dx

            return v_profile, accum_disp


        def lithology(self, damage_opt=None, width=None, ko=None, km=None, Do=None, Dm=None):


            if width == None:
                width = self.fault_width # set to the instantiated width

            if damage_opt == None:
                damage_func = lambda x: np.zeros(x.size)
                km=0; Dm=0

            # 2D profiles
            if damage_opt == 'boxcar':
                damage_func = lambda x: np.heaviside(x-fault_pos-width,1) - \
                                        np.heaviside(x-fault_pos+width,1)

            if damage_opt == 'heaviside_up':
                damage_func = lambda x: np.abs(np.heaviside(x-fault_pos,1)-1)

            if damage_opt == 'heaviside_down':
                damage_func = lambda x: np.heaviside(x-fault_pos,1)

            if damage_opt == 'exp':
                damage_func = lambda x: (width/(abs(x-fault_pos)+width))  # NEED TO DEFINE BUFF -> similar to c value in aftershocks

            def mk_dict(rmg,damage_func,ao,am):
                lith_ary = ao + (am-ao) * damage_func(rmg.node_y)
                damage_dict = dict(zip(rmg.node_y,lith_ary))
                return damage_dict

            attrs = {'K_sp':mk_dict(self.rmg,damage_func,ko,km),
                     'D':   mk_dict(self.rmg,damage_func,Do,Dm)}

            lith = Lithology(self.rmg,
                             self.rmg.ones().ravel().reshape(1,len(self.rmg.node_y)), # note that this is a dummy thickness
                             self.rmg.node_y.reshape(1,len(self.rmg.node_y)),
                             attrs,
                             rock_id = self.rmg.node_y)

            self.rmg['node']['topographic__elevation'] += 1.
            lith.run_one_step()
            #imshow_grid(self.rmg, 'K_sp', cmap='viridis', vmin=ko, vmax=km)
            return lith

    # define the position of the fault
    this_fault  = fault(rmg,fault_pos,fault_opt['width'])
    this_fault.lithology(damage_opt=fault_opt['damage_prof'], ko = landscape['ko'], km = landscape['km'],
                                                                            Do = landscape['Do'], Dm =landscape['Dm'])
    v_profile,accum_disp = this_fault.slip_profile(fault_opt['slip_profile'], fault_opt['localization']*fault_opt['width'])
    # This is an array for counting how many pixels need to be moved
    nshift = np.zeros(nrows)
    n_buff=0 # optional extra buffer zone incase you only want to move a subset.


    ################################################################################
    ## Next, we instantiate landlab components that will evolve the landscape #####
    ################################################################################

    fr = FlowAccumulator(rmg,
                         'topographic__elevation',
                         flow_director='FlowDirectorD8')                                                # standard D8 flow routing algorithm
    sp = FastscapeEroder(rmg, K_sp='K_sp',m_sp=landscape['m'],n_sp=landscape['n'],threshold_sp=0) # river eroder
    lin_diffuse = LinearDiffuser(rmg, linear_diffusivity='D')           #linear diffuser
    fill = DepressionFinderAndRouter(rmg)                               #lake filling algorithm

    ################################################################################
    ## Next, we instantiate an object to store the model output #####
    ################################################################################
    nts = int(num_frames)
    if not save_opt['save_out'] == False:
        # define the base coordinates of the grid
        ds = xr.Dataset(coords={'x': (('x'),                                                            # tuple of dimensions
                                      rmg.x_of_node.reshape(rmg.shape)[0,:],                                # 1-d array of coordinate data
                                      {'units' : 'meters'}),                                                # dictionary with data attributes
                                'y': (('y'),
                                      rmg.y_of_node.reshape(rmg.shape)[:, 1],
                                      {'units' : 'meters'}),
                                'time': (('time'),
                                         duration['dt']*np.arange(nts)/1e6,
                                         {'units': 'millions of years since model start',
                                          'standard_name' : 'time'})})
        
        # introduce the fields according the the fields specified in "save_opt"
        for of in save_opt['out_fields']:
            ds[of] = (('time', 'y', 'x'), 
                      np.empty((nts, rmg.shape[0], rmg.shape[1])),
                      {'units' : 'meters'})
    
    plotCounter = 0

    ################################################################################
    # Now that all parameters and landlab components are set, run the loop #######
    ################################################################################

    dt          = duration['dt'] # define the timestep
    mean_elev   = []
    time_array  = []
    
    with tqdm(total=duration['total_time'],position=0, leave=True) as pbar:
        while (current_time <= duration['total_time']):
            pbar.n = np.round(current_time)
            pbar.refresh()
    
            ## Looped boundary conditions ##
    
            # Because the landlab flow router isn't currently set up to use looped
            # boundary conditions, I simulated them here by duplicating the landscape
            # to the left and right such that the flow accumulator would 'see' the
            # the appropriate amount of upstream drainage area.
    
            rmg['node']['topographic__elevation'][(rmg.node_x<e1)] = (
             rmg['node']['topographic__elevation'][(rmg.node_x>=(e2-e1)) &
             (rmg.node_x<e2)])
    
            rmg['node']['topographic__elevation'][(rmg.node_x>=e2)] = (
              rmg['node']['topographic__elevation'][(rmg.node_x>=e1) &
              (rmg.node_x<(2*e1))])
    
            ## Tectonic off-fault deformation ##
    
            # To simulate off fault lateral displacement/deformation, we apply a
            # lateral velocity profile. This is done by taking the landlab elevations
            # and shifting their coordinates.
            if (current_time>duration['shear_start']):
    
              dt        = duration['dt_during_shear'] # First set a lower timestep to keep it stable
    
              # Take the landlab grid elevations and reshape into a box nrows x ncols
              elev      = rmg['node'][ 'topographic__elevation']
              elev_box  = np.reshape(elev, [nrows,ncols])
    
              # Calculate the offset that has accumulated over a timestep
              accum_disp+= v_profile*dt;
    
              # now scan up the landscape row by row looking for offset
              for r in range(nrows):
    
                # check if the accumulated offset for a row is larger than a pixel
                if accum_disp[r] >= dxy:
    
                  # if so, count the number of in the row pixels to be moved
                  nshift[r] = int(np.floor(accum_disp[r]/dxy))
    
                  # copy which pixels will be moved off the grid by displacement
                  temp = elev_box[r,n_buff:int(nshift[r])+n_buff]
    
                  # move the row over by the number of pixels of accumulated offset
                  elev_box[r,n_buff:((ncols-n_buff)-int(nshift[r]))] = elev_box[r,int(nshift[r])+n_buff:ncols-n_buff]
    
                  # replace the values on the right side by the ones from the left
                  elev_box[r,((ncols)-int(nshift[r])):ncols] = temp
    
                  # last, subtract the offset pixels from the accumulated displacement
                  accum_disp[r] -= dxy
    
              #This section is if you select a middle section to be moved independently
              elev_box[:,(ncols-n_buff):ncols]=elev_box[:,n_buff:(2*n_buff)]
              elev_box[:,0:n_buff] = elev_box[:,ncols-2*n_buff:ncols-n_buff]
    
              # Finally, reshape the elevation box into an array and feed to landlab
              elev_new = np.reshape(elev_box, nrows*ncols)
              rmg['node']['topographic__elevation'] = elev_new
    
            # record the evolution of the mean elevation
            mean_elev.append(np.mean(rmg['node']['topographic__elevation']))
            time_array.append(current_time);
    
            ## Landscape Evolution ##
    
            # Now that we have performed the tectonic deformation, lets apply our
            # landscape evolution and watch the landscape change as a result.
    
            # Uplift the landscape
            rmg['node']['topographic__elevation'][rmg.core_nodes] += tectonics['uplift_rate']*dt
    
            # set the lower boundary as fixed elevation
            #rmg['node']['topographic__elevation'][rmg.node_y==0] = 0
            #rmg['node']['topographic__elevation'][rmg.node_y==max(rmg.node_y)] = 0
    
            # Diffuse the landscape simulating hillslope sediment transport
            lin_diffuse.run_one_step(dt)
    
            # Accumulate and route flow, fill any lakes, and erode under the rivers
            fr.run_one_step() # route flow
            DepressionFinderAndRouter.map_depressions(fill) # fill lakes
            sp.run_one_step(dt) # fastscape stream power eroder
    
            ## Plotting ##
            # plot output at each [plot_num] iteration once we start lateral advection
            if i % plot_num == 0 and current_time>duration['shear_start']:
              
              if not save_opt['save_out'] == False:
                  for of in save_opt['out_fields']:
                    ds[of][plotCounter,:,:] = rmg['node'][of].reshape(rmg.shape)
              plotCounter += 1
            current_time += dt # update time tracker
            i += 1 # update iteration variable
        print('number of frames:', plotCounter)
        print('For loop complete')
    
    
    ##########################################################################
    ### We do some bookkeeping with the result ###############################
    ##########################################################################
#    plt.figure()
#    imshow_grid(rmg,'topographic__elevation',show_elements=False)
#    # tweek edges of domain so that they match the buffer kdc
#    plt.xlim((e1,e2))
#    plt.show()
#    plt.clf

    ##########################################################################
    ## plot the evolution of the mean elevation of the model:
    if tectonics['track_uplift']:
        plt.figure()
        plt.plot(time_array,mean_elev, label='Mean elevation')
        plt.axvline(duration['shear_start'],color='r',label='Start lateral advection')
        plt.xlabel('Iteration number')
        plt.ylabel('Mean elevation')
        plt.legend()
        plt.show()

    # small function to save a specific field as a gif
    ## plot the output into a gif
    get_ipython().run_line_magic('opts', "Image style(interpolation='bilinear', cmap='viridis') plot[colorbar=True]")
    get_ipython().run_line_magic('output', 'size=100')
    

    fn = save_opt['save_format']['file_name']
    dn = save_opt['save_format']['dirname']
    if save_opt['save_out'] == True:
        os.mkdir(dn)
        
        for of in save_opt['out_fields']:
            hvds_topo = hv.Dataset(ds[of])
            topo = hvds_topo.to(hv.Image, ['x', 'y'])
            topo.opts(colorbar=True, fig_size=200, xlim=(e1, e2))
            hv.save(topo, os.path.join(dn,fn + of + '.gif'))
        dill.dump_session(os.path.join(dn,fn + '.pkl'))
        # and to load the session again:
        # dill.load_session(filename)
    if save_opt['save_ascii'] == True:
        write_esri_ascii(os.path.join(dn,save_opt['save_format']['file_name'] + '.asc'), rmg,names='topographic__elevation')

    if save_opt['save_out'] == 'temp':
        hv.save(topo,'temp_output.gif')
        
    if save_opt['append_to_log'] == True:
        def NestedDictValues(d):
            for v in d.values():
                if isinstance(v, dict):
                    yield from NestedDictValues(v)
                else:
                    yield v
        with open('Model_log.csv', mode='a') as fd:
            writer = csv.writer(fd)            
            writer.writerow(NestedDictValues(mdl_input))
                
            
        
    return rmg

################################################################################
## Now we have fun! - play around with changing default values  ################
################################################################################

# below are various models to play around with. The first block is an example
# for running models over a grid of possible values of distributed deformation
# and damage intensities

# In[]

# with the code in the format you can easily play around with different input stuctures:

koAry = [0.005,    0.0075,   0.01]
locAry= [1/10**10, 1/10**2,1/10]
defaultDict = default_landscape_parameters()
defaultDict['duration']['shear_start'] = 4000
defaultDict['duration']['total_time'] = 5000
defaultDict['model_domain']['Nxy'] = 140
defaultDict['save_opt']['save_out'] = False

def plot_matrix(defaultDict,koAry,locAry,plot_field):
    # Illuminate the scene from the northwest
    rmg_arr = []
    for idxDam, iDamage in enumerate(koAry):
        for idxDef, iDef in enumerate(locAry):
            new_landscape_parameter = defaultDict
            new_landscape_parameter['landscape']['ko'] = iDamage
            new_landscape_parameter['fault_opt']['localization'] = iDef
            rmg = run_model(mdl_input = new_landscape_parameter)
            rmg_arr.append(rmg)
    
    for iField in plot_field:
        fig, ax = plt.subplots(len(koAry),len(locAry),figsize=[6,4.5])
        plot_count = 0
        for idxDam, iDamage in enumerate(koAry):
            for idxDef, iDef in enumerate(locAry):
                irmg      = rmg_arr[plot_count]; plot_count+=1
                topo_elev = irmg.node_vector_to_raster(irmg['node'][iField], True)
                ax[idxDam,idxDef].imshow(topo_elev)
                Nx = topo_elev.shape[1]
                buff = new_landscape_parameter['model_domain']['loopBuff']
                Nbuff = Nx*buff/(1+2*buff)
                ax[idxDam,idxDef,].set_xlim(Nbuff, Nx - Nbuff)
                if idxDam == 0:
                    tle = '${localization:.2e}$'.format(localization = iDef)
                    ax[idxDam,idxDef].title.set_text(tle)
        
                if idxDef == 0:
                    tle = '${damage:.2e}$'.format(damage = iDamage)
                    ax[idxDam,idxDef].set_ylabel(tle)
        
        for iax in ax.flat:
            iax.set_xticks([])
            iax.set_yticks([])
        
        
        fig
        fig.savefig('grid.png', dpi=300)
        
plot_matrix(defaultDict,koAry,locAry,['topographic__elevation','drainage_area'])


# In[]

#defaultDict = default_landscape_parameters()
#new_landscape_parameter['landscape']['km'] = 0.1

# In[]
#new_landscape_parameter = default_landscape_parameters()
#new_landscape_parameter['landscape']['km'] = 0.05
#new_landscape_parameter['landscape']['ko'] = 0.005
#new_landscape_parameter['landscape']['Do'] = 5
#new_landscape_parameter['landscape']['Dm'] = 5 # 50 * 10^-4 m^2/yr as per Estimation of the diffusion coefficient from slopeconvexity at Marin County, California
#new_landscape_parameter['duration']['shear_start'] = 5000
#new_landscape_parameter['duration']['total_time'] = 6000
#new_landscape_parameter['save_opt']['save_out'] = True
#new_landscape_parameter['save_opt']['save_ascii'] = True
#new_landscape_parameter['model_domain']['Nxy'] = 160
#new_landscape_parameter['model_domain']['xmax'] = 6000
#new_landscape_parameter['model_domain']['ymax'] = 4000
#new_landscape_parameter['fault_opt']['localization'] = 0.1
#rmg, ds = run_model(new_landscape_parameter)


# In[] Quick and dirty runs
#new_landscape_parameter = default_landscape_parameters()
#new_landscape_parameter['fault_opt']['localization'] = 0.1
#new_landscape_parameter['model_domain']['Nxy'] = 100
#new_landscape_parameter['duration']['shear_start'] = 1000
#new_landscape_parameter['duration']['total_time'] = 1300
#new_landscape_parameter['save_opt']['save_ascii'] = True
#new_landscape_parameter['save_opt']['save_out'] = True
#run_model(new_landscape_parameter)

# In[] Quick and dirty runs
# new_landscape_parameter = default_landscape_parameters()
# new_landscape_parameter['landscape']['km']          = 0.03
# new_landscape_parameter['landscape']['ko']          = 0.003
# new_landscape_parameter['fault_opt']['damage_prof'] = 'heaviside_down';
# new_landscape_parameter['fault_opt']['localization']= 0.01
# new_landscape_parameter['model_domain']['Nxy']      = 150
# new_landscape_parameter['duration']['shear_start']  = 5000
# new_landscape_parameter['duration']['total_time']   = 6000
# new_landscape_parameter['save_opt']['save_out']     = True
# new_landscape_parameter['tectonics']['track_uplift']= True
# rmg = run_model(new_landscape_parameter)
