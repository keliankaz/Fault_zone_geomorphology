#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

## First, we initialize the code by importing python libraries. ###############

import numpy as np # numerical python
import xarray as xr
import matplotlib.pyplot as plt # matplotlib plotting functions

from landlab import RasterModelGrid #landlab raster grid
from landlab.plot import imshow_grid #landlab plotting function
import holoviews as hv
hv.notebook_extension('matplotlib')

from landlab.components import (LinearDiffuser, #Hillslopes
                                FlowRouter, # Flow accumulation
                                FastscapeEroder, #river erosion
                                DepressionFinderAndRouter, # lake filling
                                Lithology) # Diffirent lithologies (damage zone)




################################################################################
## Second, let's build the landlab grid we will build our landscape on. #######
################################################################################

# Set Domain size and parameters #

xmax = 200.                 # length of the domain [meters]
ymax = 100.                 # width of the domain [meters]
dxy = 1.0                   # pixel size as the length of one side [meters]
loopBuff = 0.2;             # added buffer to loop around kdc
ncols = int(xmax*(1.+2.*loopBuff)/dxy) # number of columns, tripled for looped boundaries kdc
# ncols = int(3*xmax/dxy)   # number of columns, tripled for looped boundaries kdc
nrows = int(ymax/dxy)       # number of rows
e1 = ncols*dxy * loopBuff/(2.*loopBuff+1.)      # edge1
e2 = ncols*dxy - e1                                             # edge2

# Build the landlab grid #
rmg = RasterModelGrid((nrows,ncols),dxy) # build a landlab grid of nrows x ncols

#set boundary conditions to have top closed and all others open
rmg.set_closed_boundaries_at_grid_edges(False,True,False,False)

# Add an elevation field + slope and noise to get the flow router started
rmg.add_zeros('node','topographic__elevation')
rmg['node']['topographic__elevation'] += (rmg.node_y*0.1 +
                                          np.random.rand(nrows*ncols))


## Adapted script so that there is a band of weaker rock along the shear zone
# Possible improvements -
# Make the initialization a bit less arbritrary
# Not sure that the lithology elements get tracked with the nodes, this does not matter in the strike slip case but may matter in other cases

z = rmg['node']['topographic__elevation']

# The variable lithology is done the only way I know how (most likely not the best way)
attrs = {'K_sp':  {1: 0.05,
                   2: 0.05},
         'D':     {1: 0.02,
                   2: 0.02}}
# Dummy thickness
thickness = [1,1]

# instantiate the Lithology component
lith = Lithology(rmg, thickness, [1,2], attrs)

# put in the damage zone a third of the way in the model
spatially_variable_rock_id = rmg.ones('node')

# Comment/Uncomment below to introduce a damage zone kdc!
spatially_variable_rock_id[(rmg.y_of_node>2*ymax/5) & (rmg.y_of_node<3*ymax/5)] = 2

# grow the topography up (this is clearly dumb)
z += 1.
dz_ad = 0.
lith.run_one_step(dz_advection = dz_ad, rock_id = spatially_variable_rock_id)
imshow_grid(rmg, 'rock_type__id', cmap='viridis', vmin=0, vmax=3)

#  since we will not update z after this, the erodability parameters of the rock should not change
#  ...at least that is the idea

################################################################################
## Third, we now set parameters to control and build our landscape ############
################################################################################

# Erosion variables #
#K = 0.08 # value for the "erodability" of the landscape bedrock [kyr^-1]
m = 0.5                 # exponent on drainage area [non dimensional]
n = 1                   # exponent on river slope [non dimensional]
#D = 0.02 # Hillslope linear diffusivity [meters sqared per kiloyear]

# Tectonic variables #
uplift_rate = 1.8       # uplft rate of the landscape [meters per kiloyear]
vmax        = 6.0       # maximum off fault deformation rate [meters per kiloyear]
v_star      = 50.      # e-folding lengthscale for the deformation profile [meters]

# Model miscellaneous variables ##
total_time  = 8000      # Maximum time the simulation can run for [kyr]
shear_start = 2000      # Time to start the off-fault deformation [kyr]
dt          = 1.0       #time_step of the for loop [kyr]
dt_during_shear = 0.05  #time steps during shear

current_time= 0         # time tracking variable [kyr]
i           = 0         # iteration tracker [integer]
plot_num    = 200       # number of iterations to run before each new plot
calculate_BR= False     # calculate the metric BR used in the main paper
num_frames  = ((total_time-shear_start)/dt_during_shear)/plot_num +1


################################################################################
## Fourth, the fun part, set up the off-fault deformation profile #############
################################################################################

# define the position of the fault
fault_pos = nrows/2

# decide what kind of fault to make options include heavyside, exponential decay (half shear zone), arctan
faultOpt  = 'heaviside'
yLocation = np.arange(0.,nrows)
faultStretch = 1/5 # more distributed deformation (>1.) vs localize def (<<1)

# different fault setups
if      faultOpt == 'arctan':
    profile = np.arctan(((yLocation-fault_pos)/nrows - 0.5)*np.pi/faultStretch) # deal with magic number
elif    faultOpt == 'heaviside':
    profile = np.heaviside(yLocation-fault_pos,1)
elif    faultOpt == 'exp':
    profile = np.exp(-yLocation/(v_star/dxy))


# make velocity profile and because the grid is discretized into pixels, we need to count how much
# deformation has occurred over a timestep and move a pixel after the
# accumulated deformation is larger than than the pixel length
v_profile = profile * vmax
accum_disp= profile * float(dxy)

# This is an array for counting how many pixels need to be moved
nshift = np.zeros(np.size(yLocation))
n_buff=0 # optional extra buffer zone incase you only want to move a subset.



################################################################################
## Last, we instantiate landlab components that will evolve the landscape #####
################################################################################

fr = FlowRouter(rmg)                                                # standard D8 flow routing algorithm
sp = FastscapeEroder(rmg, K_sp='K_sp',m_sp=m,n_sp=n,threshold_sp=0) # river eroder
lin_diffuse = LinearDiffuser(rmg, linear_diffusivity='D')           #linear diffuser
fill = DepressionFinderAndRouter(rmg)                               #lake filling algorithm

nts = int(num_frames)
ds = xr.Dataset(data_vars={'topographic__elevation' : (('time', 'y', 'x'),                      # tuple of dimensions
                                                       np.empty((nts, rmg.shape[0], rmg.shape[1])), # n-d array of data
                                                      {'units' : 'meters'})},                       # dictionary with data attributes
                coords={'x': (('x'),                                                            # tuple of dimensions
                              rmg.x_of_node.reshape(rmg.shape)[0,:],                                # 1-d array of coordinate data
                              {'units' : 'meters'}),                                                # dictionary with data attributes
                        'y': (('y'),
                              rmg.y_of_node.reshape(rmg.shape)[:, 1],
                              {'units' : 'meters'}),
                        'time': (('time'),
                                 dt*np.arange(nts)/1e6,
                                 {'units': 'millions of years since model start',
                                  'standard_name' : 'time'})})
out_fields = ['topographic__elevation']
plotCounter = 0

# Now that all parameters and landlab components are set, run the loop #######

print('start for loop')

while (current_time <= total_time):

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

    if (current_time>shear_start):

      dt = dt_during_shear # First set a lower timestep to keep it stable

      # Take the landlab grid elevations and reshape into a box nrows x ncols
      elev = rmg['node'][ 'topographic__elevation']
      elev_box = np.reshape(elev, [nrows,ncols])

      # Calculate the offset that has accumulated over a timestep
      accum_disp += v_profile*dt;

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


    ## Landscape Evolution ##

    # Now that we have performed the tectonic deformation, lets apply our
    # landscape evolution and watch the landscape change as a result.

    # Uplift the landscape
    rmg['node']['topographic__elevation'][rmg.core_nodes] += uplift_rate*dt

    # set the lower boundary as fixed elevation
    rmg['node']['topographic__elevation'][rmg.node_y==0] = 0

    # Diffuse the landscape simulating hillslope sediment transport
    lin_diffuse.run_one_step(dt)

    # Accumulate and route flow, fill any lakes, and erode under the rivers
    fr.run_one_step() # route flow
    DepressionFinderAndRouter.map_depressions(fill) # fill lakes
    sp.run_one_step(dt) # fastscape stream power eroder


    ## Calculate the geomorphic metric ##

    # In the paper, we use a geomorphic metric, BR, to quantify the
    # reorientation of the channels as time goes on. The code to calculate this
    # value is below but turned off as it can slow the model. Set the
    # 'calculate_BR' variable to 'True' if you want to calculate it.

    if calculate_BR:

      aspects = rmg.calc_aspect_at_node() # measure pixel aspect

      # classify and count the number of pixels with certain directions
      asp_0_45    = float(np.logical_and(aspects>=0,aspects<=45).sum())
      asp_45_135  = float(np.logical_and(aspects>45,aspects<=135).sum())
      asp_135_225 = float(np.logical_and(aspects>135,aspects<=225).sum())
      asp_225_315 = float(np.logical_and(aspects>225,aspects<=315).sum())
      asp_315_360 = float(np.logical_and(aspects>315,aspects<=360).sum())

      # Calculate BR from Gray et al. (2017)
      BR = (asp_0_45 + asp_315_360 + asp_135_225)/(asp_45_135 + asp_225_315)

    ## Plotting ##

    if i % plot_num == 0 and current_time>shear_start:

      # Use landlab plotting function to plot the landscape
      #imshow_grid(rmg,'topographic__elevation',show_elements=False)
      #plt.xlim((rmg.shape[1]*rmg.dx/3,rmg.shape[1]*rmg.dx*2/3))
      #plt.show()
      #plt.clf
      print('Current time = ' + str(current_time)) # show current time
      for of in out_fields:
        ds[of][plotCounter,:,:] = rmg['node'][of].reshape(rmg.shape)
      plotCounter += 1
    current_time += dt # update time tracker


    i += 1 # update iteration variable

print(plotCounter)
print('For loop complete')


imshow_grid(rmg,'topographic__elevation',show_elements=False)
# tweek edges of domain so that they match the buffer kdc
plt.xlim((e1,e2))
plt.show()
plt.clf

################################################################################
## plot the output into a gif
hvds_topo = hv.Dataset(ds.topographic__elevation)
get_ipython().run_line_magic('opts', "Image style(interpolation='bilinear', cmap='viridis') plot[colorbar=True]")
get_ipython().run_line_magic('output', 'size=700')
topo = hvds_topo.to(hv.Image, ['x', 'y'])
# topo

hv.save(topo,'temp_output.gif')
