import os
import numpy as np
import xarray as xr
import dask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import holoviews as hv
hv.notebook_extension('matplotlib')

from landlab import RasterModelGrid
from landlab.components import FlowAccumulator, FastscapeEroder, LinearDiffuser, Lithology, LithoLayers
from landlab.plot import imshow_grid


# create the topography grid
xDim = 10
yDim = 15
dx   = 1
mg = RasterModelGrid((xDim, yDim), dx)
z = mg.add_zeros('node', 'topographic__elevation')

attrs = {'K_sp':  {1: 0.001,
                   2: 0.0001},
         'D':     {1: 0.01,
                   2: 0.001}}

thickness = [10,10]

lith = Lithology(mg, thickness, [1,2], attrs)

spatially_variable_rock_id = mg.ones('node')
spatially_variable_rock_id[(mg.x_of_node>xDim/3) & (mg.x_of_node<2*xDim/3)] = 2

z += 100.
dz_ad = 0.
lith.run_one_step(dz_advection = dz_ad, rock_id=spatially_variable_rock_id)
imshow_grid(mg, 'rock_type__id', cmap='viridis', vmin=0, vmax=3)

imshow_grid(mg, 'rock_type__id', cmap='viridis', vmin=0, vmax=3)
