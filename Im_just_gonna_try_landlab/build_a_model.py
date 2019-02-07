import numpy as np
from pylab import show, figure, plot
import time
from landlab import RasterModelGrid
from landlab.components.flow_routing import FlowRouter
from landlab.plot.imshow import imshow_node_grid

mg = RasterModelGrid((10, 10), 1.)  # make a grid
z = np.zeros(100, dtype=float)  # make a flat surface, elev 0
# or…
z = mg.node_y*0.01  # a flat surface dipping shallowly south
# add a little noise to the surface:
z += np.random.rand(100.)/10000.
# create the field:
mg.add_field('node', 'topographic__elevation', z, units='m')

# make the boundaries to the model
mg.set_fixed_value_boundaries_at_grid_edges(False, True, False, True)
mg.set_closed_boundaries_at_grid_edges(True, False, True, False)
