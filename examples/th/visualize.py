#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log
sys.path.append('/root/vol/octnet-completion/py/')
import pyoctnet

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)+1e-6)

def grid_wireframe(fig, ax, grid, f_name):  
  voxels = [np.zeros((grid.vx_depth()//2**i,grid.vx_height()//2**i,grid.vx_width()//2**i))
     for i in range(0,4)]

  levels = set()
  colors = np.zeros((grid.vx_depth(),grid.vx_height(),grid.vx_width(),3))
  for (leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, level) in pyoctnet.leaf_iterator(grid, leafs_only=False):
    x = gw * 8 + bw
    y = gh * 8 + bh
    z = gd * 8 + bd
    ind = 3 - level
    width = 2**ind
  
    z_dim = voxels[ind].shape[2] - 1
    # print(z_dim - z//width)
    if ind > 0 and not leaf:
      voxels[ind][x//width,z_dim-z//width,y//width] = 1

    levels.add(level)

    if ind == 0 and leaf:
      grid_data = grid.get_grid_data()
      data_idx = grid.data_idx(grid_idx, bit_idx)
      sdf = (grid_data[data_idx + 0])
      if abs(sdf) < 3 and abs(sdf) > 1e-5:
        voxels[ind][x//width,z_dim-z//width,y//width] = 1
        # c = (1,0,0)
        # if sdf < 0:
        #   c = (0,0,0)
        colors[x//width,z_dim-z//width,y//width] = [abs(sdf)*0.5, abs(sdf), 3-abs(sdf)]

  # print(levels)
  color = [(0.87,0.22,0.37,0.3), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1)]
  colors = NormalizeData(colors)
  for i,voxel in enumerate(voxels):
    x,y,z = np.indices(np.array(voxel.shape) + 1)*2**i
    ax.voxels(x,y,z, voxel, edgecolor=color[i], facecolors=(0,0,0,0) if i!=0 else colors)

  fig.savefig("junk/%s.jpg" % f_name, dpi=300)

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-45)
plt.tight_layout()
ax.set_xlim(0,32)
ax.set_ylim(0,32)
ax.set_zlim(0,32)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
# ax._axis3don = False
# ax.dist = 10


for i in ['output', 'input', 'target']:
  grid = pyoctnet.Octree.create_from_bin(bytes('junk/%s.oc' %i, encoding="ascii"))
  grid_wireframe(fig, ax, grid, i)
  print(grid.mem_using())
