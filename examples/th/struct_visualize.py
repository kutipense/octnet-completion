#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
sys.path.append('/root/vol/octnet-completion/py/')
sys.path.append('C:\\Users\\kutay\\ml43d\\octnet\\')
import pyoctnet

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)+1e-6)

def grid_wireframe(ax, grid, n=0):  
  voxels = [np.zeros((grid.vx_depth()//2**i,grid.vx_height()//2**i,grid.vx_width()//2**i))
     for i in range(0,4)]

  levels = set()
  colors = [np.zeros((grid.vx_depth()//2**i,grid.vx_height()//2**i,grid.vx_width()//2**i, 3))
     for i in range(0,4)]
  for (leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, level) in pyoctnet.leaf_iterator(grid, n=n, leafs_only=False):
    x = gw * 8 + bw
    y = gh * 8 + bh
    z = gd * 8 + bd
    ind = 3 - level
    width = 2**ind
  
    z_dim = voxels[ind].shape[2] - 1

    levels.add(level)

    if ind == 0 or (ind > 0 and leaf): # and leaf:
      grid_data = grid.get_grid_data()
      data_idx = grid.data_idx(grid_idx, bit_idx)
      sdf = (grid_data[data_idx + 0])
      if abs(sdf) < 3 and abs(sdf) >= 0.5:
        voxels[ind][x//width,z_dim-z//width,y//width] = 1
        colors[ind][x//width,z_dim-z//width,y//width] = [0, abs(sdf), 1-abs(sdf)]

  # print(levels)
  color = [(0.87,0.22,0.37,0.3), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1)]
  colors = [NormalizeData(i) for i in colors]
  for i,voxel in enumerate(voxels):
    x,y,z = np.indices(np.array(voxel.shape) + 1)*2**i
    # if i==2:
    ax.voxels(x,y,z, voxel, edgecolor=color[i], facecolors=colors[i] if i==0 else (0,0,0,0))


fig, axs = plt.subplots(2, 3, subplot_kw=dict(projection='3d'))
fig.set_size_inches(30, 20)
for r in axs:
  for ax in r:
    ax.view_init(elev=30, azim=60)
    plt.tight_layout()
    ax.set_xlim(0,32)
    ax.set_ylim(0,32)
    ax.set_zlim(0,32)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    # ax._axis3don = False
    ax.dist = 10

ls = LightSource(azdeg=225.0, altdeg=45.0)
vxs = [pyoctnet.Octree.create_from_bin(bytes('struct_junk/%s.oc' %fname, encoding="ascii")) 
    for i, fname in enumerate(['input', 'output1', 'target'])] # , 'output2', 'output3', 'output4', 



for i, vx in enumerate(vxs):
  grid_wireframe(axs[i//3][i%3], vx, n=0)
  # fig.savefig("struct_junk/img/output%d.jpg" %1, dpi=300)
plt.show()