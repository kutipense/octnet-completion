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
import pyoctnet

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)+1e-6)

def grid_wireframe(ax, grid):  
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

    if ind == 0: # and leaf:
      grid_data = grid.get_grid_data()
      data_idx = grid.data_idx(grid_idx, bit_idx)
      sdf = (grid_data[data_idx + 0])
      if abs(sdf) < 3 and abs(sdf) > 1e-5:
        voxels[ind][x//width,z_dim-z//width,y//width] = 1
        colors[x//width,z_dim-z//width,y//width] = [abs(sdf)*0.5, abs(sdf), 3-abs(sdf)]

  # print(levels)
  color = [(0.87,0.22,0.37,0.3), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1), (0.87,0.22,0.37,0.1)]
  colors = NormalizeData(colors)
  for i,voxel in enumerate(voxels):
    x,y,z = np.indices(np.array(voxel.shape) + 1)*2**i
    ax.voxels(x,y,z, voxel, edgecolor=color[i], facecolors=(0,0,0,0) if i!=0 else colors)


fig, axs = plt.subplots(1, 4, subplot_kw=dict(projection='3d'))
fig.set_size_inches(30, 10)
for ax in axs:
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
vxs = [0,0,0]
for i, fname in enumerate(['input', 'output', 'target']):
  grid = pyoctnet.Octree.create_from_bin(bytes('junk/%s.oc' %fname, encoding="ascii"))
  vxs[i] = grid.to_cdhw()
  vxs[i][vxs[i]==0] = 3
  vxs[i].dump(fname+".vx")

for k in range(1):    
  fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection='3d'))
  fig.set_size_inches(30, 10)
  for ax in axs:
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

  for i, _vx in enumerate(vxs):
    vx = _vx#[k]
    print(vx.shape)
    if i == 0:
      vx = vx[0,:,:,:]
    # print(vx.shape)
    vx = np.transpose(vx, axes=(0,2,1))
    # vx[:,:]
    verts, faces, normals, values = measure.marching_cubes(vx, 1.0)
    ax = axs[i]
    
    mesh = Poly3DCollection(verts[faces], linewidths=0, alpha=1.0)
    mesh.set_edgecolor((0,0,0,0))
    
    normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])
    min = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
    max = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
    diff = max-min
    newMin = 0.3
    newMax = 0.95
    newdiff = newMax-newMin
    colourRGB = np.array((245/255., 245/255., 220/255., 1.0))
    rgbNew = np.array([colourRGB*(newMin + newdiff*((shade-min)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])
    mesh.set_facecolor(rgbNew)
    ax.add_collection3d(mesh)

    # ax.voxels(vx, edgecolor=((245/255., 245/255., 220/255., 0.3)), facecolors=colourRGB)
    # grid_wireframe(axs[i],vx)

  print(k)
  fig.savefig("junk/output%d.jpg" %k, dpi=300)  
    # print(grid.mem_using())
