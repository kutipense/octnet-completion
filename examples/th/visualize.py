#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('/root/vol/octnet-completion/py/')
import pyoctnet

def tikz_cube(f, x0,y0,z0, x1,y1,z1, fmt_str):
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x0,y0,z0, x0,y1,z0, x1,y1,z0, x1,y0,z0))
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x0,y0,z0, x0,y0,z1, x0,y1,z1, x0,y1,z0))
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x0,y1,z0, x1,y1,z0, x1,y1,z1, x0,y1,z1))
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x0,y0,z0, x1,y0,z0, x1,y0,z1, x0,y0,z1))
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x0,y0,z1, x0,y1,z1, x1,y1,z1, x1,y0,z1))
  f.write('\\draw[%s] (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- (%d,%d,%d) -- cycle;\n' % (fmt_str, x1,y0,z0, x1,y0,z1, x1,y1,z1, x1,y1,z0))

def tikz_grid_solid(f, grid, vis_grid_idx=None, color_from_data=False):
  n_blocks = grid.num_blocks() if vis_grid_idx is None else 1

  f.write("\\documentclass{minimal}\n")
  f.write("\\usepackage{xcolor}\n")
  f.write("\\usepackage{tikz,tikz-3dplot}\n")
  f.write("\\begin{document}\n")
  f.write("\\tdplotsetmaincoords{50}{130}\n")
  f.write("\\begin{tikzpicture}[scale=0.5, tdplot_main_coords]\n")

  for (leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, level) in pyoctnet.leaf_iterator(grid, leafs_only=False):
    if color_from_data:
      if leaf:
        grid_data = grid.get_grid_data()
        data_idx = grid.data_idx(grid_idx, bit_idx)
        color = [grid_data[data_idx + 0], grid_data[data_idx + 1], grid_data[data_idx + 2]]
      else:
        color = [0,0,0]
    else:
      cm = plt.cm.get_cmap('viridis')
      color = cm(grid_idx * 1.0 / n_blocks)
      color = [int(255*color[0]), int(255*color[1]), int(255*color[2])]

    if vis_grid_idx is not None and vis_grid_idx != grid_idx:
      continue

    x = gw * 8 + bw
    y = gh * 8 + bh
    z = gd * 8 + bd
    width = 2**(3 - level)

    x0 = x
    y0 = y
    z0 = z
    x1 = x + width
    y1 = y + width
    z1 = z + width

    if leaf or level == 0:
      color_str = 'color_%d' % grid_idx
      f.write('\\definecolor{%s}{RGB}{%d,%d,%d}\n' % (color_str, color[0], color[1], color[2]))

      if leaf:
        tikz_cube(f, x0,y0,z0, x1,y1,z1, 'very thin, gray, fill=%s,fill opacity=1.0' % (color_str))
      else:
        tikz_cube(f, x0,y0,z0, x1,y1,z1, 'very thick,black')
  f.write("\\end{tikzpicture}\n")
  f.write("\\end{document}\n")

def tikz_grid_wireframe(f, grid, vis_grid_idx=None, color_from_data=False):
  n_blocks = grid.num_blocks() if vis_grid_idx is None else 1
  print(n_blocks)

  f.write("\\documentclass{minimal}\n")
  f.write("\\usepackage{xcolor}\n")
  f.write("\\usepackage{tikz,tikz-3dplot}\n")
  f.write("\\begin{document}\n")
  f.write("\\tdplotsetmaincoords{50}{130}\n")
  f.write("\\begin{tikzpicture}[scale=0.5, tdplot_main_coords]\n")

  for (leaf, grid_idx, bit_idx, gd,gh,gw, bd,bh,bw, level) in pyoctnet.leaf_iterator(grid, leafs_only=False):
    if color_from_data:
      if leaf:
        grid_data = grid.get_grid_data()
        data_idx = grid.data_idx(grid_idx, bit_idx)
        color = [grid_data[data_idx + 0], grid_data[data_idx + 1], grid_data[data_idx + 2]]
      else:
        color = [0,0,0]
    else:
      cm = plt.cm.get_cmap('viridis')
      color = cm(grid_idx * 1.0 / n_blocks)
      color = [int(255*color[0]), int(255*color[1]), int(255*color[2])]

    if vis_grid_idx is not None and vis_grid_idx != grid_idx:
      continue

    x = gw * 8 + bw
    y = gh * 8 + bh
    z = gd * 8 + bd
    width = 2**(3 - level)

    x0 = x
    y0 = y
    z0 = z
    x1 = x + width
    y1 = y + width
    z1 = z + width

    if leaf:
      color_str = 'color_%d' % grid_idx
      f.write('\\definecolor{%s}{RGB}{%d,%d,%d}\n' % (color_str, color[0], color[1], color[2]))
      tikz_cube(f, x0,y0,z0, x1,y1,z1, 'very thin, %s' % (color_str))
  f.write("\\end{tikzpicture}\n")
  f.write("\\end{document}\n")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.

off_path = 'airplane_0710.off'
factor = 2
depth, height, width  = 8 * factor, 8 * factor, 8 * factor
R = np.eye(3, dtype=np.float32)
grid = pyoctnet.Octree.create_from_bin(b'test.oc')#create_from_off(off_path, depth, height, width, R, pack=False, n_threads=6)

f = open('out_solid.tex', 'w')
tikz_grid_solid(f, grid)
f.close()

f = open('out_wireframe.tex', 'w')
tikz_grid_wireframe(f, grid)
f.close()