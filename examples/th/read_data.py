#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fname = "10155655850468db78d106ce0a280f87__0__.sdf"
header = np.fromfile(fname, dtype=np.uint64, count=3)
data = np.fromfile(fname, dtype=np.float32, offset=24) # uint64 * 3
sdf = data.reshape((header[0], header[1], header[2]))

# print(data[data==float('-inf')])

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)
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

colors = np.zeros((32,32,32,3))
colors[sdf>0] = (0,0,0)
colors[sdf<0] = (1,0,0)

l = (sdf<=3) & (sdf>=-25)
sdf[l] = 1
sdf[~l] = 0
# print(sdf.shape)

ax.voxels(sdf, edgecolor=(1,1,1,0.3), facecolors=colors)
plt.show()
# fig.savefig("wireframe2.jpg", dpi=300)