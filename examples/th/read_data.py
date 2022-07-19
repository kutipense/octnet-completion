#!/usr/bin/env python3
import numpy as np

fname = "10155655850468db78d106ce0a280f87__0__.sdf"
header = np.fromfile(fname, dtype=np.uint64, count=3)
data = np.fromfile(fname, dtype=np.float32, offset=24) # uint64 * 3
sdf = data.reshape((header[0], header[1], header[2]))

print(data[0])