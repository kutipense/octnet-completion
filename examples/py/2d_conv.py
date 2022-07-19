#!/usr/bin/env python3
# encoding: UTF-8

import sys
# import cv2
import numpy
import matplotlib.pyplot as pyplot

# Adapt depending on the working directory from which this script is called!
sys.path.insert(0, '../../common/py')

import pyoctree_cpu


def get_triangle_image(height = 400, width = 400, channels = 1):
    """
    Creates a binary image showing a triangle in the center.

    :param height: height of image
    :type height: int
    :param width: width of image
    :type width: int
    :param channels: number of channels, if 1 or smaller, no third dimension is added
    :type channels: int
    :return: image as float numpy array
    :rtype: numpy.ndarray
    """

    if channels > 1:
        image = numpy.zeros((height, width, 3), dtype=numpy.float32)
    else:
        image = numpy.zeros((height, width), dtype=numpy.float32)

    base_i = int(1 / 4. * height)
    base_height = int(1 / 2. * height)
    base_j = int(1 / 4. * width)
    base_width = int(1 / 2. * width)

    k = 0
    for i in range(base_i, base_i + base_height):
        for j in range(base_j, base_j + base_width):

            if j >= k and j <= width - k:
                image[i, j] = 1

        k += 1

    return image


def pyplot_show(images):
    """
    Show an image.

    :param image: array of images as numpy with either no channel dimension or channel dimension 3 or 4.
    :type image: [numpy.ndarray]
    """

    f, axarr = pyplot.subplots(len(images))

    for i in range(len(images)):
        axarr[i].imshow(images[i])

    #pyplot.colorbar()
    pyplot.show()

if __name__ == "__main__":

    height = 400
    width = 400
    image_in = get_triangle_image(height, width, 3)

    weights = numpy.ones((3, 3, 3, 3, 3), dtype = numpy.float32)
    #weights = numpy.zeros((3, 3, 3, 3, 3), dtype = numpy.float32)
    #weights[:, :, :, 0, :] = 1
    #weights[:, :, :, 2, :] = -1
    bias = numpy.zeros((3), dtype = numpy.float32)

    octree_in = pyoctree_cpu.Octree.create_from_dense(image_in, numpy.array([[0.5, 1.5]], dtype = numpy.float32))
    octree_out = pyoctree_cpu.Octree.create_empty()
    octree_in.conv_avg(weights, bias, octree_out)

    grid_out = octree_out.get_grid_data()
    image_out = numpy.zeros((octree_out.vx_height(), octree_out.vx_depth(), octree_out.vx_width()))

    for (leaf, grid_idx, bit_idx, gd, gh, gw, bd, bh, bw, level) in pyoctree_cpu.leaf_iterator(octree_out, leafs_only = True):
        #print(gh*8 + bh, gd*8 + bd, gw*8 + bw, grid_out[grid_idx])
        data_idx = octree_out.data_idx(grid_idx, bit_idx)

        if level == 0:
            image_out[gh*8 : gh*8 + 8, gd*8 : gd*8 + 8, gw*8 : gw*8 + 8] = (grid_out[data_idx] + grid_out[data_idx + 1] + grid_out[data_idx + 2])/3.
        elif level == 1:
            image_out[gh*8 + bh : gh*8 + bh + 4, gd*8 + bd : gd*8 + bd + 4, gw*8 + bw : gw*8 + bw + 4] = (grid_out[data_idx] + grid_out[data_idx + 1] + grid_out[data_idx + 2])/3.
        elif level == 2:
            image_out[gh*8 + bh : gh*8 + bh + 2, gd*8 + bd : gd*8 + bd + 2, gw*8 + bw : gw*8 + bw + 2] = (grid_out[data_idx] + grid_out[data_idx + 1] + grid_out[data_idx + 2])/3.
        else:
            image_out[gh*8 + bh, gd*8 + bd, gw*8 + bw] = (grid_out[data_idx] + grid_out[data_idx + 1] + grid_out[data_idx + 2])/3.

    pyplot_show([image_in, image_out[:, :, 0:3].transpose(1, 0, 2)])
