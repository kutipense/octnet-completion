#!/usr/bin/env python3
#encoding: UTF-8

import sys
#import cv2
import numpy
import matplotlib.pyplot as pyplot

# Adapt depending on the working directory from which this script is called!
sys.path.insert(0, '../../common/py')

import pyoctree_cpu

def get_triangle_image(height = 400, width = 400):
    """
    Creates a binary image showing a triangle in the center.
    
    :param height: height of image
    :type height: int
    :param width: width of image
    :type width: int
    :return: image as float numpy array
    :rtype: numpy.ndarray
    """
    
    image = numpy.zeros((height, width, 1), dtype = numpy.float32)
    base_i = int(1/4.*height)
    base_height = int(1/2.*height)
    base_j = int(1/4.*width)
    base_width = int(1/2.*width)
    
    k = 0
    for i in range(base_i, base_i + base_height):
        for j in range(base_j, base_j + base_width):

            if j >= k and j <= width - k:
                image[i, j] = 1

        k += 1
    
    return image

def pyplot_show(image):
    """
    Show an image.

    :param image: image as numpy with either no channel dimension or channel dimension 3 or 4.
    :type image: numpy.ndarray
    """

    pyplot.imshow(image)
    pyplot.colorbar()
    pyplot.show()

if __name__ == "__main__":

    height = 400
    width = 400
    triangle_image = get_triangle_image(height, width)
    pyplot_show(triangle_image.reshape(width, height))

    octree = pyoctree_cpu.Octree.create_from_dense(triangle_image, numpy.array([[0.5, 1.5]], dtype = numpy.float32))

    level_image = numpy.zeros((height, width))
    for (leaf, grid_idx, bit_idx, gd, gh, gw, bd, bh, bw, level) in pyoctree_cpu.leaf_iterator(octree, leafs_only = True):
        #print(gh, gd, bh, bd, level)

        if level == 0:
            level_image[gh*8: gh*8 + 8, gd*8:gd*8 + 8] = max(numpy.max(level_image[gh*8: gh*8 + 8, gd*8:gd*8 + 8]), level)
        elif level == 1:
            level_image[gh*8 + bh: gh*8 + bh + 4, gd*8 + bd: gd*8 + bd + 4] = max(numpy.max(level_image[gh*8 + bh: gh*8 + bh + 4, gd*8 + bd: gd*8 + bd + 4]), level)
        elif level == 2:
            level_image[gh*8 + bh: gh*8 + bh + 2, gd*8 + bd: gd*8 + bd + 2] = max(numpy.max(level_image[gh*8 + bh: gh*8 + bh + 2, gd*8 + bd: gd*8 + bd + 2]), level)
        else:
            level_image[gh*8 + bh, gd*8 + bd] = max(numpy.max(level_image[gh*8 + bh, gd*8 + bd]), level)

    pyplot_show(numpy.transpose(level_image))
