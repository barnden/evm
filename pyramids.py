#!/bin/python3

from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

import cv2
import numpy as np

def gaussian_pyramid(im, n=3):
    w, h, _c = im.shape

    p2 = 2 ** (n - 1)

    # Pad to nearest multiple of 2^(n-1)
    if w % p2 != 0 or h % p2 != 0:
        im = np.pad(
            im,
            [(0, p2 - (w % p2)), (0, p2 - (h % p2)), (0, 0)],
            mode='constant')

    levels = [im]

    # We already have G1, so iterate to n - 1
    for i in range(n - 1):
        # Get current level
        Gi = cv2.pyrDown(levels[i])
        # Add to pyramid
        levels.append(Gi)

    return levels


def laplacian_pyramid(im, n=3):
    gpyramid = gaussian_pyramid(im, n)
    levels = []

    for i in range(n - 1):
        # Get current level
        Gi = gpyramid[i]
        w, h, _c = Gi.shape
        # Filter and upsample to same dimension as Gi
        Gn = cv2.pyrUp(gpyramid[i + 1])
        # Get Laplacian level i
        Li = Gi - Gn
        levels.append(Li)

    # Last Laplacian and Gaussian are the same
    levels.append(gpyramid[-1])

    return levels


def riesz_pyramid(im, n=3):
    # Implementation of Riesz pyramids
    # From the paper "Riesz Pyramids for Fast Phase-Based Video Magnification" by Wadhwa et al.

    lpyramid = laplacian_pyramid(im, n)
    levels_x = []
    levels_y = []

    # Riesz transformation approximation (see section 3.1 from Wadhwa et al.)
    kernel_x = np.array([
        [0, 0, 0],
        [.5, 0, -.5],
        [0, 0, 0]
    ])

    kernel_y = np.array([
        [0, .5, 0],
        [0, 0, 0],
        [0, -.5, 0]
    ])

    convolve = lambda layer, kernel : np.stack([convolve2d(layer[:, :, j], kernel, mode='same') for j in range(3)], axis=-1)

    for i in range(n - 1):
        levels_x.append(convolve(lpyramid[i], kernel_x))
        levels_x.append(convolve(lpyramid[i], kernel_y))

    return (lpyramid, levels_x, levels_y)
