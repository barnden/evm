#!/bin/python3

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.signal import butter
from skimage.color import rgb2yiq, yiq2rgb
import cv2


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


def ideal_bandpass_filter(im, fps):
    freq = list(range(n))
    pass


if __name__ == '__main__':
    # Video input
    input_video_file = "./Videos/face.mp4"
    cap = cv2.VideoCapture(input_video_file)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Framerate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Depth of Laplacian pyramids
    depth = 4

    # List of Laplacian pyramids
    lpyramids = []

    # Parameters for Temporal filtering
    freq_low = .85
    freq_high = 1.5

    # Create lowpass Butterworth filters
    low_a, low_b = butter(1, freq_low / fps)
    high_a, high_b = butter(1, freq_high / fps)

    # Generate a Laplacian pyramid for each frame
    while True:
        flag, frame = cap.read()

        if flag:
            lpyramid = laplacian_pyramid(rgb2yiq(frame), depth)
            lpyramids.append(lpyramid)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    cap.release()

    lowpass1 = lpyramids[0]
    lowpass2 = lpyramids[0]

    # Dimensions of padded L1
    h, w, _c = lpyramids[0][0].shape

    # Video output
    out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'FMP4'), fps, (w, h), True)

    # Amplification parameters (see paper)
    lambda_c = 16
    alpha = 24

    for i in range(1, int(pos_frame)):
        curr = lpyramids[i]
        prev = lpyramids[i - 1]

        lowpass1 = np.divide(
            np.multiply(-high_b[1], lowpass1) + np.multiply(high_a[0], curr) + np.multiply(high_a[1], prev),
            high_b[0])
        lowpass2 = np.divide(
            np.multiply(-low_b[1], lowpass2) + np.multiply(low_a[0], curr) + np.multiply(low_a[1], prev),
            low_b[0])

        filtered = lowpass1 - lowpass2

        kappa = (w ** 2 + h ** 2) ** .5 / 3
        delta = lambda_c / 8 / (1 + alpha)

        for j in range(depth):
            alpha_j = 2 * (kappa / delta / 8 - 1)
            # Ignore the lowest and highest frequencies
            if j == depth - 1 or j == 0:
                filtered[j] *= 0
            elif alpha_j > alpha:
                filtered[j] *= alpha
            else:
                filtered[j] *= alpha_j

            kappa /= 2

        # Reconstruct Laplacian pyramid
        for j in range(depth - 1, 0, -1):
            filtered[j - 1] += cv2.pyrUp(filtered[j])

        for j in range(depth - 1, 0, -1):
            curr[j - 1] += cv2.pyrUp(curr[j])

        result = (.65 * filtered[0]) + curr[0]
        result = 255 * np.clip(yiq2rgb(result), 0, 1)
        result = result.astype(np.uint8)

        cv2.imshow("preview", result)
        out.write(result)

        if cv2.waitKey(int(1000 / fps)) == 27:
            break

    out.release()
    cv2.destroyAllWindows()
