#!/bin/python3

'''
Phase-Based Video Magnification using Riesz Pyramids
'''

import numpy as np
from scipy.signal import butter
from scipy.ndimage import gaussian_filter
import cv2

from pyramids import riesz_pyramid

fudge = .1
max_frames = -1

def phase_diff_ampl(curr, prev):
    curr_real, curr_x, curr_y = curr
    prev_real, prev_x, prev_y = prev

    q_conj_prod_real = np.multiply(curr_real, prev_real) + np.multiply(curr_x, prev_x) + np.multiply(curr_y, prev_y)
    q_conj_prod_x = np.multiply(prev_real, curr_x) - np.multiply(curr_real, prev_x)
    q_conj_prod_y = np.multiply(prev_real, curr_y) - np.multiply(curr_real, prev_y)

    q_conj_prod_ampl = np.sqrt(fudge + np.power(q_conj_prod_real, 2) + np.power(q_conj_prod_x, 2) + np.power(q_conj_prod_y, 2))
    phase_diff = np.arccos(np.divide(q_conj_prod_real, q_conj_prod_ampl))

    denominator = fudge + np.power(q_conj_prod_x, 2) + np.power(q_conj_prod_y, 2)
    cos_orientation = np.divide(q_conj_prod_x, denominator)
    sin_orientation = np.divide(q_conj_prod_y, denominator)

    phase_diff_cos = np.multiply(phase_diff, cos_orientation)
    phase_diff_sin = np.multiply(phase_diff, sin_orientation)

    amplitude = np.sqrt(q_conj_prod_ampl)

    return (phase_diff_cos, phase_diff_sin, amplitude)

def iir_temporal_filter(B, A, phase, register0, register1):
    temporally_filtered_phase = B[0] * phase + register0
    register0 = B[1] * phase + register1 - A[1] * temporally_filtered_phase
    reigster1 = B[2] * phase             - A[2] * temporally_filtered_phase

    return (temporally_filtered_phase, register0, register1)

def amplitude_weighted_blur(temporally_filtered_phase, amplitude, sigma):
    denominator = gaussian_filter(amplitude, sigma)
    numerator = gaussian_filter(np.multiply(temporally_filtered_phase, amplitude), sigma)

    # Spatially smooth temporally filtered phase
    return np.divide(numerator, denominator)

def phase_shift_coeff_real(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin):
    phase_mag = np.sqrt(np.power(phase_cos, 2) + np.power(phase_sin, 2))
    exp_phase_real = np.cos(phase_mag)

    denominator = fudge + np.multiply(phase_mag, np.sin(phase_mag))
    exp_phase_x = np.divide(phase_cos, denominator)
    exp_phase_y = np.divide(phase_sin, denominator)

    # Real part of quarternion mult
    return np.multiply(exp_phase_real, riesz_real) - np.multiply(exp_phase_x, riesz_x) - np.multiply(exp_phase_y, riesz_y)

if __name__ == '__main__':
    # Video input
    input_video_file = "./Videos/baby.mp4"
    cap = cv2.VideoCapture(input_video_file)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Framerate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Depth of Laplacian pyramids
    depth = 4

    # Parameters for Temporal filtering
    freq_low = .4
    freq_high = 3

    # Stdev for Gaussian kernel
    sigma = 2

    # Amplification factor
    ampl_factor = 20

    # Create lowpass Butterworth filters
    A, B = butter(1, (freq_low / fps / 2, freq_high / fps / 2), 'bandpass')

    # Perform Phase-based Video Magnification
    prev = None

    # Dimensions of padded L1
    h = None
    w = None

    # Video output
    out = None

    # Quaternionic phase information
    phase_cos = None
    phase_sin = None

    # IIR temporal filter values
    register0_cos = None
    register1_cos = None

    register0_sin = None
    register1_sin = None

    while True:
        flag, frame = cap.read()

        if flag:
            frame = frame.astype(float) / 255
            curr = riesz_pyramid(frame, depth)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if pos_frame == 1:
                prev = curr
                h, w, _c = prev[0][0].shape

                out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'FMP4'), fps, (w, h), True)

                phase_cos = [np.zeros(curr[0][j].shape) for j in range(depth - 1)]
                phase_sin = phase_cos.copy()

                register0_cos = phase_cos.copy()
                register1_cos = phase_cos.copy()

                register0_sin = phase_cos.copy()
                register1_sin = phase_cos.copy()

                continue

            magnified_lpyramid = [[]] * depth

            for j in range(depth - 1):
                # Compute quaternionic phase diff between current and prev Riesz pyramids
                phase_diff_cos, phase_diff_sin, ampl = phase_diff_ampl(
                    [curr[0][j], curr[1][j], curr[2][j]],
                    [prev[0][j], prev[1][j], prev[2][j]]
                )

                # Add quaternionic phase diff to current quaternionic phase
                phase_cos[j] += phase_diff_cos
                phase_sin[j] += phase_diff_sin

                # Filter the quaternionic phase temporally
                phase_filtered_cos, register0_cos[j], register1_cos[j] = iir_temporal_filter(B, A, phase_cos[j], register0_cos[j], register1_cos[j])
                phase_filtered_sin, register0_sin[j], register1_sin[j] = iir_temporal_filter(B, A, phase_sin[j], register0_sin[j], register1_sin[j])

                # Denoising and smoothing of errors by spatially blurring the temporally filtered quaternionic phase signals
                phase_filtered_cos = amplitude_weighted_blur(phase_filtered_cos, ampl, sigma)
                phase_filtered_sin = amplitude_weighted_blur(phase_filtered_sin, ampl, sigma)

                # Compute motion magnified pyramid
                # Phase shift input pyramid by spatiotemporally filtered quaternionic phase
                phase_mag_filtered_cos = ampl_factor * phase_filtered_cos
                phase_mag_filtered_sin = ampl_factor * phase_filtered_sin

                magnified_lpyramid[j] = phase_shift_coeff_real(curr[0][j], curr[1][j], curr[2][j], phase_mag_filtered_cos, phase_mag_filtered_sin)

            # Use residual lowpass from current frame
            magnified_lpyramid[-1] = curr[0][-1]

            # Pyramid collapse
            for j in range(depth - 1, 0, -1):
                magnified_lpyramid[j - 1] += cv2.pyrUp(magnified_lpyramid[j])

            result = 255 * np.clip(magnified_lpyramid[0], 0, 1)
            result = result.astype(np.uint8)

            cv2.imshow('hi', result)
            out.write(result)

            if cv2.waitKey(10) == 27:
                break

            if max_frames > 0 and pos_frame >= max_frames:
                break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    cap.release()

    out.release()
    cv2.destroyAllWindows()

