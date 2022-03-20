# evm

Implementations of [Eulerian Video Magnification](https://people.csail.mit.edu/mrub/evm/) (EVM) and [Phase-based Video Magnification](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf) (PVM)

The EVM (`evm.py`) implementation focuses on motion-magnification by using a Butterworth filter applied on a Laplacian pyramid to selectively amplify spatiotemporal frequency variations.

It is also possible to use other signal filters in EVM such as the ideal bandpass filter to select temporal variations with low spatial frequency variations to amplify colour variations rather than motion.

The PVM (`pvm.py`) implementation uses [Riesz pyramids](https://people.csail.mit.edu/nwadhwa/riesz-pyramid/RieszPyr.pdf) to compute and amplify quaternionic phase differences to magnify motion in the dominant orientiation.

The Riesz pyramid implementation (`pyramids.py`) uses an approximation of the Riesz transformation using three tap finite difference convolution filters.
