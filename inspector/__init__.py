"""Budget-aware surface inspection using only NumPy and OpenCV."""

import cv2

# Bound native parallelism, including callers that import the package directly.
cv2.setNumThreads(2)

__version__ = "0.1.0"
