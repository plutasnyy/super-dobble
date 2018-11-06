import cv2
import numpy as np
from ImageProcessor import ImageProcessor
from ImageReader import ImageReader

imageProcessor = ImageProcessor()

imageReader = ImageReader()
image = imageReader.get_image()
im = cv2.imread("data/circles/2.JPG", cv2.IMREAD_GRAYSCALE)


result = imageProcessor.process(image)
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 10000;

# Filter by Area.
params.filterByArea = True
params.minArea = 1000
params.maxArea = 200000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.maxConvexity=1

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
# Set up the detector with default parameters.

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)