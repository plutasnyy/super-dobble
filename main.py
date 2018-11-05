import cv2
import numpy as np
from ImageProcessor import ImageProcessor
from ImageReader import ImageReader

imageProcessor = ImageProcessor()

imageReader = ImageReader()
image = imageReader.get_image()

result = imageProcessor.process(image)
#
# img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
# ret, img  =cv2.threshold(img,122 ,255,cv2.THRESH_TOZERO)
# img = cv2.medianBlur(img,5)
# img = cv2.medianBlur(img,5)
# img = cv2.medianBlur(img,5)
# # Set up the detector with default parameters.
#
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 300
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 90
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.01
#
# # Filter by Convexity
# params.filterByConvexity = False
# params.minConvexity = 0.87
#
# # Filter by Inertia
# params.filterByInertia = False
# params.minInertiaRatio = 0.01
#
# detector = cv2.SimpleBlobDetector_create(params)
#
# # Detect blobs.
# keypoints = detector.detect(img)
# print(keypoints)
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255))
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
