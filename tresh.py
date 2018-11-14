import cv2
import numpy as np
import imutils


def nothing(x):
    pass


# Load an image
img = cv2.imread('data/hard/8.JPG')

# Resize The image
if img.shape[1] > 600:
    img = imutils.resize(img, width=600)
new = img.copy()
new = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
print(np.average(new[:,:,0]))
print(np.average(new[:,:,1]))
print(np.average(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]))

# Create a window
cv2.namedWindow('Treshed')

# create trackbars for treshold change
cv2.createTrackbar('Treshold', 'Treshed', 0, 255, nothing)

while (1):

    # Clone original image to not overlap drawings
    clone = img.copy()

    # Convert to gray
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('Treshold', 'Treshed')

    # Thresholding the gray image
    ret, gray_threshed = cv2.threshold(gray, r, 255, cv2.THRESH_BINARY)

    # Blur an image

    # Find contours
    _, contours, _ = cv2.findContours(gray_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        # approximte for circles
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 30)):
            contour_list.append(contour)

    # Draw contours on the original image
    cv2.drawContours(clone, contour_list, -1, (255, 0, 0), 2)

    # there is an outer boundary and inner boundary for each eadge, so contours double
    print('Number of found circles: {}'.format(int(len(contour_list) / 2)))

    # Displaying the results
    cv2.imshow('Objects Detected', clone)
    cv2.imshow("Treshed", gray_threshed)

    # ESC to break
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# close all open windows
cv2.destroyAllWindows()