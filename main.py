from copy import deepcopy

import cv2
import imutils
import numpy as np


class CircleImage(object):

    def __init__(self, img, cx, cy, radius):
        self.img = img
        self.cx = cx
        self.cy = cy
        self.radius = radius


class AnimalImage(object):
    def __init__(self):
        pass


def resize_img(img, size=500):
    return imutils.resize(img, width=size)


def print_image(img):
    cv2.imshow('Print Image', img)
    cv2.waitKey(0)


def transfom_bgr(bgr_img):
    b, g, r = cv2.split(bgr_img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return rgb_img, gray_img


def add_contrast_to_gray_img(img):
    new_image = np.zeros(img.shape, img.dtype)
    for i in range(new_image.shape[0]):
        new_image[i] = np.clip(1.5 * img[i], 0, 255)
    return new_image


def get_image():
    i = 'easy/3'
    file_name = 'data/{}.JPG'.format(i)
    img = cv2.imread(file_name)
    return img


def get_conturous_from_img(img):
    img = cv2.bilateralFilter(img, 3, 175, 175)
    # ret, bgr_img = cv2.threshold(bilateral_filtered_image, 122, 255, cv2.THRESH_TOZERO)

    img = cv2.medianBlur(img, 5)

    edge_detected_image = cv2.Canny(img, 75, 200)

    image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = list()
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contour_list.append(contour)

    return contour_list


def get_circles_list(img, circles_conturous):
    circles_images_list = list()
    for i, contour in enumerate(circles_conturous):
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [contour], (1, 1, 1))
        masked_image = img * (mask.astype(img.dtype))

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius) + 10
        crop_image = masked_image[cy - radius:cy + radius, cx - radius:cx + radius].copy()
        print_image(crop_image)
        circle_image = CircleImage(crop_image, cx, cy, radius)
        circles_images_list.append(deepcopy(circle_image))
    return circles_images_list


def prepare_circle_to_cut_animals(img):
    b, g, r = cv2.split(img)
    new_img = img
    for i in range(len(b)):
        for j in range(len(b[0])):
            new_img[i][j] = min(b[i][j], g[i][j], r[i][j])

    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 2) * 255.0, 0, 255)

    new_img = cv2.LUT(new_img, lookUpTable)
    # cv2.imshow("a",img)
    # cv2.waitKey(0)
    ret, new_img = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)

    return new_img


def get_animals_from_circle(circleImage):
    img = circleImage.img.copy()
    img = prepare_circle_to_cut_animals(img)
    img = cv2.bilateralFilter(img, 5, 175, 175)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (3, 3), 15)

    cv2.imshow('Bilateral', img)
    cv2.waitKey(0)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    edge_detected_image = cv2.Canny(img, 75, 200)

    image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if hierarchy[0][i][3] == 1:
            print(area)
            contour_list.append(contour)
    contour_list = sorted(contour_list, key=cv2.contourArea, reverse=True)[:8]

    cv2.drawContours(img, contour_list, -1, (0, 255, 0), 2)
    cv2.imshow('Objects Detected', img)
    cv2.waitKey(0)

    # # blank mask:
    # mask = np.zeros_like(img)
    #
    # # filling pixels inside the polygon defined by "vertices" with the fill color
    # cv2.fillPoly(mask, contour, 255)

    # returning the image only where mask pixels are nonzero
    # masked = cv2.bitwise_and(img, mask)
    # cv2.imshow("wad", masked)
    # cv2.waitKey(0)
    # print(cx, cy, radius)

    pass


img = get_image()
img = resize_img(img, 2000)

circles_conturous = get_conturous_from_img(img)
circles_list = get_circles_list(img, circles_conturous)

for i in circles_list:
    get_animals_from_circle(i)
