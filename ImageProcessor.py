import cv2
import imutils
import numpy as np
from skimage import color

from matplotlib import pyplot as plt


def merge_img_with_circles(image, circles):
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image


class ImageProcessor(object):

    @staticmethod
    def resize_img(img, size=500):
        return imutils.resize(img, width=size)

    @staticmethod
    def print_image(img):
        cv2.imshow('Print Image', img)
        cv2.waitKey(0)

    @staticmethod
    def get_circles_from_gray_img(img):
        rows = img.shape[0]
        return cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=40, param2=40, minRadius=0, maxRadius=0)

    @staticmethod
    def transfom_bgr(bgr_img):
        if bgr_img.shape[-1] == 3:  # color image
            b, g, r = cv2.split(bgr_img)  # get b,g,r
            rgb_img = cv2.merge([r, g, b])  # switch it to rgb
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = bgr_img
            raise ValueError
        return rgb_img, gray_img

    @staticmethod
    def add_contrast_to_gray_img(img):
        new_image = np.zeros(img.shape, img.dtype)
        for i in range(new_image.shape[0]):
            new_image[i] = np.clip(3 * img[i], 0, 255)
        return new_image

    def hough(self, bgr_image):

        bgr_img = self.resize_img(bgr_img, 1000)
        rgb_image, gray_image = self.transfom_bgr(bgr_img)

        gray_image_with_contrast = self.add_contrast_to_gray_img(gray_image)
        # ret, gray_image_with_contrast = cv2.threshold(gray_image_with_contrast, 122, 255, cv2.THRESH_TOZERO)
        for i in range(5):
            gray_image_with_contrast = cv2.medianBlur(gray_image_with_contrast, 5)
        circles = self.get_circles_from_gray_img(gray_image_with_contrast)
        print(len(circles), len(circles[0]), circles)
        img = merge_img_with_circles(rgb_image, circles)

        plt.subplot(121), plt.imshow(img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.imshow(img)
        plt.subplot(122), plt.imshow(gray_image_with_contrast)
        plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        plt.show()

    def process(self, bgr_img):
        bgr_img = self.resize_img(bgr_img, 1000)
        # rgb_image, gray_image = self.transfom_bgr(bgr_img)
        # gray_image_with_contrast = self.add_contrast_to_gray_img(gray_image)

        cv2.imshow('Original Image', bgr_img)
        cv2.waitKey(0)
        #
        # bilateral_filtered_image = cv2.bilateralFilter(gray_image_with_contrast, 5, 175, 175)
        # ret, gray_image_with_contrast = cv2.threshold(bilateral_filtered_image, 122, 255, cv2.THRESH_TOZERO)
        #
        # img = cv2.medianBlur(gray_image_with_contrast, 3)
        # bilateral_filtered_image = img
        # cv2.imshow('Bilateral', bgr_img)
        # cv2.waitKey(0)

        edge_detected_image = cv2.Canny(bgr_img, 75, 200)
        # cv2.imshow('Edge', edge_detected_image)
        # cv2.waitKey(0)

        image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = []
        for i, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if hierarchy[0][i][3] == -1:
                contour_list.append(contour)
                print(cx, cy, radius)
        cv2.drawContours(bgr_img, contour_list, -1, (255, 0, 0), 2)
        cv2.imshow('Objects Detected', bgr_img)
        cv2.waitKey(0)
