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
        return cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 6,
                                param1=80, param2=40, minRadius=45, maxRadius=70)

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

    def process(self, bgr_img):
        bgr_img = self.resize_img(bgr_img, 600)
        rgb_image, gray_image = self.transfom_bgr(bgr_img)
        gray_image_with_contrast = self.add_contrast_to_gray_img(gray_image)
        circles = self.get_circles_from_gray_img(gray_image_with_contrast)
        img = merge_img_with_circles(rgb_image, circles)

        # plt.subplot(121), plt.imshow(rgb_img)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.imshow(img)
        # plt.subplot(122), plt.imshow(cimg)
        # plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
        plt.show()
