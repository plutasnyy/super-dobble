from copy import deepcopy
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Image(object):

    def __init__(self, img, cx, cy, radius):
        self.img = img
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.animals_list = list()

    def add_animal(self, animal):
        point, type = animal
        x = self.cx - self.radius + point[0]
        y = self.cy - self.radius + point[1]
        self.animals_list.append([(int(x), int(y)), type])


def resize_img(img, size=500):
    return imutils.resize(img, width=size)


def print_image(img, text='Title'):
    cv2.imshow(text, img)
    cv2.waitKey(0)


def print_tuple_images(img_tpl, text='Compare images'):
    numpy_horizontal = np.hstack(img_tpl)
    cv2.imshow(text, numpy_horizontal)
    cv2.waitKey(0)


def transfom_bgr(bgr_img):
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return rgb_img, gray_img


def add_contrast_to_gray_img(img):
    new_image = np.zeros(img.shape, img.dtype)
    for i in range(new_image.shape[0]):
        new_image[i] = np.clip(1.5 * img[i], 0, 255)
    return new_image


def get_image(name):
    file_name = 'data/{}.JPG'.format(name)
    img = cv2.imread(file_name)
    return img


def get_contours_from_img(img):
    # img = cv2.bilateralFilter(img, 3, 175, 175)
    img = cv2.medianBlur(img, 5)
    edge_detected_image = cv2.Canny(img, 75, 200)
    image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = list()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if hierarchy[0][i][3] == -1 and area > 100:
            contour_list.append(contour)

    return contour_list


def get_objects_list(img, conturous):
    images_list = list()
    for i, contour in enumerate(conturous):
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [contour], (1, 1, 1))
        masked_image = img * (mask.astype(img.dtype))

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius) + 10

        l_perpen, r_perpen = cy - radius, cy + radius
        b_level, t_level = cx - radius, cx + radius
        if l_perpen < 0:
            l_perpen = 0
        if b_level < 0:
            b_level = 0
        if r_perpen > img.shape[0]:
            r_perpen = img.shape[0]
        if t_level > img.shape[1]:
            t_level = img.shape[1]

        crop_image = masked_image[l_perpen:r_perpen, b_level:t_level].copy()
        # print_image(crop_image, "Wyciety obraz")
        circle_image = Image(crop_image, cx, cy, radius)
        images_list.append(deepcopy(circle_image))
    return images_list


def find_animal_on_circle(circle, pattern, type):
    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 0

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(pattern, None)
    kp2, des2 = sift.detectAndCompute(circle, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = list()
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None
        h, w = pattern.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cx = sum([x[0][0] for x in dst]) / 4
        cy = sum([y[0][1] for y in dst]) / 4
        return (cx, cy), type
    return None


def get_center_circle(img, circles_list):
    height, width = img.shape[:2]
    x = width / 2
    y = height / 2

    min_diff = 99999999999
    center_circle_ind = 0
    for ind, circle in enumerate(circles_list):
        diff = abs(circle.cx - x) + abs(circle.cy - y)
        if diff < min_diff:
            min_diff = diff
            center_circle_ind = ind
    return center_circle_ind


def find_pair(center_circle, circle):
    for center_animal in center_circle.animals_list:
        for other_animal in circle.animals_list:
            if center_animal[1] == other_animal[1]:
                return center_animal[0], other_animal[0]
    return None, None


img = get_image('easy/11') #medium2 easy11

img = resize_img(img, 1200)
circles_conturous = get_contours_from_img(img)

circles_list = get_objects_list(img, circles_conturous)
types_names = ['ant', 'bat', 'bear', 'beaver', 'buffalo', 'camel', 'capricorn', 'cat', 'catfish', 'cock', 'cow', 'crab',
               'cricket', 'crocodile', 'dog', 'dolphin', 'donkey', 'duck', 'eagle', 'fish', 'flamingo', 'frog',
               'gorilla', 'hedgehog', 'hippo', 'horse', 'jellyfish', 'kangaroo', 'lion', 'mosquito', 'mouse', 'octopus',
               'orca',
               'owl', 'pandabear', 'parrot', 'pelican', 'penguin', 'pig', 'pigeon', 'polarbear', 'rabbit', 'racoon',
               'reindeer', 'scorpio', 'seagull', 'seahorse', 'seal', 'shark', 'sheep', 'sloth', 'squirrel', 'starfish',
               'turtle', 'wolf', 'zebra']
types = list()
for name in types_names:
    types.append([name, cv2.imread('data/animals/{}.png'.format(name), 0)])

for ind, circle in enumerate(circles_list):
    # print_image(circle.img)
    for type in types:
        result = find_animal_on_circle(circle.img, type[1], type[0])
        if result is not None:
            circle.add_animal(result)
    print(ind, "({})".format(len(circle.animals_list)), circle.animals_list)

center_circle_index = get_center_circle(img, circles_list)
center_circle = circles_list[center_circle_index]

for ind, circle in enumerate(circles_list):
    for ind2, circle2 in enumerate(circles_list):
        if ind == ind2:
            continue
        p1, p2 = find_pair(circle2, circle)
        if p1 is not None:
            cv2.line(img, p1, p2, (0, 255, 0), 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
