from copy import deepcopy
import cv2
import imutils
import numpy as np


class Image(object):

    def __init__(self, img, cx, cy, radius):
        self.img = img
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.animals_list = list()

    def set_animals_list(self, animals_list):
        self.animals_list = animals_list


class AnimalImage(Image):
    def __init__(self, rgb, image):
        super().__init__(rgb, image.cx, image.cy, image.radius)
        self.bw = image.img
        self.hu = cv2.HuMoments(cv2.moments(self.bw))


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
    img = cv2.bilateralFilter(img, 3, 175, 175)
    img = cv2.medianBlur(img, 5)
    edge_detected_image = cv2.Canny(img, 75, 200)
    image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = list()
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contour_list.append(contour)

    return contour_list


def get_objects_list(img, conturous, rgb_circle=None):
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
        if rgb_circle is not None:
            rgb_circle[:, :, 0] *= (mask.astype(img.dtype))
            rgb_circle[:, :, 1] *= (mask.astype(img.dtype))
            rgb_circle[:, :, 2] *= (mask.astype(img.dtype))
            crop_color_image = rgb_circle[cy - radius:cy + radius, cx - radius:cx + radius].copy()
            circle_image = AnimalImage(crop_color_image, circle_image)
        images_list.append(deepcopy(circle_image))
    return images_list


def prepare_circle_to_cut_animals(img):
    b, g, r = cv2.split(img)

    new_img = b.copy()
    for i in range(len(b)):
        for j in range(len(b[0])):
            new_img[i][j] = min(b[i][j], g[i][j], r[i][j])

    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, 2) * 255.0, 0, 255)

    new_img = cv2.LUT(new_img, look_up_table)

    _, new_img = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)

    new_img = cv2.bilateralFilter(new_img, 5, 175, 175)
    new_img = cv2.medianBlur(new_img, 5)
    new_img = cv2.GaussianBlur(new_img, (3, 3), 15)
    _, new_img = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)

    return new_img


def get_animals_from_circle(circleImage):
    img = circleImage.img.copy()
    img = prepare_circle_to_cut_animals(img)
    edge_detected_image = cv2.Canny(img, 75, 200)
    image, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if hierarchy[0][i][3] == 1:
            contour_list.append(contour)
    contour_list = sorted(contour_list, key=cv2.contourArea, reverse=True)[:8]
    white_image = np.ones(img.shape, np.uint8) * 255
    animals_images_list = get_objects_list(white_image, contour_list, circleImage.img.copy())
    return animals_images_list


def calculate_hu_diff(hu, hu1):
    return sum([abs(hu[i] - hu1[i]) for i in range(7)])*1000


def calculate_center_in_animal(circle, ind):
    cx = circle.cx - circle.radius + circle.animals_list[ind].cx
    cy = circle.cy - circle.radius + circle.animals_list[ind].cy
    return cx, cy


def compare_par_circles(circle1, circle2):
    best_pair = [None, None, 9999999]
    for i in range(len(circle1.animals_list)):
        for j in range(len(circle2.animals_list)):
            first = circle1.animals_list[i]
            second = circle2.animals_list[j]

            hu_diff = calculate_hu_diff(first.hu, second.hu)
            if hu_diff < best_pair[2]:
                best_pair = [i, j, hu_diff]
    if best_pair[0] is not None:
        return (calculate_center_in_animal(circle1, best_pair[0]), calculate_center_in_animal(circle2, best_pair[1]))
    return None


img = get_image('easy/4')
img = resize_img(img, 1000)

circles_conturous = get_contours_from_img(img)
circles_list = get_objects_list(img, circles_conturous)

for circle in circles_list:
    animals_list = get_animals_from_circle(circle)
    circle.set_animals_list(animals_list)

x = [len(i.animals_list) for i in circles_list]
print("Dlugosci listy zwierzat: ", x)

for i in range(len(circles_list) - 1):
    for j in range(i + 1, len(circles_list)):
        result = compare_par_circles(circles_list[i], circles_list[j])
        if result is not None:
            (a,b),(c,d) = result
            cv2.line(img, (a, b), (c, d), (0, 255, 0), 2)
print_image(img)
