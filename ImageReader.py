import cv2


class ImageReader(object):

    @staticmethod
    def get_image():
        i = 'easy1'
        file_name = 'data/{}.jpg'.format(i)
        img = cv2.imread(file_name)
        return img
