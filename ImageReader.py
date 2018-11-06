import cv2


class ImageReader(object):

    @staticmethod
    def get_image():
        i = 'circles/1'
        file_name = 'data/{}.JPG'.format(i)
        img = cv2.imread(file_name)
        return img
