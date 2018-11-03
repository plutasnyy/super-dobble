import cv2

from ImageProcessor import ImageProcessor
from ImageReader import ImageReader

imageProcessor = ImageProcessor()

imageReader = ImageReader()
image = imageReader.get_image()

result = imageProcessor.process(image)

