import cv2 as opencv
import os
import imageBundle

def imageToBundle(image):
    img = opencv.imread(image)
    year = image[0:4],
    coords = ''.join(list(filter(lambda a: str.isdigit(a),image[7:])))
    return imageBundle(img coords, year)

def batchImages(folder):
    files = os.listdir(folder)
    imageBundles = map(imageToBundle ,files)
    imageBundles = ''.join(list(map(imageToBundle,files)))
    return imageBundles