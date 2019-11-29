#!/usr/bin/env python
# coding: utf-8

# # Image preprocessing
#
# - Cropping images
# - Center images
#

# <div class="alert alert-block alert-info">
# <b>Attention:</b> The following variables must be edited.
#
# - __imgDir__ contains all original images.
# - __destImgDir__ is a folder in which all preprocessed images are saved at the end.
# - __cropImageTolerance__ specifies the pixel tolerance to be taken into account when cropping the images.
#
#
# </div>

# Set the path to the root folder containing the training data.
# If you want to have access to the data please contact ...
basePath = ''

imgDir     = basePath + 'Trainingsdatensatz_Klassengroesse/'
destImgDir = basePath + 'Trainingsdatensatz_preprocessed/'

cropImageTolerance = 245


from PIL import Image
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import numpy as np
import ntpath


def show_images_unscaled(img):
    cols = 5
    rows = math.ceil(len(img)/ 5) # Anzahl der Bilder durch 5; Aufgerundet

    fix, axes = plt.subplots(rows, cols, figsize=(1 + 3 * cols, 3 * rows), subplot_kw={'xticks': (), 'yticks': ()})
    for image, ax in zip(img, axes.ravel()):
        ax.imshow(image, cmap=plt.get_cmap('gray'))


def maxHeight(images):
    maxHeight=(0,0)
    for img in images:
        if img.shape[0] > maxHeight[0]:
            maxHeight = (img.shape[0],0)
        if img.shape[1] > maxHeight[0]:
            maxHeight = (img.shape[1],0)
    return maxHeight[0]


# Bilder zuschneiden

def crop_images(images, tol=0):
    croppedImages = []
    for img in images:
        croppedImages.append(crop_image(img, tol))
    return croppedImages

# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
def crop_image(img,tol=0):
    mask = img<tol
    return img[np.ix_(mask.any(1),mask.any(0))]


# Bilder einlesen
imageList = []
pathList  = []
for filename in glob.glob(imgDir + "*.jpg"):
    image = io.imread(filename)
    imageList.append(image)
    pathList.append(filename)


#reduce dimension

i = 0
for item in imageList:
    imageList[i] = item[:,:,0]
    i = i+1


len(imageList)


#Nur zum Test

#imageList = imageList[50:500]
#pathList  = pathList[50:500]


#show_images_unscaled(imageList)


croppedImages= crop_images(imageList,245)


#show_images_unscaled(croppedImages)


# just extract filename, not the whoel path. that would lead to the source directory
imgNames = []

for path in pathList:
    imgNames.append(ntpath.basename(path))


images = []

height = maxHeight(croppedImages)

i = 0
for imgTemp, filenameTemp in zip(croppedImages, imgNames):
    img = Image.fromarray((imgTemp).astype('uint8'), mode='L')
    img_w, img_h = img.size
    background = Image.new('L', (height, height),"white") # Schwarzes Bild wird erstellt.
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset) # Ursprungsbild wird mittig in das schwarze Bild gesetzt.

    pathTemp = destImgDir + filenameTemp
    background.save(pathTemp)
    #croppedImages[i] = background
    i = i+1
