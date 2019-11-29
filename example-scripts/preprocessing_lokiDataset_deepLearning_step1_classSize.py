#!/usr/bin/env python
# coding: utf-8

# # Begrenzen des Datasets auf 1000
#
# Dieses Notebook begrenzt jede Klasse auf 1000.
# Klassen > 1000 werden auf 1000 beschränkt, dafür wird erst nach timestamp sortiert, anschliessend jedes xte Element gewählt.
# Klassen < 1000 werden auf 1000 augmented und anschliessend überzählige zufällig gelöscht.
#


from os import listdir,remove
from os.path import isfile, join
import csv
from skimage import io
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import random


# <div class="alert alert-block alert-info">
# <b>Attention:</b> The following variables must be edited.
#
# - __whitelistPath__ specifies which classes are to be considered.
# - __origImageDir__ contains all original images.
# - __origEcotaxaFile__ contains metadata of the images, stored in a tsv file.
#
#
# - __destImageDir__ is a folder in which all selected images are saved at the end.
# - __destEcotaxaFile__ is a tsv file in which the metadata for the selected images is stored.
#
#
# - __onlyValidated__ indicates whether only validated samples should be considered.
# - __noOfSamplesPerClass__ indicates the number of samples per class.
#
#
# </div>



# Set the path to the root folder containing the training data.
# If you want to have access to the data please contact ...
basePath = ''

whitelistPath =   basePath + 'whitelist.txt'
origImageDir =    basePath + 'Trainingsdatensatz/Trainingsdatensatz/'
origEcotaxaFile = basePath + 'trainingsdatensatz.tsv'

destImageDir =    basePath + 'Trainingsdatensatz_Klassengroesse/'
destEcotaxaFile = basePath + 'Trainingsdatensatz_Klassengroesse.tsv'

onlyValidated =   True
noOfSamplesPerClass = 1000


# In this cell the __whitelist__ is read in.


whitelist = []

with open(whitelistPath, errors='ignore') as whitelistfile:
    reader = csv.reader(whitelistfile, delimiter=',')
    for row in reader:
        whitelist.append(str(row)[2:-2])

print("Number of elements in the whitelist: " + str(len(whitelist)))


# In this cell only rows are read from the __origEcotaxaFile__ which correspond to the __validated__ variable as well as to the __whitelist__.


filteredTsv = []

with open(origEcotaxaFile, errors='ignore') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if row[8] == 'validated': # is validated
            for wl in whitelist:  # is in whitelist
                if row[14].startswith(wl):
                    filteredTsv.append(row)

print("Number of remaining samples after cleaning: " + str(len(filteredTsv)))


# This method converts PNG images to JPEG and stores them in __destImageDir__. In addition, the associated metadata is stored in __destEcotaxaFile__.


def loadAndSave(classList):

    udialect=csv.unix_dialect
    for f in classList:
        path = origImageDir + f[0]
        if isfile(path+".png"):
            im = Image.open(path+".png")
            im = im.convert('RGB')

        elif isfile(path+".jpg"):
            im = Image.open(path+".jpg")
        else:
            print("Error: " + f[0])
            continue
        im.save(destImageDir + f[0] + ".jpg")
        im.close()

    with open(destEcotaxaFile, 'a') as tsvfile:
        writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
        writer.writerows(classList)


# This method is a helper method for sorting by timestamp.


def sortTimestamp(val):
    return val[4]


# This method sorts all images by their time stamp and then takes every nth element until __noSamplesPerClass__ samples were reached. N is calculated by the number of samples in the respective class divided by __noSamplesPerClass__ as an integer.


def moreThanNoOfSamplesPerClass(classList):

    classList.sort(key = sortTimestamp)
    n = int(len(classList) / noOfSamplesPerClass)

    classList = classList[0::n]
    classList = classList[:noOfSamplesPerClass]

    loadAndSave(classList)


# In this method, small classes were filled up to __noOfSamplesPerClass__ samples through data augmentation. Augmented samples get an extension of their object_id field in the metadata. The original object_id is enhanced by the addition "augmented“ and a serial number.


def lessThanNoOfSamplesPerClass(classList):

    udialect=csv.unix_dialect
    n = int(np.ceil(noOfSamplesPerClass / len(classList)))
    newClassList=[]
    images = []

    for item in classList:

        path = origImageDir + item[0]
        if isfile(path+".png"):
            im = Image.open(path+".png")
            im = im.convert('RGB')

        elif isfile(path+".jpg"):
            im = Image.open(path+".jpg")

        else:
            print("Error: " + item[0])
            continue

        im.save(destImageDir + item[0] + ".jpg")

        with open(destEcotaxaFile, 'a') as tsvfile:
            writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
            writer.writerow(item)


        #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        # create image data augmentation generator
        datagen = ImageDataGenerator(horizontal_flip = True,
                                    rotation_range=180,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    fill_mode='nearest')


        sample = np.expand_dims(im, 0)
        # prepare iterator
        it = datagen.flow(sample, batch_size=1)
        # generate samples and plot
        for i in range(n-1):

            # generate batch of images
            batch = it.next()

            images.append(batch[0])

            # new class list append
            temp = item.copy()
            temp[0] = temp[0] + " augmented " + str(i)
            newClassList.append(temp)


    combined = list(zip(images, newClassList))
    random.shuffle(combined)
    images[:], newClassList[:] = zip(*combined)
    images = images[:(noOfSamplesPerClass-len(classList))]
    newClassList = newClassList[:(noOfSamplesPerClass- len(classList))]

    for imgTemp, tsvRow in zip(images,newClassList):
        image = Image.fromarray(np.uint8(imgTemp), 'RGB')
        image.save(destImageDir + tsvRow[0] + ".jpg")

    with open(destEcotaxaFile, 'a') as tsvfile:
        writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
        writer.writerows(newClassList)


# This cell checks for each class whether it contains more or less samples than __noOfSamplesPerClass__. The classes are then reduced or extended and saved.


for itemClass in whitelist:
    temp = []
    for tsvRow in filteredTsv:
        if tsvRow[14].startswith(itemClass):
            temp.append(tsvRow)

    if len(temp) >= noOfSamplesPerClass:

        moreThanNoOfSamplesPerClass(temp)
    else:

        lessThanNoOfSamplesPerClass(temp)
