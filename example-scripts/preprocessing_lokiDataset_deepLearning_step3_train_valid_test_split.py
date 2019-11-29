#!/usr/bin/env python
# coding: utf-8

# # Split in Train-, Valid- and Testdataset
#
#

# <div class="alert alert-block alert-info">
# <b>Attention:</b> The following variables must be edited.
#
# - __whitelistPath__ specifies which classes are to be considered.
# - __tsvFile__ contains metadata of all images, stored in a tsv file.
#
#
# - __tsvFileTrain__ is a tsv file in which the metadata for all training images is saved at the end.
# - __tsvFileValid__ is a tsv file in which the metadata for all validation images is saved at the end.
# - __tsvFileTest__ is a tsv file in which the metadata for all test images is saved at the end.
#
#
# - __trainPercentage__ specifies the percentage of total samples to be used as training data set.
# - __validPercentage__ specifies the percentage of total samples to be used as validation data set.
# - __testPercentage__ specifies the percentage of total samples to be used as test data set.
#
#
# - __randomSeed__ xy specifies a random seed value that is used through the entire notebook. This is necessary for reproducibility.
#
# </div>

# Set the path to the root folder containing the training data.
# If you want to have access to the data please contact ...
basePath = ''

whitelistPath = basePath + 'whitelist.txt'
tsvFile =       basePath + 'Trainingsdatensatz_Klassengroesse.tsv'
tsvFileTrain =  basePath + 'train.tsv'
tsvFileValid =  basePath + 'val.tsv'
tsvFileTest  =  basePath + 'test.tsv'

trainPercentage = 60
validPercentage = 20
testPercentage  = 20

randomSeed = 3


import csv
import random


# read whitelist

whitelist = []

with open(whitelistPath, errors='ignore') as whitelistfile:
    reader = csv.reader(whitelistfile, delimiter=',')
    for row in reader:
        whitelist.append(str(row)[2:-2])

len(whitelist)



# read tsv

tsv = []
udialect=csv.unix_dialect

with open(tsvFile, dialect=udialect, errors='ignore') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        tsv.append(row)
len(tsv)


def percentage(percent, whole):
    return percent/100*whole


# This method randomly divides samples in train, validation, and test datase for classes that have been augmented.The original samples are randomly split at first. The augmented samples were then added to the corresponding of the three data sets containing the original sample.


def splitAugmentedClass(classList):
    print("Enter splitAugmentedClass...")

    origTrain = []
    origValid = []
    origTest  = []

    tempTrain = []
    tempValid = []
    tempTest  = []

    tempTrainNo = 0
    tempValidNo = 0
    tempTestNo  = 0

    origList = [] # Store all original rows (without "augmentation")
    augmentedList = []
    for row in classList:
        if "augmented" not in row[0]:
            origList.append(row)
        if "augmented" in row[0]:
            augmentedList.append(row)

    random.Random(randomSeed).shuffle(origList)

    trainNoOrig = int(percentage(trainPercentage, len(origList)))
    validNoOrig = int(percentage(validPercentage, len(origList)))
    testNoOrig  = int(percentage(testPercentage,  len(origList)))

    for i in range(trainNoOrig):
           origTrain.append(origList.pop())
    for i in range(validNoOrig):
           origValid.append(origList.pop())
    for i in range(testNoOrig):
           origTest.append(origList.pop())


    for origRow in origTrain:
        for row in augmentedList:
            if row[0].startswith(origRow[0]):
                tempTrain.append(row)

    for origRow in origValid:
        for row in augmentedList:
            if row[0].startswith(origRow[0]):
                tempValid.append(row)

    for origRow in origTest:
        for row in augmentedList:
            if row[0].startswith(origRow[0]):
                tempTest.append(row)

    # Remaining Elements in origList
    for origRow in origList:
        for row in augmentedList:
            if row[0].startswith(origRow[0]):
                tempTrain.append(origRow)
        tempTrain.append(origRow)

    tempTrain = tempTrain + origTrain
    tempValid = tempValid + origValid
    tempTest  = tempTest  + origTest

    return tempTrain, tempValid, tempTest


# This method randomly divides samples in train, validation, and test datase for classes that have not been augmented.


def splitClass(classList):
    print("Enter splitClass...")

    trainNo = int(percentage(trainPercentage, len(classList)))
    validNo = int(percentage(validPercentage, len(classList)))
    testNo  = int(percentage(testPercentage,  len(classList)))


    random.Random(randomSeed).shuffle(classList)

    tr = []
    va = []
    te = []

    for i in range(trainNo):
           tr.append(classList.pop())
    for i in range(validNo):
           va.append(classList.pop())
    for i in range(testNo):
           te.append(classList.pop())

    return tr, va, te


train = [] # 60%
valid = [] # 20%
test  = [] # 20%
udialect=csv.unix_dialect


for planktonClass in whitelist:
    classList = [] # Alle Tiere der aktuellen Klasse
    augmented = False # handelt es sich bei dieser Klasse, um eine augmented Klasse?
    for row in tsv:
        if row[14].startswith(planktonClass):
            classList.append(row)
            if 'augmented' in row[0]:
                augmented = True

    if augmented:
        tr, va, te = splitAugmentedClass(classList)
        train = train + tr
        valid = valid + va
        test  = test  + te
    else:
        tr, va, te = splitClass(classList)
        train = train + tr
        valid = valid + va
        test  = test  + te


print(len(train))
print(len(valid))
print(len(test))
print(len(train +valid + test))


with open(tsvFileTrain, 'w') as tsvfile:
    writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
    writer.writerows(train)


with open(tsvFileValid, 'w') as tsvfile:
    writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
    writer.writerows(valid)


with open(tsvFileTest, 'w') as tsvfile:
    writer = csv.writer(tsvfile, dialect=udialect, delimiter='\t', quotechar="'")
    writer.writerows(test)
