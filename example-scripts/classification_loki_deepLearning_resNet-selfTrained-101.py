#!/usr/bin/env python
# coding: utf-8

# # resnet - self-trained

# lsls



# Set the path to the root folder containing the training data.
# If you want to have access to the data please contact ...
basePath = ''

imgDir    = basePath + 'images/Trainingsdatensatz_cropped_scaled/'
trainTsv  = basePath + 'tsvDatein/final_dataset_splitting/train.tsv'
validTsv  = basePath + 'tsvDatein/final_dataset_splitting/val.tsv'
testTsv   = basePath + 'tsvDatein/final_dataset_splitting/test.tsv'
whitelist = basePath + 'whitelist/whitelist1.txt'

saveDir   = basePath + 'experiments/ResNet_selfTrained_101/'

imgShape    = (1000,1000)
num_classes = 11
batch_size  = 1
max_epochs  = 40
preTrained  = False




#Imports
import csv
from keras_applications.resnet import ResNet101, preprocess_input


import keras
import keras_applications
keras_applications.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils
)

from keras.utils import to_categorical, Sequence
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
import numpy as np
from skimage import io
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pandas import DataFrame
from contextlib import redirect_stdout
import keras
import pydot
import pydotplus
from keras.utils.vis_utils import plot_model
from IPython.display import SVG


from keras.utils.vis_utils import model_to_dot
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

import time
import random




# Kopie aus der mgLearn Bibliothek. 
def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    
    
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)

    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
    ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
    ax.set_xticklabels(xticklabels, rotation='vertical')
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)
    
    
    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center", fontsize=12)
    
    
    return img




# https://stackoverflow.com/a/43186440

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        path = saveDir + "history/timeHistory.csv"
        with open(path,'a') as fd:
            fd.write(str(time.time()-self.epoch_time_start) + "\n")




def readTargetList(tsv, target_map):
    
    
# Read target list.
    with open(tsv) as f:
        reader = csv.reader(f,delimiter='\t')
        target = []
        imgId = []
        next(reader)
        i = 0
        for class_name in reader:
            if class_name[14] == "":
                print(className)
                continue

            target.append([value for key, value in target_map.items() if key in class_name[14]]) # nur substring ist wichtig
            imgId.append(class_name[0])
            i = i+1
            
    return target, imgId




def getImgPaths(imgId):
    filenames = (str(idx) + '.jpg' for idx in imgId)
    return [imgDir + filename for filename in filenames]




def bounds(old_size, new_size):
    if new_size >= old_size:
        return (0, old_size)
    else:
        diff = old_size - new_size
        low = diff // 2 + diff % 2
        high = low + new_size
        return (low, high)




def crop_image(img, shape):
    left, right = bounds(img.shape[0], shape[0])
    top, bottom = bounds(img.shape[1], shape[1])
    img = img[left:right, top:bottom]
    img = img[:, :,np.newaxis]
    
    return img




# Sequence class for training using lazy batches of images.
# See example at https://keras.io/utils/#sequence

#
# `X_set` is list of path to the images, and `y_set` are the associated classes.
#
class LokiImageSequence(Sequence):
    def __init__(self, X_set, y_set, batch_size, image_shape):
        self._X = list(X_set)
        self._y = list(y_set)
        self._batch_size = batch_size
        self._image_shape = image_shape

    def __len__(self):
        return int(np.ceil(len(self._X) / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_X = self._X[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]
        
        x = []
        for file_name in batch_X:
            z = io.imread(file_name)
            t = crop_image(z,self._image_shape)
            d = t[:,:,0]
            b = np.repeat(d[..., np.newaxis], 3, -1)
            x.append(b)
            
        x = preprocess_input(np.array(x))
        
        return(np.array(x), np.array(batch_y, dtype=np.int8))



# Datapreparation

# tsv einlesen und Generator erstellen

with open(whitelist) as f:
    inverse_target_map = dict(enumerate(f))
    target_map = {v[:-1]: k for (k, v) in inverse_target_map.items()}
    num_classes=(1 + max(inverse_target_map))

trainTarget, trainImgId = readTargetList(trainTsv, target_map)
validTarget, validImgId = readTargetList(validTsv, target_map)
testTarget,  testImgId  = readTargetList(testTsv,  target_map)
    
# shuffle
combined = list(zip(trainTarget, trainImgId))
random.shuffle(combined)
trainTarget[:], trainImgId[:] = zip(*combined)

# shuffle
combined = list(zip(validTarget, validImgId))
random.shuffle(combined)
validTarget[:], validImgId[:] = zip(*combined)

# shuffle
combined = list(zip(testTarget, testImgId))
random.shuffle(combined)
testTarget[:], testImgId[:] = zip(*combined)

        
# Test anzahl an Bildern beschränken
#trainTarget = trainTarget[:10]
#trainImgId = trainImgId[:10]
#validTarget = validTarget[:10]
#validImgId = validImgId[:10]
#testTarget = testTarget[:10]
#testImgId = testImgId[:10]


# image file paths
X_trainImgPath = getImgPaths(trainImgId)
X_validImgPath = getImgPaths(validImgId)
X_testImgPath  = getImgPaths(testImgId)

# Convert class vectors to binary class matrices (format required by Keras).
y_train = to_categorical(trainTarget, num_classes)
y_valid = to_categorical(validTarget, num_classes)
y_test  = to_categorical(testTarget,  num_classes)

# Constructing sequences
train_seq = LokiImageSequence(X_trainImgPath, y_train, batch_size, imgShape)
valid_seq = LokiImageSequence(X_validImgPath, y_valid, batch_size, imgShape)
test_seq  = LokiImageSequence(X_testImgPath,  y_test,  batch_size, imgShape)


print("Length trainingsset: "  + str(len(y_train)))
print("Length validationset: " + str(len(y_valid)))
print("Length testset: "       + str(len(y_test)))
print("Number of classes: "    + str(num_classes))


# Model customization

if preTrained:
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(1000,1000,3))
else:
    base_model = ResNet101(weights=None, include_top=False, input_shape=(1000,1000,3))

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers
if preTrained:
    for layer in base_model.layers:
        layer.trainable = False

# evtl. anpassen
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

csv_logger_callback = CSVLogger(saveDir + "history/model_history_log.csv", append=True)
checkpointEveryEpoch_callback = ModelCheckpoint(saveDir + "modelFiles/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=False, mode='max')
time_callback            = TimeHistory()
earlyStopping_callback   = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta = 0.01)
modelCheckpoint_callback = ModelCheckpoint(saveDir + 'modelFiles/best_model.h5', monitor='val_loss', verbose=1)

callback_list = [time_callback, earlyStopping_callback, modelCheckpoint_callback, checkpointEveryEpoch_callback,csv_logger_callback]


# Transfer learning
history = model.fit_generator(train_seq,
                     epochs = max_epochs,
            validation_data = valid_seq,
                  callbacks = callback_list)


# Speichern Model und weights

model_json = model.to_json()
with open(saveDir + "modelFiles/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(saveDir + "modelFiles/weights.h5")


# load a saved model
model = load_model(saveDir + 'modelFiles/best_model.h5')


# ## history


# convert the history.history dict to a pandas DataFrame:     
hist_df = DataFrame(history.history) 

# save to json:  
hist_json_file = saveDir + 'history/history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)



# summarize history for accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(saveDir + 'history/accuracy.svg', transparent = True, bbox_inches='tight')
plt.show()


# summarize history for loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(saveDir + 'history/loss.svg', transparent = True, bbox_inches='tight')
plt.show()


keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file=saveDir+'model_architecture_charts/model_small.png')


def plot_keras_model_verbose(model, show_shapes=True, show_layer_names=True):
    return SVG(model_to_dot(model, show_shapes=show_shapes,         
            show_layer_names=show_layer_names).create(prog='dot',format='svg'))


svg = plot_keras_model_verbose(model, show_shapes=True, show_layer_names=False)

with open(saveDir + "model_architecture_charts/model_verbose.svg", "w") as txt:
    txt.write(svg.data)

svg


# Save mode summary
with open(saveDir + 'model_architecture_charts/model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


# ## Trainings duration


times = time_callback.times


df = DataFrame(times)
df.to_csv (saveDir + r'trainingsDuration/durationPerEpoch.csv') 


sum = df.sum()
sum.to_csv(saveDir + r'trainingsDuration/durationSum.csv') 
print(sum)


avg = df.mean()
avg.to_csv(saveDir + r'trainingsDuration/durationAvgPerEpoch.csv') 
print(avg)


predValid = model.predict_generator(valid_seq)
predTest  = model.predict_generator(test_seq)

loss, acc = model.evaluate_generator(test_seq)


# ## Validationset


trueClassNum=[]
for x in y_valid:
    ind = np.array(x).argmax()
    y = ind
    trueClassNum.append(y)
    
    
trueClassName = []
for f in trueClassNum:
     trueClassName.append(inverse_target_map[f][:-1])


predMultilabelAll=[]
predProbabilityAll = []

counter = 0

for x in predValid:
    maxProb = x.max()
    predProbabilityAll.append(maxProb)
    
    ind = x.argmax()
    y = [0]*len(x)
    y[ind]=1
    predMultilabelAll.append(y)

    counter +=1


# Convert to int
predClassNum=[]
for x in predMultilabelAll:
    ind = np.array(x).argmax()
    y = ind
    predClassNum.append(y)


# Convert to name
predClassName = []
for f in predClassNum:
     predClassName.append(inverse_target_map[f][:-1])


cl = classification_report(trueClassName, predClassName, output_dict=True)
df = DataFrame(cl).transpose()
df.to_csv (saveDir + r'classification_reports/valid.csv', index = True, header=True) 
df


plt.figure(figsize=(15,15))

cm = confusion_matrix(trueClassName, predClassName)
df = DataFrame(cm)
df.to_csv (saveDir + r'confusion_matrix/valid_total.csv', index = True, header=True) 

hm = heatmap(
 cm, xlabel='Predicted label',
 ylabel='True label', xticklabels=np.unique(trueClassName),
 yticklabels=np.unique(trueClassName), cmap=plt.cm.gray_r, fmt="%d")
plt.title("Total values \n")

plt.colorbar(hm)
plt.gca().invert_yaxis()

plt.savefig(saveDir + 'confusion_matrix/valid_total.svg', transparent = True, bbox_inches='tight')


plt.figure(figsize=(15,15))

cm = confusion_matrix(trueClassName, predClassName)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df = DataFrame(cm)
df.to_csv (saveDir + r'confusion_matrix/valid_normalised.csv', index = True, header=True) 

plt.figure(figsize=(20,20))
cm = heatmap(
 cm, xlabel='Predicted label',
 ylabel='True label', xticklabels=np.unique(trueClassName),
 yticklabels=np.unique(trueClassName), cmap=plt.cm.gray_r)
plt.title("Normalised values\n")

plt.colorbar(cm)
plt.gca().invert_yaxis()

plt.savefig(saveDir + 'confusion_matrix/valid_normalised.svg', transparent = True, bbox_inches='tight')


# Save pred and prob to tsv

df = DataFrame(list(zip(validImgId,trueClassName, predClassName,predProbabilityAll )), 
               columns =['ImgId', 'True', 'Predicted', 'Probability']) 
df = df.set_index('ImgId')

df.to_csv (saveDir + r'predictions/valid.csv', index = True, header=True)


# ## Testset

trueClassNum=[]
for x in y_test:
    ind = np.array(x).argmax()
    y = ind
    trueClassNum.append(y)
    
    
trueClassName = []
for f in trueClassNum:
     trueClassName.append(inverse_target_map[f][:-1])


predMultilabelAll=[]
predProbabilityAll = []

counter = 0

for x in predTest:
    
    maxProb = x.max()
    predProbabilityAll.append(maxProb)
    
    ind = x.argmax()
    y = [0]*len(x)
    y[ind]=1
    predMultilabelAll.append(y)

    counter +=1


# Convert to int
predClassNum=[]
for x in predMultilabelAll:
    ind = np.array(x).argmax()
    y = ind
    predClassNum.append(y)


# Convert to name
predClassName = []
for f in predClassNum:
     predClassName.append(inverse_target_map[f][:-1])


cl = classification_report(trueClassName, predClassName,output_dict=True)
df = DataFrame(cl).transpose()
df.to_csv (saveDir + r'classification_reports/test.csv', index = True, header=True) 
df


plt.figure(figsize=(15,15))

cm = confusion_matrix(trueClassName, predClassName)
df = DataFrame(cm)
df.to_csv (saveDir + r'confusion_matrix/test_total.csv', index = True, header=True) 

hm = heatmap(
 cm, xlabel='Predicted label',
 ylabel='True label', xticklabels=np.unique(trueClassName),
 yticklabels=np.unique(trueClassName), cmap=plt.cm.gray_r, fmt="%d")
plt.title("Total values \n")

plt.colorbar(hm)
plt.gca().invert_yaxis()

plt.savefig(saveDir + 'confusion_matrix/test_total.svg', transparent = True, bbox_inches='tight')


plt.figure(figsize=(15,15))

cm = confusion_matrix(trueClassName, predClassName)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df = DataFrame(cm)
df.to_csv (saveDir + r'confusion_matrix/test_normalised.csv', index = True, header=True) 

plt.figure(figsize=(20,20))
cm = heatmap(
 cm, xlabel='Predicted label',
 ylabel='True label', xticklabels=np.unique(trueClassName),
 yticklabels=np.unique(trueClassName), cmap=plt.cm.gray_r)
plt.title("Normalised values\n")

plt.colorbar(cm)
plt.gca().invert_yaxis()

plt.savefig(saveDir + 'confusion_matrix/test_normalised.svg', transparent = True, bbox_inches='tight')


# Save pred and prob to tsv

df = DataFrame(list(zip(validImgId,trueClassName, predClassName,predProbabilityAll )), 
               columns =['ImgId', 'True', 'Predicted', 'Probability']) 
df = df.set_index('ImgId')

df.to_csv (saveDir + r'predictions/test.csv', index = True, header=True)


modelName        = "Modelname: " + base_model.name
trained          = "Pre-Trained: " + str(preTrained)
overallRuntime   = "Overall runtime: " + str(sum.get_values()[0]) + "s"
runtimePerEpoch  = "Avg. runtime per Epoch: " + str(avg.get_values()[0]) + "s"
dsImg            = "Dataset image: " + imgDir
dsTrain          = "Dataset train: " + trainTsv
dsValid          = "Dataset validation: " + validTsv
dsTest           = "Dataset test: " + testTsv
testAcc          = "Accuracy testset: " + str(acc)
testLoss         = "Loss testset: " + str(loss)
numEpochs        = "Num. Epochs: " + str(len(history.epoch))
earlyStop        = "Early stop (0 if it didn't stop early): " + str(earlyStopping_callback.stopped_epoch)
    


with open(saveDir + 'README.txt','w') as out:
    out.write('{}\n{}\n\n{}\n{}\n\n{}\n{}\n{}\n\n{}\n{}\n{}\n{}\n{}\n'.format(modelName,
                                                                          trained,
                                                                          testAcc,
                                                                          testLoss,
                                                                          numEpochs,
                                                                          overallRuntime,
                                                                          runtimePerEpoch,
                                                                          dsImg,
                                                                          dsTrain,
                                                                          dsValid,
                                                                          dsTest,
                                                                          earlyStop
                                                                         ))

earlyStopping_callback.stopped_epoch
