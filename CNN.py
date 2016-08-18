from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
import numpy as np
from keras.layers.core import MaxoutDense
from sklearn import cross_validation
import os
from PIL import Image
import cPickle
from scipy import stats
import sys
sys.setrecursionlimit(100000)
def load_data():
    data = np.empty((50000,1,48,48),dtype="float32")
    label = np.empty((50000,),dtype="float32")
    imgs = os.listdir("./train_images/")
    num = len(imgs)
    count = 0
    raw_Data = np.load('train_inputs.npy')
    mean = np.mean(raw_Data)
    raw_Data = raw_Data - mean
    for i in range(50000):
        arr = raw_Data[i]
        data[i,0,:,:] = arr.reshape(48,48)
        if i%5000 == 0:
            count += 10
            print(str(count)+'%complete')
    label = np.load("train_outputs.npy")
    return data,label,mean

Train_X, Train_Y, mean = load_data()
Train_Y = np_utils.to_categorical(Train_Y, 10)

model = Sequential()
model.input_shape = (1,48, 48)
model.add(Activation("relu"))
model.add(Convolution2D(4, 1, 6, 6, border_mode='valid',))
model.add(Activation("relu"))

model.add(Convolution2D(8,4, 4, 4, border_mode='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))

model.add(Convolution2D(16, 8, 4, 4, border_mode='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(2, 2)))

model.add(Flatten())
model.add(MaxoutDense(1024, 128, init='normal'))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(MaxoutDense(128, 10, init = 'normal'))
model.add(Activation('softmax'))

sgd = SGD(l2=0.0,lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")

model.fit(Train_X, Train_Y, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, show_accuracy=True)
cPickle.dump(model, open("./modelRELU_MO2_1M2.pkl","wb"))
np.save("mean.npy", mean)
'''5 is the best epoch'''