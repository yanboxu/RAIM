import argparse
import numpy as np 
import keras
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense 
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, TimeDistributed, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
# from file_logger import FileLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()

class MetricsHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
                    file_logger.write([str(epoch),
                                       str(logs['loss']),
                                       str(logs['val_loss']),
                                       str(logs['acc']),
                                       str(logs['val_acc'])])




def cnn_model():
    input_length =120*3750 
    num_classes = 2
    ## conv model
    model = Sequential()
    model.add(Conv1D(256,
                 input_shape=[input_length, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    model.add(Conv1D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=4, strides=None))
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return  model

