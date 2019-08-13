'''
script to train CNN and CNN-RNN code
'''

import sys
import pdb
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from file_logger import FileLogger
#from model_data import DataReader
from model_resnet import resnet_34
from models import *
import numpy as np
from sklearn.model_selection import train_test_split
from model2ch import *

class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss']),
                           str(logs['acc']),
                           str(logs['val_acc'])])


if __name__ == '__main__':
    model_name = 'm11'

    args = sys.argv
    if len(args) == 2:
        model_name = args[1].lower()
    print('Model selected:', model_name)
    file_logger = FileLogger('out_{}.tsv'.format(model_name), ['step', 'tr_loss', 'te_loss',
                                                               'tr_acc', 'te_acc'])
    model = None
    num_classes = 5
    if model_name == 'm3':
        model = m3(num_classes=num_classes)
    elif model_name == 'm5':
        model = m5(num_classes=num_classes)
    elif model_name == 'm11':
        model = m11(num_classes=num_classes)
    elif model_name == 'm18':
        model = m18(num_classes=num_classes)
    elif model_name == 'm34':
        model = resnet_34(num_classes=num_classes)

    #model = m_rec(num_classes = num_classes)

    #model = m3_2D(num_classes = num_classes)
    #model = m3_2dv2(num_classes = num_classes)
    model = model6ch(num_classes = 2)

    if model is None:
        exit('Please choose a valid model: [m3, m5, m11, m18, m34]')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # load train and test file
    train_file = ''
    test_file  = ''
    x_tr1 = np.load(train_file)
    y_tr1 = np.load(test_file)


    # x_tr, x_te, y_tr, y_te = train_test_split( x_tr1, y_tr1, test_size=0.1, random_state=0)

    # x_tr1 = x_tr[:,:,0].reshape(x_tr.shape[0], 6000,1)
    # x_tr2 = x_tr[:,:,1].reshape(x_tr.shape[0], 6000,1)
    # x_tr3 = x_tr[:,:,2].reshape(x_tr.shape[0], 6000,1)
    # x_tr4 = x_tr[:,:,3].reshape(x_tr.shape[0], 6000,1)
    # x_tr5 = x_tr[:,:,4].reshape(x_tr.shape[0], 6000,1)
    # x_tr6 = x_tr[:,:,5].reshape(x_tr.shape[0], 6000,1)


    # x_te1 = x_te[:,:,0].reshape(x_te.shape[0], 6000,1)
    # x_te2 = x_te[:,:,1].reshape(x_te.shape[0], 6000,1)
    # x_te3 = x_te[:,:,2].reshape(x_te.shape[0], 6000,1)
    # x_te4= x_te[:,:,3].reshape(x_te.shape[0], 6000,1)
    # x_te5 = x_te[:,:,4].reshape(x_te.shape[0], 6000,1)
    # x_te6 = x_te[:,:,5].reshape(x_te.shape[0], 6000,1)

    # print('x_tr.shape =', x_tr.shape)
    # print('y_tr.shape =', y_tr.shape)
    # print('x_te.shape =', x_te.shape)
    # print('y_te.shape =', y_te.shape)

    #x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1],1,1)
    #x_te = x_te.reshape(x_te.shape[0], x_te.shape[1],1,1)

    filepath="./model_files/raim-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]



    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    metrics_history = MetricsHistory()
    batch_size = 128
    model.fit(x= [x_tr1,x_tr2, x_tr3, x_tr4, x_tr5, x_tr6 ],
              y=y_tr,
              batch_size=batch_size,
              epochs=200,
              verbose=1,
              shuffle=True,
              validation_data=([x_te1, x_te2, x_te3, x_te4, x_te5, x_te6], y_te),
              callbacks=[metrics_history, reduce_lr,checkpoint])
    model.save('./model_files/m5_run1.h5')
    pdb.set_trace()
    file_logger.close()
