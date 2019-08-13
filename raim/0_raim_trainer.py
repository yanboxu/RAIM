'''
This is RAIM code for training the RAIM models
'''


# RAIM training code

from __future__ import division
import numpy as np
import torch
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import argparse
import glob
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
import math
from raim_att import RNNNet
import sys



def data_loader_fn_train(batch, start, end):
	cnn_featues = []
	dr_data_features = []
	y_data_features = []

	for file in train_files[start:end]:
		cnn_file = file
		cnn_data =np.load(file)

		ff = file.split('/')[-1]
		y_file = './Y_files_DC/' +ff
		y_data = np.load(y_file)

		cnn_featues.append(cnn_data)
		y_data_features.append(y_data[()])

	# ipdb.set_trace()
	cnn_featues1 = np.stack(cnn_featues,axis = 0)

	y_data_features1  = np.stack(y_data_features)

	ecg_features = Variable(torch.from_numpy(cnn_featues1.astype(np.float32)))
	y_target = Variable(torch.from_numpy(y_data_features1.astype(np.float32)))

	return  ecg_features, y_target
 

def data_loader_fn_test(batch, start, end):
	cnn_featues = []
	dr_data_features = []
	y_data_features = []

	for file in test_files[start:end]:
		cnn_file = file
		cnn_data =np.load(file)

		ff = file.split('/')[-1]
		y_file = './Y_files_DC/' +ff
		y_data = np.load(y_file)

		cnn_featues.append(cnn_data)
		y_data_features.append(y_data)

	cnn_featues1 = np.stack(cnn_featues,axis = 0).reshape(48,10,500)
	y_data_features1  = np.stack(y_data_features)

	ecg_features = Variable(torch.from_numpy(cnn_featues1.astype(np.float32)))
	y_target = Variable(torch.from_numpy(y_data_features1.astype(np.float32)))

	return  ecg_features, y_target
 

hidden_size = 200 
batch_size = 10

use_cuda = False

# Device configuration
device = torch.device("cuda" if use_cuda else "cpu")


### model instantiation
# model = model()
model = RNNNet(hidden_size = 200, batch_size = 10).to(device)

train_files = glob.glob('./features/*.npy')
# test_files = glob.glob('/shared/temp/48hr/test/*_CNN_f.npy')



batches = int(math.floor(len(train_files)/batch_size))
epochs = 5
batches =10

## define optimizer and add model parameters to it 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
train_losses = []


# training loop 
for i in range(epochs):
    print('Epoch', i)
    train_loss = 0.0
    train_counter = 0



    start, end = 0, batch_size
    for batch in range(batches-1):


    	# load data from data loader
        ecg_features, y_target = data_loader_fn_train(batch, start, end)


        # convert the loaded data to a specific device
        ecg_features = ecg_features.to(device)
        y_target = y_target.to(device)
        

        # forward pass
        output = model(ecg_features)
        loss = criterion(output, y_target.long())


        #backward pass and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += (loss.data[0] * ecg_features.size(0))
        train_counter += ecg_features.size(0)
        print(train_loss)
        
        start = start+ batch_size
        end   = end +  batch_size
    train_losses.append(train_loss/train_counter)



# Test the model
model.eval()
with torch.no_grad():
	correct = 0 
	total   = 0 

	start, end = 0, batch_size
	for batch in range(batches-1):

	    ecg_features, drug_features, targets = data_loader_fn(batch, start, end)
	    output = model(ecg_features, drug_features)
	    c = (predicted == output).sum().item()
	    correct += (predicted == output).sum().item()
	    total += targets.size(0)

	    # for i, label in enumerate(targets):
	    #     class_correct[label] += c[i]
	    #     class_total[label] += 1

	    start = start+ batch_size
	    end   = end +  batch_size

    print('Accuracy of the model on the test set: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'raim_model.ckpt')


