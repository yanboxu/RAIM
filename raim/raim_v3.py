
import argparse
import glob
import pdb
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm


# model definition -- use full attention when there is no guidance 
class RNNNetv1(nn.Module):
    def __init__(self, batch_size, hidden_size):
        super(RNNNetv1, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = nn.LSTM(500, 200, 2, dropout=0.5)
        self.attn = nn.Linear(10, 10)
        self.attn1 = nn.Linear(60,10)

        self.dense_h = nn.Linear(200,1)
        self.softmax = nn.Softmax()
        

        self.hidden2label = nn.Linear(200, 2)
        self.hidden = self.init_hidden()
        self.grucell = nn.GRUCell(500, 200)
        
        self.mlp_for_x      = nn.Linear(500,1, bias=False)
        self.mlp_for_hidden = nn.Linear(200,12,bias=True)
        

    def init_hidden(self):
        return Variable(torch.zeros(self.batch_size, self.hidden_size))


    def forward(self, x, guidance):
        
        print(self.hidden.shape, x[0:1].shape)
        
        for i in range(12):
            
            
            # print('---------starting---------')
            # print(i, x[:,0:i+1,:].shape, guidance[:,i,:].shape)
            
            tt = x[:,0:i+1,:].reshape(self.batch_size, (i+1)* x[:,0:i+1,:].shape[2])
            if i<11:
                padding = torch.zeros(self.batch_size, 6000-tt.shape[1])
                self.temp1 = torch.cat((tt, padding),1)
            else:
                self.temp1 = tt
           
            self.input_padded = self.temp1.reshape(10,12,500)
            # print(self.input_padded.shape, self.hidden.shape, self.temp1.shape, guidance.shape, 'input for mlp')
            
            #### multuply with guidance #######
            temp_guidance = torch.zeros(10,12,1)
            
            temp_guidance[:,0:i+1,:] = guidance[:,0:i+1,:]
          
            if i>0:
               
                zero_idx = np.where(torch.sum(guidance[:,:i,0], dim=1)==0)
#                 print(zero_idx[0])
                if len(zero_idx[0])>0:

                    temp_guidance[zero_idx[0],:i,0] = 1

            temp_guidance[:,i,:] = 1
#             print('temp guidance', temp_guidance)

            self.guided_input = torch.mul(self.input_padded, temp_guidance)
            
            ######### MLP ###########
            self.t1 = self.mlp_for_x(self.guided_input) + self.mlp_for_hidden(self.hidden).reshape(10,12,1)
            
            ######### softmax-> multiply->  context vector ###########
            self.t1_softmax = self.softmax(self.t1)
            final_output = torch.mul(self.input_padded, self.t1_softmax)
            # print('softmax shape',self.t1_softmax.shape, self.input_padded.shape, final_output.shape)

            context_vec = torch.sum(final_output,dim=1)            
            # print('context vec',context_vec.shape, x.shape)
            
            self.hx = self.grucell(context_vec, self.hidden)
            self.hidden = self.hx
            
        y  = self.hidden2label(self.hidden)
        # print('y shape', y.shape, y)
        return self.hidden