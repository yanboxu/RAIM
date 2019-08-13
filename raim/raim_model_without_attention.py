'''
Model definition of RAIM without attention code
'''

import argparse
import glob
import pdb
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


# model definition
class RNNNet(nn.Module):
    def __init__(self, batch_size, hidden_size):
        super(RNNNet, self).__init__()
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


    def forward(self, x):
        
        for i in range(12):            
            tt = x[:,0:i+1,:].contiguous().view(self.batch_size, (i+1)* x[:,0:i+1,:].size()[2])
            if i<11:
                padding = torch.zeros(self.batch_size, 6000-tt.size()[1])
                self.temp1 = torch.cat((tt, padding),1)
            else:
                self.temp1 = tt
           
            self.input_padded = self.temp1.contiguous().view(10,12,500)
            
            
            ######### MLP ###########
            # self.t1 = self.mlp_for_x(self.input_padded) + self.mlp_for_hidden(self.hidden).contiguous().view(10,12,1)
            
            ######### softmax-> multiply->  context vector ###########
            # print(self.t1.size())

            # self.t1_softmax = self.softmax(self.t1)
            # final_output = torch.mul(self.input_padded, self.t1_softmax)

            context_vec = torch.sum(input_padded,dim=1)            
            
            ######### GRUCell ###########
            self.hx = self.grucell(context_vec, self.hidden)
            self.hidden = self.hx
            
        y  = self.hidden2label(self.hidden)
        return y