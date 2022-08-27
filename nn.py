import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.fc1=nn.Linear(input_size,10)
        self.fc1_bn=nn.BatchNorm1d(10)
        self.fc2=nn.Linear(10,10)
        self.fc2_bn=nn.BatchNorm1d(10)

        self.fc3 = nn.Linear(10, 10)
        self.fc3_bn = nn.BatchNorm1d(10)

        # self.fc3=nn.Linear(8,8)
        self.fc4=nn.Linear(10,output_size)
        self.relu=nn.ReLU()
        #define dropout
        self.dropout=nn.Dropout(0.25)


    def get_weights(self):
        return self.weight

    def forward(self,x):

        batch_size=x.shape[0]
        x=x.view(batch_size,-1)
        x=self.fc1(x)
        h_1 = self.relu(self.fc1_bn(x))
        # h_1=self.relu(self.fc1(x))

        #h_1 = self.dropout(h_1)
        x=self.fc2(h_1)
        h_2=self.relu(self.fc2_bn(x))
        #h_2=self.dropout(h_2)
        x=self.fc3(h_2)
        h_3=self.relu(self.fc3_bn(x))
        out=self.fc4(h_3)
        return out,h_3

