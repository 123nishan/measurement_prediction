import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from Size.male.size_constant import *

# waist = 'Waist Circumference, Pref (mm)'
# neck='Neck Base Circumference (mm)'
# hip= 'Hip Circumference, Maximum (mm)'
# chest='Chest Circumference (mm)'
# crotch_height='Crotch Height (mm)'
class CustomDataset(Dataset):
    def __init__(self,X_data,y_data):
        # to_convert=size
        # for i in to_convert:
        #     y_data[i] =pd.to_numeric(y_data[i],errors='coerce')
        #     self.y_data = torch.FloatTensor(y_data.astype(np.float32))
        self.y_data=y_data
        self.X_data=X_data
        # self.y_data=y_data

    def __getitem__(self, index):
        return self.X_data[index],self.y_data[index]

    def __len__(self):
        return len(self.X_data)
    # def __init__(self, data):
    #     to_convert=size
    #     for i in to_convert:
    #         data[i] =pd.to_numeric(data[i],errors='coerce')
    #     self.data = torch.FloatTensor(data.values.astype(np.float32))
    #
    #
    #
    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, idx):
    #
    #     target = self.data[idx][-len(size):]
    #     data_val = self.data[idx][:-len(size)]
    #
    #     return data_val, target
