import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from constant import *

# waist = 'Waist Circumference, Pref (mm)'
# neck='Neck Base Circumference (mm)'
# hip= 'Hip Circumference, Maximum (mm)'
# chest='Chest Circumference (mm)'
# crotch_height='Crotch Height (mm)'
class CustomDataset(Dataset):
    def __init__(self, data):
        to_convert=[height,weight,shoe_size,neck,female_chest,waist,crotch_height]
        for i in to_convert:
            data[i] =pd.to_numeric(data[i],errors='coerce')
        self.data = torch.FloatTensor(data.values.astype(np.float32))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx][-5:]
        data_val = self.data[idx][:-5]
        return data_val, target
