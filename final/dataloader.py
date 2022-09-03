import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from final.constant import *


class CustomDataset(Dataset):
    def __init__(self, data,measurement_list):
        self.to_convert=measurement_list
        for i in self.to_convert:
            data[i] =pd.to_numeric(data[i],errors='coerce')
        self.data = torch.FloatTensor(data.values.astype(np.float32))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        target = self.data[idx][-len(self.to_convert):]
        data_val = self.data[idx][:-len(self.to_convert)]
        # target = self.data[idx][-len(measurement_column):]
        # data_val = self.data[idx][:-len(measurement_column)]
        return data_val, target
