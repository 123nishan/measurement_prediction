import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from final.constant import *


class CustomDataset(Dataset):
    def __init__(self, data):
        to_convert=measurement
        for i in to_convert:
            data[i] =pd.to_numeric(data[i],errors='coerce')
        self.data = torch.FloatTensor(data.values.astype(np.float32))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx][-len(measurement):]
        data_val = self.data[idx][:-len(measurement)]
        return data_val, target
