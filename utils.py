import torch.nn as nn
import torch
import numpy as np
from constant import *
from pickle import load
import csv
import optuna
class Engine():
    def __init__(self,model,optimizer,criterion,device):
        self.model = model
        self.optimizer = optimizer
        self.criterion=criterion
        self.device = device
    @staticmethod
    def loss_fn(target,outputs):
        # criterion=nn.L1Loss()
        # criterion=criterion.cuda()
        return nn.L1Loss()(outputs,target)

    def train(self,data_loader):
        self.model.train()
        final_loss = 0
        hip_loss = 0.0

        for batch_idx, (features, targets) in enumerate(data_loader):
            self.optimizer.zero_grad()

            features=features.to(self.device)
            targets=targets.to(self.device)
            # inputs = data["x"]
            # targets = data["y"]

            outputs = self.model(features)

            hip_loss += sum(outputs[:, 0])
            # loss = self.loss_fn(targets,outputs)
            loss=self.criterion(outputs,targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader),hip_loss / len(data_loader)

    def eval(self,data_loader):
        # self.model.trai()
        self.model.eval()

        final_loss = 0
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_loader):
                # self.optimizer.zero_grad()
                features=features.to(self.device)
                targets=targets.to(self.device)
                # inputs = data["x"]
                # targets = data["y"]
                outputs = self.model(features)
                # loss = self.loss_fn(targets,outputs)
                loss=self.criterion(outputs,targets)
                # loss.backward()
                # self.optimizer.step()
                final_loss += loss.item()
        return final_loss / len(data_loader)



