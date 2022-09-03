import torch.nn as nn
import torch
import numpy as np
from final.constant import *
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
            loss=self.criterion(targets,outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader),hip_loss / len(data_loader)

    def eval(self,data_loader):
        # self.model.trai()
        final_loss = 0
        for batch_idx, (features, targets) in enumerate(data_loader):
            # self.optimizer.zero_grad()
            features=features.to(self.device)
            targets=targets.to(self.device)
            # inputs = data["x"]
            # targets = data["y"]
            outputs = self.model(features)
            # loss = self.loss_fn(targets,outputs)
            loss=self.criterion(targets,outputs)
            # loss.backward()
            # self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def test(self,test_loader,target_column):
        print("Testing")

        # test_model= Model(len(features_column),len(target_column))
        # test_model.load_state_dict(torch.load("multiple_model_with_shape(measured).pt"))
        test_model = torch.load("multiple_model_with_shape(measured).pt")
        test_model.eval()
        test_loss = 0.0
        avg_values = []
        all_pred = []
        all_target = []
        output_scaler = load(open('multiple_output_scaler.pkl', 'rb'))
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(test_loader):
                output = test_model(features)
                target = target.float()
                output = output_scaler.inverse_transform(output.detach().numpy())

                error = np.absolute(np.subtract(output, target.detach().numpy()))
                all_pred.append(output.tolist())
                all_target.append(target.detach().numpy().tolist())

                avg_values.append((np.average(error, axis=0)).tolist())

        avg_values = np.array(avg_values)
        avg_values = avg_values.mean(axis=0)
        twoD_pred = [elem for twod in all_pred for elem in twod]
        twoD_target = [elem for twod in all_target for elem in twod]
        # print(twoD_pred[0])
        twoD_pred = np.array(twoD_pred)
        twoD_target = np.array(twoD_target)

        outerInseam = np.subtract(twoD_pred[:, 7], twoD_pred[:, 10])
        innerInseam = np.subtract(twoD_pred[:, 4], twoD_pred[:, 11])
        outerInseam = np.reshape(outerInseam, (-1, 1))
        innerInseam = np.reshape(innerInseam, (-1, 1))

        #
        # print(twoD_target)
        # print(twoD_pred)
        # avg_error = output_scaler.inverse_transform(np.array(avg_values))
        print("avg_values", avg_values / 10)
        target_column_prediction = target_column.copy()
        target_column_prediction.append(outer_inseam)
        target_column_prediction.append(inner_inseam)

        twoD_pred = np.append(twoD_pred, outerInseam, axis=1)

        twoD_pred = np.append(twoD_pred, innerInseam, axis=1)
        twoD_pred = twoD_pred / 10
        twoD_target = twoD_target / 10
        with open('male_pred_with_shape(measured).csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(target_column_prediction)

            # write multiple rows
            writer.writerows(twoD_pred)

        with open('male_target_with_shape(measured).csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(target_column)

            # write multiple rows
            writer.writerows(twoD_target)

