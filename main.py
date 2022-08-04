# This is a sample Python script.
import numpy as np
import pandas as pd
import dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from constant import *
from nn import Model
# from tqdm import tqdm
import time
from data import handle_data
from sklearn import preprocessing
from pickle import dump
import matplotlib.pyplot as plt


def TrainModel(model, criterion, optimizer, train_loader, test_loader, epochs=1000):
    train_loss = 0.0
    writer = SummaryWriter()
    writer_validation = SummaryWriter()
    valid_loss = 0.0
    trainLosses = []
    validLosses = []
    valid_loss_min = np.Inf
    for i in range(epochs):
        model.train()
        for batch_idx, (features, target) in enumerate(train_loader):
            output, _ = model(features)

            loss = criterion(output, target.unsqueeze(1))
            writer.add_scalar('train', loss, i)
            writer.flush()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # record the average training loss
            # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            # train_loss+=loss.item()*features.size(0)

        #####################
        # Validating the model#
        #####################
        model.eval()
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(test_loader):
                target = target.unsqueeze(1)
                output, _ = model(features)

                target = target.float()
                loss = criterion(output, target)
                writer_validation.add_scalar('valid', loss, i)
                writer_validation.flush()
                # update average validation loss
                valid_loss += loss.item()
                # valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                # valid_loss+=loss.item()*features.size(0)
                # print(batch_idx)

                # if (i == epochs - 1):  # last epoch
                #     if batch_idx == 5:
                #         target_unscaled=output_scaler.inverse_transform(target.detach().numpy())
                #         output_unscaled=output_scaler.inverse_transform(output.detach().numpy())
                #         print(target_unscaled)
                #         print(output_unscaled)  # last batch of last epoch
                    # print(x_scaled.shape)

            #         whole_data=torch.cat((features,target),-1)
            #         print(whole_data.shape)
            # scale_back_target=min_max_scaler.inverse_transform(whole_data.numpy())
            # feature_output=torch.cat((features,output),-1)
            # sacle_back_output=min_max_scaler.inverse_transform(feature_output.detach().numpy())
            # print(scale_back_target)
            # print(sacle_back_output)

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(test_loader)
        trainLosses.append(train_loss)
        validLosses.append(valid_loss)
        print('Epoch {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(i + 1, epochs, train_loss,
                                                                                       valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'neck_model.pt')
            valid_loss_min = valid_loss

    writer.close()
    return trainLosses, validLosses


if __name__ == '__main__':
    data, total_features = handle_data()
    # data=handle_data(normalize=False)
    # rescale back
    # scale_back=min_max_scaler.inverse_transform(x_scaled)
    # data=pd.DataFrame(scale_back, columns=data.columns)

    features = data.columns.tolist()

    target = features[-1:] #get last column

    #features = list(set(features) - set([target]))
    features = features[:-1]

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define scalar

    scaler = preprocessing.StandardScaler()
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    output_scaler = preprocessing.StandardScaler()
    scaler.fit(X_train.values)
    output_scaler.fit(y_train.values)
    X_train_scaled = scaler.transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    #

    y_train_scaled = output_scaler.transform(y_train.values)
    y_test_scaled = output_scaler.transform(y_test.values)

    X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Train_data = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    # Valid_data = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled,columns=y_train.columns)], axis=1)
    Valid_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test_scaled,columns=y_test.columns)], axis=1)

    # scaler.fit(Train_data)
    # Train_data_scaled = scaler.transform(Train_data.values)
    # Valid_data_scaled = scaler.transform(Valid_data.values)
    dump(scaler, open('scaler_neck.pkl', 'wb'))
    dump(output_scaler, open('output_scaler_neck.pkl', 'wb'))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # # fit scaler on training data
    # min_max_scaler.fit(Train_data)
    # train_data_scaled = min_max_scaler.transform(Train_data)
    # test_data_scaled = min_max_scaler.transform(Valid_data)
    # dump(min_max_scaler,open('scaler.pkl','wb'))
    # Train_data = pd.DataFrame(train_data_scaled, columns=Train_data.columns)
    # Valid_data = pd.DataFrame(test_data_scaled, columns=Valid_data.columns)
    # Train_data= pd.DataFrame(Train_data_scaled, columns=Train_data.columns)
    # Valid_data = pd.DataFrame(Valid_data_scaled, columns=Valid_data.columns)
    train_data = dataloader.CustomDataset(Train_data)
    test_data = dataloader.CustomDataset(Valid_data)

    train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=20, shuffle=True)

    model = Model(total_features, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()  # Mean Absolute Error
    # features,target=next(iter(train_loader))
    # print(features)
    # print(target)

    trainLosses, validLosses = TrainModel(model, criterion, optimizer, train_loader, test_loader)
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(validLosses, label='Validation Loss')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('average loss', fontsize=16)
    plt.legend()
    plt.show()

    # total_step = len(train_loader)
    # for epoch in range(5):
    #     for i, (features, labels) in enumerate(train_loader):
    #
    #         #         print(images.shape)
    #         outputs,_ = model(features)
    #         loss = criterion(outputs, labels.unsqueeze(1))
    #         print(outputs)
    #         print(labels)
    #         print("----------------------------------------------------")
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if (i + 1) % 100 == 0:
    #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #                   .format(epoch + 1, 1000, i + 1, total_step, loss.item()))

    # model.eval()  # it is typically used for batchnorm so that it uses moving variance and mean instead of the whole batch
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()
