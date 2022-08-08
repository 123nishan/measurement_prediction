import numpy as np
import pandas as pd
import multiple_dataloader

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
from multiple_data import handle_data
from sklearn import preprocessing
from pickle import dump
import matplotlib.pyplot as plt


def TrainModel(model, criterion, optimizer, train_loader, test_loader, epochs=1000):
    train_loss = 0.0
    # writer = SummaryWriter()
    # writer_validation = SummaryWriter()
    valid_loss = 0.0
    trainLosses = []
    validLosses = []
    hip_loss = 0.0
    hipLosses = []
    valid_loss_min = np.Inf
    for i in range(epochs):
        model.train()
        for batch_idx, (features, target) in enumerate(train_loader):
            output, _ = model(features)
            # target size: 34,5
            loss = criterion(output, target)

            # print(output[:, 0])
            # print("------------------------------------------------------")
            # loss_hip=criterion(output[:,0], target[:,0])
            hip_loss += sum(output[:, 0])
            # hip_loss+=loss_hip.item()

            # writer.add_scalar('train', loss, i)
            # writer.flush()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch_idx==4:
            #     print("size of output:",output.shape)
            train_loss += loss.item()
            # print(train_loss)

        #####################
        # Validating the model#
        #####################
        model.eval()
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(test_loader):
                # target = target.unsqueeze(1)
                output, _ = model(features)

                target = target.float()
                loss = criterion(output, target)
                # writer_validation.add_scalar('valid', loss, i)
                # writer_validation.flush()
                # update average validation loss
                valid_loss += loss.item() #loss is tensor, so loss.item() is a float
                # valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                # valid_loss+=loss.item()*features.size(0)
                # print(batch_idx)

                # if (i == epochs - 1):  # last epoch
                #     if batch_idx == 5:
                #
                #         target_unscaled=output_scaler.inverse_transform(target.detach().numpy())
                #         output_unscaled=output_scaler.inverse_transform(output.detach().numpy())
                #         print("target",target_unscaled)
                #         print("output",output_unscaled)  # last batch of last epoch

        # calculate average losses
        train_loss = train_loss / len(train_loader) #len(train_loader) is the number of batches

        valid_loss = valid_loss / len(test_loader)
        hip_loss = hip_loss / len(train_loader)
        hipLosses.append(hip_loss)
        trainLosses.append(train_loss)
        validLosses.append(valid_loss)
        print('Epoch {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(i + 1, epochs, train_loss,
                                                                                       valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'multiple_model.pt')
            valid_loss_min = valid_loss

    # writer.close()
    return trainLosses, validLosses, hipLosses


def test_model(X_test, y_test, scaler, output_scaler,model,criterion):
    # Scale
    print("TESTING")
    y_test = pd.DataFrame(y_test)

    X_test_scaled = scaler.transform(X_test.values)

    #y_test_scaled = output_scaler.transform(y_test.values)

    X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    Test_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_test.columns)], axis=1)
    test_data = multiple_dataloader.CustomDataset(Test_data)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    model.eval()
    test_loss = 0.0
    avg_values=[]
    all_pred=[]
    all_output=[]
    with torch.no_grad():
        for batch_idx, (features, target) in enumerate(test_loader):
            output, _ = model(features)
            target = target.float()
            output = output_scaler.inverse_transform(output.detach().numpy())

            error=np.absolute(np.subtract(output,target.detach().numpy()))
            all_pred.append(output.tolist())
            all_output.append(target.detach().numpy().tolist())

            avg_values.append((np.average(error,axis=0)).tolist())
            # print(output)
            # loss = criterion(output, target)
            #
            #
            # test_loss += loss.item()

            # print(output)
            # print(output[:, 0]) #output of column index 0
            #print("------------------------------------------------------")
            # loss_hip=criterion(output[:,0], target[:,0])
            #hip_loss += sum(output[:, 0])
            # hip_loss+=loss_hip.item()
            # writer.add_scalar('train', loss, i)
            # writer.flush()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # if batch_idx==4:
            #     print("size of output:",output.shape)
            # train_loss += loss.item()
            # print(train_loss)

    avg_values=np.array(avg_values)
    avg_values=avg_values.mean(axis=0)
    twoD_pred=[elem for twod in all_pred for elem in twod]
    twoD_output=[elem for twod in all_output for elem in twod]

    print(twoD_output)
    print(twoD_pred)
    #avg_error = output_scaler.inverse_transform(np.array(avg_values))
    print("avg_values",avg_values/10)



if __name__ == '__main__':
    data, total_features,output = handle_data()
    features = data.columns.tolist()

    target = features[-5:]  # get last column

    # features = list(set(features) - set([target]))
    features = features[:-5]

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # define scalar

    scaler = preprocessing.StandardScaler()
    y_train = pd.DataFrame(y_train)
    y_eval = pd.DataFrame(y_eval)
    output_scaler = preprocessing.StandardScaler()
    scaler.fit(X_train.values)
    output_scaler.fit(y_train.values)
    X_train_scaled = scaler.transform(X_train.values)
    X_eval_scaled = scaler.transform(X_eval.values)
    #


    y_train_scaled = output_scaler.transform(y_train.values)
    y_eval_scaled = output_scaler.transform(y_eval.values)

    X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_eval_scaled_data = pd.DataFrame(X_eval_scaled, columns=X_test.columns)

    # Train_data = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    # Valid_data = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled, columns=y_train.columns)], axis=1)
    Valid_data = pd.concat([X_eval_scaled_data, pd.DataFrame(y_eval_scaled, columns=y_test.columns)], axis=1)

    # scaler.fit(Train_data)
    # Train_data_scaled = scaler.transform(Train_data.values)
    # Valid_data_scaled = scaler.transform(Valid_data.values)
    dump(scaler, open('multiple_scaler.pkl', 'wb'))
    dump(output_scaler, open('multiple_output_scaler.pkl', 'wb'))

    train_data = multiple_dataloader.CustomDataset(Train_data)
    eval_data = multiple_dataloader.CustomDataset(Valid_data)

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=32, shuffle=True)

    model = Model(total_features, output)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.L1Loss()  # Mean Absolute Error

    #figure, axis = plt.subplots(1, 2, figsize=(10, 10))
    trainLosses, validLosses, hiplosses = TrainModel(model, criterion, optimizer, train_loader, eval_loader)
    test_model(X_test, y_test, scaler, output_scaler,model,criterion)
    # axis[0].plot(trainLosses, label="Training Loss")
    # axis[0].plot(validLosses, label="Validation Loss")
    # axis[1].plot(hiplosses,label="Hip Loss")
    # axis[0].xlabel("Epochs")
    # axis[0].ylabel("Loss")
    # plt.plot(trainLosses, label='Training Loss')
    # plt.plot(validLosses, label='Validation Loss')
    # plt.xlabel('epochs', fontsize=18)
    # plt.ylabel('average loss', fontsize=16)
    plt.plot (trainLosses, label='Training Loss')
    plt.plot (validLosses, label='Validation Loss')
    plt.legend()
    plt.show()
