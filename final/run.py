import numpy as np
import pandas as pd
import multiple_dataloader
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import utils
from final.constant import *
from nn import Model
# from tqdm import tqdm
import time
from final.data import handle_split_data
from sklearn import preprocessing
from pickle import dump
import matplotlib.pyplot as plt
import csv

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
            loss = criterion(target, output)

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
            # torch.save(model.state_dict(), 'multiple_model_with_shape(measured).pt')
            # torch.save(model, 'multiple_model_with_shape(measured).pt')
            torch.save(model, './'+gender+'/results/model.pt')
            valid_loss_min = valid_loss

    # writer.close()
    return trainLosses, validLosses, hipLosses


# def test_model(X_test, y_test, scaler, output_scaler,model,criterion):
#     # Scale
#     print("TESTING")
#     y_test = pd.DataFrame(y_test)
#
#     X_test_scaled = scaler.transform(X_test.values)
#
#     #y_test_scaled = output_scaler.transform(y_test.values)
#
#     X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#     Test_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_test.columns)], axis=1)
#     test_data = multiple_dataloader.CustomDataset(Test_data)
#     test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
#     model.eval()
#     test_loss = 0.0
#     avg_values=[]
#     all_pred=[]
#     all_output=[]
#     with torch.no_grad():
#         for batch_idx, (features, target) in enumerate(test_loader):
#             output, _ = model(features)
#             target = target.float()
#             output = output_scaler.inverse_transform(output.detach().numpy())
#
#             error=np.absolute(np.subtract(output,target.detach().numpy()))
#             all_pred.append(output.tolist())
#             all_output.append(target.detach().numpy().tolist())
#
#             avg_values.append((np.average(error,axis=0)).tolist())
#             # print(output)
#             # loss = criterion(output, target)
#             #
#             #
#             # test_loss += loss.item()
#
#             # print(output)
#             # print(output[:, 0]) #output of column index 0
#             #print("------------------------------------------------------")
#             # loss_hip=criterion(output[:,0], target[:,0])
#             #hip_loss += sum(output[:, 0])
#             # hip_loss+=loss_hip.item()
#             # writer.add_scalar('train', loss, i)
#             # writer.flush()
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             # if batch_idx==4:
#             #     print("size of output:",output.shape)
#             # train_loss += loss.item()
#             # print(train_loss)
#
#     avg_values=np.array(avg_values)
#     avg_values=avg_values.mean(axis=0)
#     twoD_pred=[elem for twod in all_pred for elem in twod]
#     twoD_output=[elem for twod in all_output for elem in twod]
#
#     print(twoD_output)
#     print(twoD_pred)
#     #avg_error = output_scaler.inverse_transform(np.array(avg_values))
#     print("avg_values",avg_values/10)
def test(test_loader,target_column,device):
        print("Testing")

        # test_model= Model(len(features_column),len(target_column))
        # test_model.load_state_dict(torch.load("multiple_model_with_shape(measured).pt"))
        # test_model = torch.load("multiple_model_with_shape(measured).pt")
        test_model = torch.load('./' + gender + '/results/model.pt')

        test_model.eval()
        test_loss = 0.0
        avg_values = []
        all_pred = []
        all_target = []
        output_scaler = load(open('./'+gender.lower()+'/output_scaler.pkl', 'rb'))
        with torch.no_grad():
            for batch_idx, (features, target) in enumerate(test_loader):
                features=features.to(device)
                target=target.to(device)
                output = test_model(features)
                target = target.float()
                output = output_scaler.inverse_transform(output.cpu().detach().numpy())

                error = np.absolute(np.subtract(output, target.cpu().detach().numpy()))
                all_pred.append(output.tolist())
                all_target.append(target.cpu().detach().numpy().tolist())

                avg_values.append((np.average(error, axis=0)).tolist())

        avg_values = np.array(avg_values)
        avg_values = avg_values.mean(axis=0)
        twoD_pred = [elem for twod in all_pred for elem in twod]
        twoD_target = [elem for twod in all_target for elem in twod]
        # print(twoD_pred[0])
        twoD_pred = np.array(twoD_pred)
        twoD_target = np.array(twoD_target)

        if gender.lower()=='female':
            outerInseam = np.subtract(twoD_pred[:, 8], twoD_pred[:, 11])
            innerInseam = np.subtract(twoD_pred[:, 5], twoD_pred[:, 12])
            outerInseam = np.reshape(outerInseam, (-1, 1))
            innerInseam = np.reshape(innerInseam, (-1, 1))
        else:
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
        with open('./'+gender+'/results/pred.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(target_column_prediction)

            # write multiple rows
            writer.writerows(twoD_pred)

        with open('./'+gender+'/results/target.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(target_column)

            # write multiple rows
            writer.writerows(twoD_target)
# def test_model(test_loader,features_column,target_column):
#     print("Testing")
#
#     # test_model= Model(len(features_column),len(target_column))
#     # test_model.load_state_dict(torch.load("multiple_model_with_shape(measured).pt"))
#     test_model=torch.load("multiple_model_with_shape(measured).pt")
#     test_model.eval()
#     test_loss = 0.0
#     avg_values = []
#     all_pred = []
#     all_target = []
#     output_scaler = load(open('multiple_output_scaler.pkl', 'rb'))
#     with torch.no_grad():
#         for batch_idx, (features, target) in enumerate(test_loader):
#             output, _ = test_model(features)
#             target = target.float()
#             output = output_scaler.inverse_transform(output.detach().numpy())
#
#             error = np.absolute(np.subtract(output, target.detach().numpy()))
#             all_pred.append(output.tolist())
#             all_target.append(target.detach().numpy().tolist())
#
#             avg_values.append((np.average(error, axis=0)).tolist())
#             # print(output)
#             # loss = criterion(output, target)
#             #
#             #
#             # test_loss += loss.item()
#
#             # print(output)
#             # print(output[:, 0]) #output of column index 0
#             # print("------------------------------------------------------")
#             # loss_hip=criterion(output[:,0], target[:,0])
#             # hip_loss += sum(output[:, 0])
#             # hip_loss+=loss_hip.item()
#             # writer.add_scalar('train', loss, i)
#             # writer.flush()
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             # if batch_idx==4:
#             #     print("size of output:",output.shape)
#             # train_loss += loss.item()
#             # print(train_loss)
#
#     avg_values = np.array(avg_values)
#     avg_values = avg_values.mean(axis=0)
#     twoD_pred = [elem for twod in all_pred for elem in twod]
#     twoD_target = [elem for twod in all_target for elem in twod]
#     # print(twoD_pred[0])
#     twoD_pred=np.array(twoD_pred)
#     twoD_target = np.array(twoD_target)
#
#     outerInseam=np.subtract(twoD_pred[:,7],twoD_pred[:,10])
#     innerInseam=np.subtract(twoD_pred[:,4],twoD_pred[:,11])
#     outerInseam=np.reshape(outerInseam,(-1,1))
#     innerInseam=np.reshape(innerInseam,(-1,1))
#
#     #
#     # print(twoD_target)
#     # print(twoD_pred)
#     # avg_error = output_scaler.inverse_transform(np.array(avg_values))
#     print("avg_values", avg_values / 10)
#     target_column_prediction=target_column.copy()
#     target_column_prediction.append(outer_inseam)
#     target_column_prediction.append(inner_inseam)
#
#     twoD_pred=np.append(twoD_pred,outerInseam,axis=1)
#
#     twoD_pred=np.append(twoD_pred,innerInseam,axis=1)
#     twoD_pred = twoD_pred / 10
#     twoD_target = twoD_target / 10
#     with open('male_pred_with_shape(measured).csv', 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#
#         # write the header
#         writer.writerow(target_column_prediction)
#
#         # write multiple rows
#         writer.writerows(twoD_pred)
#
#     with open('male_target_with_shape(measured).csv', 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#
#         # write the header
#         writer.writerow(target_column)
#
#         # write multiple rows
#         writer.writerows(twoD_target)
def objective(trial):
        params = {
            "num_layers": trial.suggest_int("num_layers", 2, 5),
            "hidden_size": trial.suggest_int("hidden_size", 5, 10),

            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            # 'optimizer':trial.suggst_

        }

        loss=run_training(params,save_model=False)
        return loss
        all_losses=[]
        # for f in range(5):
        #     print("F value is ",f)
        #     temp_loss=run_training(f,params,save_model=False)
        #     all_losses.append(temp_loss)
        # return np.mean(all_losses)
def run_training(params,save_model=False):

    train_loader, val_loader, test_loader, features, target, scaler, output_scaler = handle_split_data(gender.lower(),demographic_col,measurement_col)
    # input_size, output_size, hidden_size = 10, dropout = 0.25, nlayers = 4
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    model = Model(
        input_size=len(features),
        output_size=len(target),
        hidden_size=params["hidden_size"],
        nlayers=params["num_layers"]
    )
    criterion = nn.L1Loss()
    if use_cuda:
        model=model.cuda()
        criterion=criterion.cuda()


    # lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01)
     # Mean Absolute Error

    eng = utils.Engine(model, optimizer,criterion,device)
    # best_loss = np.inf
    valid_loss_min = np.Inf
    trainLosses = []
    validLosses = []
    EPOCHS = 500
    hipLosses = []
    for epoch in range(EPOCHS):
        train_loss, hip_loss = eng.train(train_loader)
        valid_loss = eng.eval(val_loader)
        hipLosses.append(hip_loss)
        trainLosses.append(train_loss)
        validLosses.append(valid_loss)
        print('Epoch {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(epoch + 1, EPOCHS, train_loss,
                                                                                        valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). '.format(valid_loss_min, valid_loss))
            # torch.save(model.state_dict(), 'multiple_model_with_shape(measured).pt')
            if save_model:
                print("SAVING MODEL")
                torch.save(model, './'+gender+'/results/model.pt')
            valid_loss_min = valid_loss

    # eng.test(test_loader, target)
    # plt.title("")
    # plt.xlabel('epochs', fontsize=18)
    # plt.ylabel('average loss', fontsize=16)
    # plt.plot(trainLosses, label='Training Loss')
    # plt.plot(validLosses, label='Validation Loss')
    # plt.legend()
    # plt.show()
    return valid_loss



if __name__ == '__main__':

    gender = input("Enter your Gender(Male/Female): ")
    if gender.lower() == 'male':
        demographic_col = male_inputs_list
        measurement_col = male_output_list
    else:
        demographic_col = female_input_list
        measurement_col = female_output_list

    #
    #

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial_=study.best_trial
    print(trial_.values)
    print(trial_.params)


    run_training(trial_.params, save_model=True)



    #for best parameter save
    # scores=0
    # for j in range(5):
    #     scr=run_training(j,trial_.params,save_model=True)
    #     scores+=scr
    # print(scores/5)

    train_loader, val_loader, test_loader, features, target, scaler, output_scaler = handle_split_data(
        gender.lower(),demographic_col, measurement_col)
    test(test_loader,target,device)
    # run_training(fold=0)
    # train_loader,val_loader,test_loader,features,target,scaler,output_scaler=handle_split_data()
    #
    #
    #
    #
    # model=Model(len(features),len(target))
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.L1Loss()  # Mean Absolute Error
    # eng = utils.Engine(model,optimizer)
    # best_loss= np.inf
    # valid_loss_min = np.Inf
    # trainLosses = []
    # validLosses = []
    # EPOCHS=100
    # hipLosses = []
    # for epoch in range(EPOCHS):
    #     train_loss,hip_loss=eng.train(train_loader)
    #     valid_loss=eng.eval(val_loader)
    #     hipLosses.append(hip_loss)
    #     trainLosses.append(train_loss)
    #     validLosses.append(valid_loss)
    #     print('Epoch {}/{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(epoch + 1, EPOCHS, train_loss,
    #                                                                                    valid_loss))
    #     if valid_loss <= valid_loss_min:
    #         print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
    #         # torch.save(model.state_dict(), 'multiple_model_with_shape(measured).pt')
    #         torch.save(model, 'multiple_model_with_shape(measured).pt')
    #         valid_loss_min = valid_loss
    #
    # eng.test(test_loader,target)
    # #figure, axis = plt.subplots(1, 2, figsize=(10, 10))
    #
    #
    # # trainLosses, validLosses, hiplosses = TrainModel(model, criterion, optimizer, train_loader, val_loader)
    #
    #
    # # test_model(X_test, y_test, scaler, output_scaler,model,criterion)
    # # test_model(test_loader,features,target)
    # # axis[0].plot(trainLosses, label="Training Loss")
    # # axis[0].plot(validLosses, label="Validation Loss")
    # # axis[1].plot(hiplosses,label="Hip Loss")
    # # axis[0].xlabel("Epochs")
    # # axis[0].ylabel("Loss")
    # # plt.plot(trainLosses, label='Training Loss')
    # # plt.plot(validLosses, label='Validation Loss')
    #
    # #
    # plt.xlabel('epochs', fontsize=18)
    # plt.ylabel('average loss', fontsize=16)
    # plt.plot (trainLosses, label='Training Loss')
    # plt.plot (validLosses, label='Validation Loss')
    # plt.legend()
    # plt.show()
