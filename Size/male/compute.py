import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import load
from nn import Model
# from tqdm import tqdm
from Size.male.size_data import handle_split_data


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
            torch.save(model.state_dict(), 'size_male_dutch.pt')
            valid_loss_min = valid_loss

    # writer.close()
    return trainLosses, validLosses, hipLosses

def test_model(test_loader,features_column,target_column):
    print("Testing")

    test_model= Model(len(features_column),len(target_column))
    test_model.load_state_dict(torch.load("size_male_dutch.pt"))
    test_model.eval()
    test_loss = 0.0
    avg_values = []
    all_pred = []
    all_target = []
    output_scaler = load(open('../size_output_scaler_dutch.pkl', 'rb'))
    with torch.no_grad():
        for batch_idx, (features, target) in enumerate(test_loader):
            output, _ = test_model(features)
            target = target.float()
            output = output_scaler.inverse_transform(output.detach().numpy())

            error = np.absolute(np.subtract(output, target.detach().numpy()))
            all_pred.append(output.tolist())
            all_target.append(target.detach().numpy().tolist())

            avg_values.append((np.average(error, axis=0)).tolist())
            # print(output)
            # loss = criterion(output, target)
            #
            #
            # test_loss += loss.item()

            # print(output)
            # print(output[:, 0]) #output of column index 0
            # print("------------------------------------------------------")
            # loss_hip=criterion(output[:,0], target[:,0])
            # hip_loss += sum(output[:, 0])
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

    avg_values = np.array(avg_values)
    avg_values = avg_values.mean(axis=0)
    twoD_pred = [elem for twod in all_pred for elem in twod]
    twoD_target = [elem for twod in all_target for elem in twod]
    # print(twoD_pred[0])
    twoD_pred=np.array(twoD_pred)
    twoD_target = np.array(twoD_target)

    # outerInseam=np.subtract(twoD_pred[:,7],twoD_pred[:,10])
    # innerInseam=np.subtract(twoD_pred[:,4],twoD_pred[:,11])
    # outerInseam=np.reshape(outerInseam,(-1,1))
    # innerInseam=np.reshape(innerInseam,(-1,1))

    #
    # print(twoD_target)
    # print(twoD_pred)
    # avg_error = output_scaler.inverse_transform(np.array(avg_values))
    print("avg_values", avg_values )
    # target_column_prediction=target_column.copy()
    # target_column_prediction.append(outer_inseam)
    # target_column_prediction.append(inner_inseam)

    # twoD_pred=np.append(twoD_pred,outerInseam,axis=1)
    #
    # twoD_pred=np.append(twoD_pred,innerInseam,axis=1)
    twoD_pred = twoD_pred
    twoD_target = twoD_target

    # with open('male_pred(measured_weight)_without_shape.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #
    #     # write the header
    #     writer.writerow(target_column_prediction)
    #
    #     # write multiple rows
    #     writer.writerows(twoD_pred)
    #
    # with open('male_target(measured_weight)_without_shape.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #
    #     # write the header
    #     writer.writerow(target_column)
    #
    #     # write multiple rows
    #     writer.writerows(twoD_target)

if __name__ == '__main__':

    train_loader,val_loader,test_loader,features,target,scaler,output_scaler=handle_split_data("dutch")



    model=Model(len(features),len(target))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.L1Loss()  # Mean Absolute Error

    #figure, axis = plt.subplots(1, 2, figsize=(10, 10))


    trainLosses, validLosses, hiplosses = TrainModel(model, criterion, optimizer, train_loader, val_loader)


    # test_model(X_test, y_test, scaler, output_scaler,model,criterion)
    test_model(test_loader,features,target)
    # axis[0].plot(trainLosses, label="Training Loss")
    # axis[0].plot(validLosses, label="Validation Loss")
    # axis[1].plot(hiplosses,label="Hip Loss")
    # axis[0].xlabel("Epochs")
    # axis[0].ylabel("Loss")
    # plt.plot(trainLosses, label='Training Loss')
    # plt.plot(validLosses, label='Validation Loss')

    #
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('average loss', fontsize=16)
    plt.plot (trainLosses, label='Training Loss')
    plt.plot (validLosses, label='Validation Loss')
    plt.legend()
    plt.show()

