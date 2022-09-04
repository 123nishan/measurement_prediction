import pandas as pd
from torch.utils.data import DataLoader
from sklearn import preprocessing
from pickle import dump
import torch
from Size.male import size_constant, size_dataloader


def handle_split_data(country):

    X_train=pd.read_csv("./" + country +"/X_train.csv", skipinitialspace=True, usecols=size_constant.demographic)
    y_train=pd.read_csv("./" + country +"/y_train.csv", skipinitialspace=True, usecols=size_constant.size)

    X_val=pd.read_csv("./" + country +"/X_val.csv", skipinitialspace=True, usecols=size_constant.demographic)
    y_val=pd.read_csv("./" + country +"/y_val.csv", skipinitialspace=True, usecols=size_constant.size)

    X_test=pd.read_csv("./" + country +"/X_test.csv", skipinitialspace=True, usecols=size_constant.demographic)
    y_test=pd.read_csv("./" + country +"/y_test.csv", skipinitialspace=True, usecols=size_constant.size)

    scaler=preprocessing.StandardScaler()
    output_scaler=preprocessing.StandardScaler()

    scaler.fit(X_train.values)
    output_scaler.fit(y_train.values)

    X_train_scaled=scaler.transform(X_train.values)
    #y_train_scaled=output_scaler.transform(y_train.values)

    X_val_scaled=scaler.transform(X_val.values)
    #y_val_scaled=output_scaler.transform(y_val.values)

    X_test_scaled=scaler.transform(X_test.values)
    # y_test_scaled=output_scaler.transform(y_test.values)

    X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled_data = pd.DataFrame(X_val_scaled, columns=X_train.columns)
    X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    # Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled, columns=y_train.columns)], axis=1)
    # Valid_data = pd.concat([X_val_scaled_data, pd.DataFrame(y_val_scaled, columns=y_train.columns)], axis=1)
    # Test_data= pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_train.columns)], axis=1)
    Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train.values, columns=y_train.columns)], axis=1)
    Valid_data = pd.concat([X_val_scaled_data, pd.DataFrame(y_val.values, columns=y_train.columns)], axis=1)
    Test_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_train.columns)], axis=1)

    train_data = size_dataloader.CustomDataset(torch.from_numpy(X_train_scaled_data.values),(y_train.values))
    val_data = size_dataloader.CustomDataset(torch.from_numpy(X_val_scaled_data.values),(y_val.values))
    test_data = size_dataloader.CustomDataset(torch.from_numpy(X_test_scaled_data.values),(y_test.values))

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    dump(scaler, open('../size_scaler_dutch.pkl', 'wb'))
    dump(output_scaler, open('../size_output_scaler_dutch.pkl', 'wb'))

    return train_loader, val_loader, test_loader, size_constant.demographic, size_constant.size, scaler, output_scaler

