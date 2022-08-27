import pandas as pd
from torch.utils.data import DataLoader

from constant import *
import female_dataloader
from sklearn import preprocessing
from pickle import dump

def handle_split_data():
    # demographic = [age, height, weight, shoe_size]
    # measurement = [waist, female_chest, neck, hip, crotch_height]
    X_train = pd.read_csv("X_train.csv", skipinitialspace=True, usecols=demographic)
    y_train = pd.read_csv("y_train.csv", skipinitialspace=True, usecols=female_measurement)

    X_val = pd.read_csv("X_val.csv", skipinitialspace=True, usecols=demographic)
    y_val = pd.read_csv("y_val.csv", skipinitialspace=True, usecols=female_measurement)

    X_test = pd.read_csv("X_test.csv", skipinitialspace=True, usecols=demographic)
    y_test = pd.read_csv("y_test.csv", skipinitialspace=True, usecols=female_measurement)

    scaler = preprocessing.StandardScaler()
    output_scaler = preprocessing.StandardScaler()

    scaler.fit(X_train.values)
    output_scaler.fit(y_train.values)

    X_train_scaled = scaler.transform(X_train.values)
    y_train_scaled = output_scaler.transform(y_train.values)

    X_val_scaled = scaler.transform(X_val.values)
    y_val_scaled = output_scaler.transform(y_val.values)

    X_test_scaled = scaler.transform(X_test.values)
    # y_test_scaled=output_scaler.transform(y_test.values)

    X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled_data = pd.DataFrame(X_val_scaled, columns=X_train.columns)
    X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled, columns=y_train.columns)], axis=1)
    Valid_data = pd.concat([X_val_scaled_data, pd.DataFrame(y_val_scaled, columns=y_train.columns)], axis=1)
    Test_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_train.columns)], axis=1)

    train_data = female_dataloader.CustomDataset(Train_data)
    val_data = female_dataloader.CustomDataset(Valid_data)
    test_data = female_dataloader.CustomDataset(Test_data)

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    dump(scaler, open('multiple_scaler_female.pkl', 'wb'))
    dump(output_scaler, open('multiple_output_scaler_female.pkl', 'wb'))

    return train_loader, val_loader, test_loader, demographic, female_measurement, scaler, output_scaler



