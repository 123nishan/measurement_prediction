import pandas as pd
from torch.utils.data import DataLoader

from final.constant import *
import dataloader
from sklearn import preprocessing
from pickle import dump




def handle_split_data(demographic_column,measurement_column):
    X_train=pd.read_csv("./split_data/X_train.csv",skipinitialspace=True,usecols=demographic_column)
    y_train=pd.read_csv("./split_data/y_train.csv",skipinitialspace=True,usecols=measurement_column)

    X_val=pd.read_csv("./split_data/X_val.csv",skipinitialspace=True,usecols=demographic_column)
    y_val=pd.read_csv("./split_data/y_val.csv",skipinitialspace=True,usecols=measurement_column)

    X_test=pd.read_csv("./split_data/X_test.csv",skipinitialspace=True,usecols=demographic_column)
    y_test=pd.read_csv("./split_data/y_test.csv",skipinitialspace=True,usecols=measurement_column)

    scaler=preprocessing.StandardScaler()
    output_scaler=preprocessing.StandardScaler()

    scaler.fit(X_train.values)
    output_scaler.fit(y_train.values)

    X_train_scaled=scaler.transform(X_train.values)
    y_train_scaled=output_scaler.transform(y_train.values)

    X_val_scaled=scaler.transform(X_val.values)
    y_val_scaled=output_scaler.transform(y_val.values)

    X_test_scaled=scaler.transform(X_test.values)
    # y_test_scaled=output_scaler.transform(y_test.values)

    X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled_data = pd.DataFrame(X_val_scaled, columns=X_train.columns)
    X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled, columns=y_train.columns)], axis=1)
    Valid_data = pd.concat([X_val_scaled_data, pd.DataFrame(y_val_scaled, columns=y_train.columns)], axis=1)
    Test_data= pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_train.columns)], axis=1)

    train_data = dataloader.CustomDataset(Train_data)
    val_data = dataloader.CustomDataset(Valid_data)
    test_data = dataloader.CustomDataset(Test_data)

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    dump(scaler, open('multiple_scaler.pkl', 'wb'))
    dump(output_scaler, open('multiple_output_scaler.pkl', 'wb'))

    return train_loader,val_loader,test_loader,demographic_column,measurement_column,scaler,output_scaler

