import pandas as pd
from torch.utils.data import DataLoader

from constant import *
import multiple_dataloader
from sklearn import preprocessing
from pickle import dump

def handle_data(normalize=True):
    demographic = [subject_number, gender, age, height, weight, shoe_size]
    mearuments = [subject_number, chest, neck, waist,hip,crotch_height]
    italy_demographic = [subject_number, gender, age, height, weight, shoe_size_italy]
    demographic_path = "./dutch/demographic_metric.csv"
    measurement_path = "./dutch/measurement_metric.csv"
    italy_demographic_path = "./italy/ItalyDemographics_csv.csv"
    italy_measurement_path = "./italy/ItalyMeasurements_csv.csv"

    demographic_data = pd.read_csv(demographic_path, skipinitialspace=True, usecols=demographic)
    measurement_data = pd.read_csv(measurement_path, skipinitialspace=True, usecols=mearuments)
    demographic_data = demographic_data.loc[demographic_data[gender] == 'Male']  # shape  Male (567,6) Female (700,6)

    data = demographic_data.merge(measurement_data, on=subject_number, how='left')  # shape (567, *)

    data = data.drop(subject_number, axis=1)

    data = data.drop(gender, axis=1)

    data.dropna(axis=0, how='any', inplace=True)  # shape (564,*)
    data = data[data[height].str.contains("No Response") == False]
    data = data[data[weight].str.contains("No Response") == False]
    data = data[data[shoe_size].str.contains("No Response") == False]  # shape (560,*)
    italy_demographic_data = pd.read_csv(italy_demographic_path, skipinitialspace=True, usecols=italy_demographic)
    italy_measurement_data = pd.read_csv(italy_measurement_path, skipinitialspace=True, usecols=mearuments)
    italy_measurement_data = italy_measurement_data[[subject_number, waist, chest, neck, hip, crotch_height]]
    italy_demographic_data = italy_demographic_data.loc[italy_demographic_data[gender] == 'Male']

    italy_demographic_data.columns = [subject_number, age, gender, height, weight, shoe_size]
    italy_data = italy_demographic_data.merge(italy_measurement_data, on=subject_number, how='left')
    italy_data = italy_data[italy_data[shoe_size].str.contains("44 or Larger|35 or Smaller") == False]
    italy_data = italy_data.drop(subject_number, axis=1)
    italy_data = italy_data.drop(gender, axis=1)
    italy_data.dropna(axis=0, how='any', inplace=True)  # shape (564,*)

    frames = [data, italy_data]
    result = pd.concat(frames)

    output=5
    total_features = result.shape[1] - output  # -1 because of target

    return result, total_features,output
    # return data


def handle_split_data():





    X_train=pd.read_csv("./split_data/X_train.csv",skipinitialspace=True,usecols=demographic_male)
    y_train=pd.read_csv("./split_data/y_train.csv",skipinitialspace=True,usecols=measurement)

    X_val=pd.read_csv("./split_data/X_val.csv",skipinitialspace=True,usecols=demographic_male)
    y_val=pd.read_csv("./split_data/y_val.csv",skipinitialspace=True,usecols=measurement)

    X_test=pd.read_csv("./split_data/X_test.csv",skipinitialspace=True,usecols=demographic_male)
    y_test=pd.read_csv("./split_data/y_test.csv",skipinitialspace=True,usecols=measurement)

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

    train_data = multiple_dataloader.CustomDataset(Train_data)
    val_data = multiple_dataloader.CustomDataset(Valid_data)
    test_data = multiple_dataloader.CustomDataset(Test_data)

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    dump(scaler, open('multiple_scaler.pkl', 'wb'))
    dump(output_scaler, open('multiple_output_scaler.pkl', 'wb'))

    return train_loader,val_loader,test_loader,demographic_male,measurement,scaler,output_scaler

