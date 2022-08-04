import pandas as pd
from constant import *
from sklearn import preprocessing


def handle_data(normalize=True):
    demographic = [subject_number, gender, age, height, weight, shoe_size]
    mearuments = [subject_number, chest, neck, waist,hip,crotch_height]
    demographic_path = "./dutch/demographic_metric.csv"
    measurement_path = "./dutch/measurement_metric.csv"

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

    output=5
    total_features = data.shape[1] - output  # -1 because of target

    return data, total_features,output
    # return data





