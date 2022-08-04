import pandas as pd
from constant import *
from sklearn import preprocessing

def handle_data(normalize=True):
    demographic=[subject_number,gender,age, height, weight, shoe_size]
    mearuments=[subject_number,waist,neck,chest]
    demographic_path = "./dutch/demographic_metric.csv"
    measurement_path = "./dutch/measurement_metric.csv"

    demographic_data=pd.read_csv(demographic_path,skipinitialspace=True,usecols=demographic)
    measurement_data=pd.read_csv(measurement_path,skipinitialspace=True,usecols=mearuments)
    measurement_data=measurement_data[[subject_number,chest,neck,waist]]
    demographic_data=demographic_data.loc[demographic_data[gender]=='Male'] #shape  Male (567,6) Female (700,6)

    data = demographic_data.merge(measurement_data, on=subject_number, how='left') #shape (567, *)

    data = data.drop(subject_number, axis=1)

    data=data.drop(gender,axis=1)

    data.dropna(axis=0,how='any',inplace=True) # shape (564,*)
    data=data[data[height].str.contains("No Response")==False]
    data = data[data[weight].str.contains("No Response") == False]
    data = data[data[shoe_size].str.contains("No Response") == False] #shape (560,*)
    # data = data[data[chest].str.contains("No Response") == False]
    # data = data[data[hip].str.contains("No Response") == False]
    # data = data[data[crotch_height].str.contains("No Response") == False]
    # data = data[data[neck].str.contains("No Response") == False]
    # data = data[data[waist].str.contains("No Response") == False]
    # x=data.values
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # data=pd.DataFrame(x_scaled, columns=data.columns)

   
        
        
    total_features=data.shape[1]-1 #-1 because of target
    
   
    return data,total_features
    #return data


# def standing():
#     features=["Chest Ht Stand (mm)",""]
#     standing_path = "./data/standing_metric.csv"
#     standing_data=pd.read_csv(standing_path,skipinitialspace=True,usecols=[subject_number,standing])



