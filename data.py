import pandas as pd
from constant import *
from sklearn import preprocessing

def handle_data(normalize=True):
    demographic=[subject_number,gender,age, height, weight, shoe_size]
    mearuments=[subject_number,waist,neck,chest]
    italy_demographic = [subject_number, gender, age, height, weight, shoe_size_italy]
    demographic_path = "./dutch/demographic_metric.csv"
    measurement_path = "./dutch/measurement_metric.csv"

    italy_demographic_path = "./italy/ItalyDemographics_csv.csv"
    italy_measurement_path = "./italy/ItalyMeasurements_csv.csv"

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

    italy_demographic_data = pd.read_csv(italy_demographic_path, skipinitialspace=True, usecols=italy_demographic)
    italy_measurement_data = pd.read_csv(italy_measurement_path, skipinitialspace=True, usecols=mearuments)
    italy_measurement_data = italy_measurement_data[[subject_number, waist, chest, neck, hip, crotch_height]]
    italy_demographic_data = italy_demographic_data.loc[italy_demographic_data[gender] == 'Male']

    italy_demographic_data.columns = [subject_number, age, gender, height, weight, shoe_size]
    italy_data = italy_demographic_data.merge(italy_measurement_data, on=subject_number, how='left')

    italy_data = italy_data.drop(subject_number, axis=1)
    italy_data = italy_data.drop(gender, axis=1)
    frames = [data, italy_data]
    result = pd.concat(frames)




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

   
        
        
    total_features=result.shape[1]-1 #-1 because of target
    print(result.shape)
   
    return result,total_features,output
    #return data


# def standing():
#     features=["Chest Ht Stand (mm)",""]
#     standing_path = "./data/standing_metric.csv"
#     standing_data=pd.read_csv(standing_path,skipinitialspace=True,usecols=[subject_number,standing])



