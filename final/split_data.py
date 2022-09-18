import sys

import torch
import pandas as pd
from constant import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import re
from base64 import encode

def body_shape(bust, hip, waist, shoulder):
    """
    This function calculates body shape for female using the parameters. .
    NOTE: if want to add more body shape change if condition accordingly and please check code in get_data() METHOD Where
          one hot encoding is applied.
    :param bust: float
    :param hip: float
    :param waist: float
    :param shoulder: float
    :return: string
        The strings values are defined in constant.py file.
    """
    bust = bust / 25.4  # convert mm to inch
    hip = hip / 25.4
    waist = waist / 25.4
    shoulder = shoulder / 25.4

    if (bust - hip) <= 1 and (hip - bust) < 3.6 and (bust - waist) >= 9 or (hip - waist) >= 10:
        return hourglass
    # elif (hip - bust) > 1 and (bust - hip) < 10 and (bust - waist) >= 9:
    #     return top_hourglass
    elif (hip - bust) >= 3.6 and (hip - waist) < 9:
        return triangle
    elif (bust - hip) >= 3.6 and (bust - waist) < 9:
        return inverted_triangle
    elif (hip - bust) < 3.6 and (bust - hip) < 3.6 and (bust - waist) < 9 and (hip - waist) < 10:
        return rectangle


def check_ratio(upper_chest, waist):
    """
    This function calculates the ratio of upper chest and waist. The ratio is than used to categorise male body shape.
    NOTE: if want o add more body shape change if condition accordingly and please check code in get_data() METHOD Where
          one hot encoding is applied.
    :param upper_chest: float
    :param waist: float
    :return: dataframe without header containing two column
        - e.g. [rectangle, 1] here rectangle is constant and 1 is the ratio of chest and wasit
    """
    ratio = upper_chest / waist

    if ratio >= 1.11 and ratio <= 1.17:  # 1.11 33th percentile, 66 th percentile 1.17
        return pd.Series([rectangle, ratio])
    elif ratio < 1.11:
        return pd.Series([triangle, ratio])
    else:
        return pd.Series([inverted_triangle, ratio])

def get_data(Gender,body_shape_constraint=True):
    """
    This function read the data from csv file and few data pre-processing techniques is applied,
        such as feature engineering, calculating body shape and target value for inner inseam and outer inseam,
        removal of null values or any string that the network cannot take as input (e.g. few row contains string like 46 or larger).
        The csv is filtered from gender parameter.
        It merges dutch and italy dataframe and at the need the dataset is splited into training, test and validation dataset and saved into
        male folder if gender is male and in female folder if gender is female.

    :param Gender: String
        For male the string value will be 'male' and for female the string value will be 'female'.
    :param body_shape_constraint: boolean
        You can ignore this boolean, this was used to test.
    :return: boolean
        It always returns true. It was added just to check the function process.
    """
    if Gender.lower()=='female':
        measurement_col=female_measurement_column
        italy_measurements_col=female_italy_measurements_column
    else:
        measurement_col=measurement_column
        italy_measurements_col=italy_measurements_column

    feature_size = len(extracted_column) + len(measurement_col) - 2

    demographic_data=pd.read_csv(dutch_demographic_path,skipinitialspace=True,usecols=demographic_column)
    measurement_data=pd.read_csv(dutch_measurement_path,skipinitialspace=True,usecols=measurement_col)
    measurement_data=measurement_data[measurement_col]

    demographic_data=demographic_data.loc[(demographic_data[gender]).str.lower()==Gender.lower()] #Filters csv data using gender parameter.
    dutch_df=demographic_data.merge(measurement_data,on=subject_number,how='left')

    dutch_df=dutch_df.drop(gender,axis=1)

    extracted_data=pd.read_csv(dutch_extracted_path,skipinitialspace=True,usecols=extracted_column)
    dutch_df=dutch_df.merge(extracted_data,on=subject_number,how='left') # merge two different dataframe (csv), demographic and measurement.

    #CLEAN DATA
    dutch_df.dropna(axis=0,how='any',inplace=True) #removing any row with empty column
    dutch_df = dutch_df[dutch_df[height].str.contains("No Response") == False] # removing any rwo with stirng value ' No response'
    dutch_df = dutch_df[dutch_df[weight].str.contains("No Response") == False]
    dutch_df = dutch_df[dutch_df[shoe_size].str.contains("No Response") == False]

    features=dutch_df.columns.tolist()
    target=[]
    if Gender.lower()=='male': # please dont not change the number below. Extracting columns for target and features
        target=features[-13:]
        features=features[:-13]



    #CONVERT object to INT/Float
    dutch_df[height] = pd.to_numeric(dutch_df[height])
    dutch_df[weight] = pd.to_numeric(dutch_df[weight])
    dutch_df[shoe_size] = pd.to_numeric(dutch_df[shoe_size])

   #ITALY

    italy_demographic_data = pd.read_csv(italy_demographic_path, skipinitialspace=True, usecols=italy_demographic_column)
    italy_measurement_data = pd.read_csv(italy_measurement_path, skipinitialspace=True, usecols=italy_measurements_col)
    italy_measurement_data.rename(columns={italy_upper_chest: upper_chest}, inplace=True)
    italy_measurement_data = italy_measurement_data[measurement_col]
    italy_demographic_data = italy_demographic_data.loc[(italy_demographic_data[gender]).str.lower() == Gender.lower()]

    italy_demographic_data.columns = [subject_number, age, gender, height, weight, shoe_size]
    italy_df = italy_demographic_data.merge(italy_measurement_data, on=subject_number, how='left')
    italy_extracted = pd.read_csv(italy_extracted_path, skipinitialspace=True, usecols=extracted_column)

    italy_df = italy_df.merge(italy_extracted, on=subject_number, how='left')
    # italy_data=italy_data.drop(subject_number,axis=1)

    italy_df = italy_df.drop(gender, axis=1)
    #Remove object/String from shoe size column
    italy_df = italy_df[italy_df[shoe_size].str.contains("44 or Larger|35 or Smaller") == False]
    if Gender.lower()=='female':
        italy_df = italy_df[italy_df[shoe_size].str.contains("Don't Know") == False]
    italy_df.dropna(axis=0, how='any', inplace=True)
    italy_df[shoe_size] = pd.to_numeric(italy_df[shoe_size])

    combined_df = [dutch_df, italy_df]
    combined_df = pd.concat(combined_df)

    if body_shape_constraint and (Gender.lower())=='male':

        X = combined_df[features] # Extracting all columns for features
        y = combined_df[target] #Extracting all columns for target

        chest_waist = y[[upper_chest, waist]]
        # loop through each row and check ratio
        chest_waist[['shape', 'ratio']] = chest_waist.apply(lambda row: check_ratio(row[upper_chest], row[waist]),
                                                            axis=1) # Calls check_ration method for each row for male and add the return value in dataframe.
        #one hot encoder
        encoder = preprocessing.OneHotEncoder().fit_transform(chest_waist['shape'].values.reshape(-1, 1)).toarray() #one hot encoder for check_wasit dataframe
        '''
        e.g. of one hot encode
            person=[
            {'23','B'},
            {'25,'A'},
            {'26','C'}
            ]
        Lets say we have a list with 3 person detail which contains age and their visa type (A,B,C)
        As neural network require inputs to be in number so we use one hot encoder, The end result of one hot encoder will be:
            person=[
            {23,0,1,0},
            {25,1,0,0},
            {26,0,0,1}
            ]
            Here the  index 1 represents A,  index 2 for B and index 3 for C
        '''
        # extracting encoded value
        chest_waist[inverted_triangle] = encoder[:, 0]
        chest_waist[rectangle] = encoder[:, 1]
        chest_waist[triangle] = encoder[:, 2]
        chest_waist = chest_waist.drop(upper_chest, axis=1)
        chest_waist = chest_waist.drop(waist, axis=1)
        X = pd.concat([X, chest_waist], axis=1, join='inner')

    elif body_shape_constraint and Gender.lower()=='female':
        data = combined_df[[chest, hip, waist, shoulder_breadth]]
        data['shape'] = data.apply(lambda row: body_shape(row[chest], row[hip], row[waist], row[shoulder_breadth]),axis=1) # This calls body_shape method for all female dataset row to calculate body shape
        encoder = preprocessing.OneHotEncoder().fit_transform(data['shape'].values.reshape(-1, 1)).toarray()
        # convert list to dataframe
        # print(encoder)
        # Same as discussed for male above
        data[hourglass] = encoder[:, 0]
        data[inverted_triangle] = encoder[:, 1]
        data[rectangle] = encoder[:, 2]
        data[triangle] = encoder[:, 3]

        data = data.drop(chest, axis=1)
        data = data.drop(waist, axis=1)
        data = data.drop(shoulder_breadth, axis=1)
        data = data.drop(hip, axis=1)

        result = pd.concat([combined_df, data], axis=1, join='inner')
        '''
        Adds new column to feature list
        new column includes body shape constants
        '''
        features.insert(5,'shape')
        features.insert(6,hourglass)
        features.insert(7,inverted_triangle)
        features.insert(8,rectangle)
        features.insert(9,triangle)
        features.insert(10,measured_weight)

        target = features[-14:]  # get last column


        features = features[:-14]


        result.dropna(axis=0, how='any', inplace=True)

        X = result[features]
        y = result[target]

    #Train and Test
    '''
    Splits data into 3 dataset, Training, Validation and test dataset. Then it is saved into csv format in male folder is parameter gender is male and female folder.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    additional_df = pd.concat([X_test[subject_number], X_test['shape']], axis=1)
    X_test = X_test.drop(subject_number, axis=1)
    X_train = X_train.drop(subject_number, axis=1)
    X_val = X_val.drop(subject_number, axis=1)

    X_train.to_csv("./"+Gender.lower()+"/X_train.csv", encoding='utf-8', index=False)
    y_train.to_csv("./"+Gender.lower()+"/y_train.csv", encoding='utf-8', index=False)

    X_test.to_csv("./"+Gender.lower()+"/X_test.csv", encoding='utf-8', index=False)
    y_test.to_csv("./"+Gender.lower()+"/y_test.csv", encoding='utf-8', index=False)

    X_val.to_csv("./"+Gender.lower()+"/X_val.csv", encoding='utf-8', index=False)
    y_val.to_csv("./"+Gender.lower()+"/y_val.csv", encoding='utf-8', index=False)

    additional_df.to_csv("./"+Gender.lower()+"/additional_df.csv", encoding='utf-8', index=False)
    print("PRE PROCESSING COMPLETED")
    return True

# get_data('Male',body_shape_constraint=True)
# get_data('Female',body_shape_constraint=True)
if __name__ == "__main__":
    gender_arg=sys.argv[1]
    get_data(gender_arg, body_shape_constraint=True)