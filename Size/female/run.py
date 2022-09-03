import torch
from nn import Model
from pickle import load
from Size.female.size_constant_female import *
import numpy as np
from numpy import asarray
from sklearn import preprocessing
#data order: Age,height,weight,shoes,, :pred chest,crotch_ht,hip,neck,waist
repeat="y"
while repeat=="y":
    age=float(input("Enter age: "))
    height=float(input("Enter height in cm: "))
    weight=float(input("Enter weight in kg: "))
    shoe_size =float(input( 'Enter Shoe Size NL'))

    input_data = torch.tensor([[age, height, weight]])
    #'Age (Years)', 'Reported Height (cm)', 'Reported Weight (kg)', 'Shoe Size NL', 'Chest Circumference (mm)', 'Neck Base Circumference (mm)'
    model = Model(len(demographic),len(size))
    model.load_state_dict(torch.load("size_female_dutch.pt"))
    model.eval()


    # data_1=torch.tensor([[33,174,72,43,1026,504]]) #865
    #data=torch.tensor([[-0.43659,-0.95316,-0.72755,-0.60770,0.12374,0.42921]])#-0.46423
    scaler=load(open('size_scaler_female.pkl','rb'))
    # test=scaler.inverse_transform(data.detach().numpy())
    # print(test)

    scale_data=scaler.transform(input_data)
    scale_data=torch.from_numpy(scale_data)

    output,_=model(scale_data.float())

    print(output.detach().numpy())
    output_scaler=load(open('size_output_scaler_female.pkl','rb'))
    output=output_scaler.inverse_transform(output.detach().numpy()) #working fine


    # print("chest, crotch ht,hip,neck,wasit all in cm",output/25.4)
    print("Pant waist size, Bra size", output)

    repeat=input("Do you want to repeat? (y/n) ")

