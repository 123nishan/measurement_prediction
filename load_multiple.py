import torch
from nn import Model
from pickle import load
from constant import *
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
    # chest=float(input('Enter chest size '))
    # neck = float(input('Enter neck Circumference, Pref (mm)'))

    # hip= float(input('Enter Hip Circumference, Maximum (mm)'))
    # crotch_height=float(input('Enter Crotch Height (mm)'))
    input_data = torch.tensor([[age, height, weight, shoe_size]])
    #'Age (Years)', 'Reported Height (cm)', 'Reported Weight (kg)', 'Shoe Size NL', 'Chest Circumference (mm)', 'Neck Base Circumference (mm)'
    model = Model(len(demographic),len(measurement))
    model.load_state_dict(torch.load("multiple_model_DI.pt"))
    model.eval()

    # #array order: neck,shoe,height,age,weight,chest
    # input_data=torch.tensor([[44,56,82,983,187,492]])
    # new=torch.tensor([[56,187,82,44,983,492]]) #856
    # another=torch.tensor([[500.,43.,178.,60.,98.,1153.]])#103
    #scaled

    # data_1=torch.tensor([[33,174,72,43,1026,504]]) #865
    #data=torch.tensor([[-0.43659,-0.95316,-0.72755,-0.60770,0.12374,0.42921]])#-0.46423
    scaler=load(open('multiple_scaler.pkl','rb'))
    # test=scaler.inverse_transform(data.detach().numpy())
    # print(test)

    scale_data=scaler.transform(input_data)
    scale_data=torch.from_numpy(scale_data)

    output,_=model(scale_data.float())

    print(output.detach().numpy())
    output_scaler=load(open('multiple_output_scaler.pkl','rb'))
    output=output_scaler.inverse_transform(output.detach().numpy()) #working fine

    []
    # print("chest, crotch ht,hip,neck,wasit all in cm",output/25.4)
    print("Waist Circumference, Pref (mm), Chest Circumference (mm), Neck Base Circumference (mm),Hip Circumference, Maximum (mm), Crotch Height (mm), Thigh Circumference (mm), Shoulder Breadth (mm), Waist Height, Preferred (mm), Arm Length (Shoulder to Wrist) (mm) all in cm", output / 10)

    repeat=input("Do you want to repeat? (y/n) ")

