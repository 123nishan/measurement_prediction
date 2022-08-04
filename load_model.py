import torch
from nn import Model
from pickle import load
from numpy import asarray
from sklearn import preprocessing
option=float(input("Enter 1 for waist, 2 for chest, 3 for neck:"))
if option!=1:
    waist = float(input('Enter waist Circumference, Pref (mm)'))
if option!=2:
    chest = float(input('Enter chest Circumference (mm)'))
if option!=3:
    neck = float(input('Enter neck Circumference, Pref (mm)'))

#data order: Age,height,weight,shoes, chest,neck, :pred waist
age=float(input("Enter age: "))
height=float(input("Enter height: "))
weight=float(input("Enter weight: "))
shoe_size =float(input( 'Enter Shoe Size NL'))
model = Model(6,1)
model.load_state_dict(torch.load("waist_model.pt"))

if option==1:
    data = torch.tensor([[age,height,weight,shoe_size,chest,neck]])
    model.load_state_dict(torch.load("waist_model.pt"))
    scaler=load(open('scaler.pkl','rb'))
    output_scaler=load(open('output_scaler.pkl','rb'))
elif option==2:
    data = torch.tensor([[age,height,weight,shoe_size,neck,waist]])
    model.load_state_dict(torch.load("chest_model.pt"))
    scaler = load(open('scaler_chest.pkl', 'rb'))
    output_scaler = load(open('output_scaler_chest.pkl', 'rb'))
elif option==3:
    data = torch.tensor([[age,height,weight,shoe_size,waist,chest]])
    model.load_state_dict(torch.load("neck_model.pt"))
    scaler = load(open('scaler_neck.pkl', 'rb'))
    output_scaler = load(open('output_scaler_neck.pkl', 'rb'))

model.eval()
# chest=float(input('Enter chest size '))
# neck = float(input('Enter neck Circumference, Pref (mm)'))
# waist=float(input('Enter waist Circumference, Pref (mm)'))
# hip= float(input('Enter Hip Circumference, Maximum (mm)'))
# crotch_height=float(input('Enter Crotch Height (mm)'))
#'Age (Years)', 'Reported Height (cm)', 'Reported Weight (kg)', 'Shoe Size NL', 'Chest Circumference (mm)', 'Neck Base Circumference (mm)'

# input_data = torch.tensor([[age,height,weight,shoe_size,chest,neck]])
# #array order: neck,shoe,height,age,weight,chest
# input_data=torch.tensor([[44,56,82,983,187,492]])
# new=torch.tensor([[56,187,82,44,983,492]]) #856
# another=torch.tensor([[500.,43.,178.,60.,98.,1153.]])#1036
#scaled

# data_1=torch.tensor([[33,174,72,43,1026,504]]) #865
#data=torch.tensor([[-0.43659,-0.95316,-0.72755,-0.60770,0.12374,0.42921]])#-0.46423
scaler=load(open('scaler.pkl','rb'))
# test=scaler.inverse_transform(data.detach().numpy())
# print(test)

scale_data=scaler.transform(data)
scale_data=torch.from_numpy(scale_data)

# x = input_data.numpy()
# min_max_scaler = preprocessing.MinMaxScaler()output,_=model(input_data)
# print(output)
# x_scaled = min_max_scaler.fit_transform(x)
# print(x_scaled)
# data = pd.DataFrame(x_scaled, columns=data.columns)
#
#
output,_=model(scale_data.float())
print(output)

#output_scaler=load(open('output_scaler.pkl','rb'))
output=output_scaler.inverse_transform(output.detach().numpy()) #working fine
print(output)
