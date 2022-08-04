
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.
# def train(model, train_loader,criterion,optimizer):
#     epoch_loss = 0
#     y_true = []
#     y_pred = []
#     model.train()
#     for (x, y) in tqdm(train_loader, desc='Training', leave=False):
#         optimizer.zero_grad()
#         output, _ = model(x)
#
#         loss = criterion(output, y.unsqueeze(1))#calc loss
#
#         loss.backward() #calc gradients
#         optimizer.step()#update weights
#         epoch_loss += loss
#         y = y.float()
#         y_true.extend(y.tolist())
#         y_pred.extend(output.reshape(-1).tolist())
#     return epoch_loss, y_true, y_pred
    # print("Accuracy: ", r2_score(y_true, y_pred))
    # print("************************************")



    # def train(model,criterion,optimizer,train_loader,epochs=1000):
#     for epoch in range(epochs):
#         model.train()
#         for i,(features,label) in enumerate(train_loader):
#             output, _ = model(features)
#             loss = criterion(output, label.unsqueeze(1))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             if i%100==0:
#                 print('Epoch [{}/{}],step[{}/{}],loss:{:.4f}'.format(epoch+1,epochs,i+1,len(train_loader),loss.item()))

 # waist = 'Waist Circumference, Pref (mm)'
    # subject_number = 'Subject Number'
    # age = 'Age (Years)'
    # gender = 'Gender'
    # height = 'Reported Height (cm)'
    # weight = 'Reported Weight (kg)'
    # shoe_size = 'Shoe Size NL'
    # features=[subject_number,gender,age, height, weight, shoe_size]
    # target=[subject_number,waist]
    # demographic_path = "./data/demographic_metric.csv"
    # measurement_path = "./data/measurement_metric.csv"

    # demographic_data=pd.read_csv(demographic_path,skipinitialspace=True,usecols=features)
    # measurement_data=pd.read_csv(measurement_path,skipinitialspace=True,usecols=target)
    #demographic_data, measurement_data = read_data.ReadData(demographic_path, measurement_path).read_data()




    # measurement_data = measurement_data[[subject_number, waist]]
    # data = demographic_data[[subject_number, age, gender, height, weight, shoe_size]]
    
    # merge two dataset
    # data = demographic_data.merge(measurement_data, on=subject_number, how='left')
    # data['Gender'].replace('Female', 0, inplace=True)
    # data['Gender'].replace('Male', 1, inplace=True)

    # data = data.drop(subject_number, axis=1)

    # data.dropna(axis=0,how='any',inplace=True)

    # data=data[data[height].str.contains("No Response")==False]
    # data = data[data[weight].str.contains("No Response") == False]
    # data = data[data[shoe_size].str.contains("No Response") == False]