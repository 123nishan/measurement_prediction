
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


    #Function for test model
# def test_model(X_test, y_test, scaler, output_scaler,model,criterion):
#     # Scale
#     print("TESTING")
#     y_test = pd.DataFrame(y_test)
#
#     X_test_scaled = scaler.transform(X_test.values)
#
#     #y_test_scaled = output_scaler.transform(y_test.values)
#
#     X_test_scaled_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#     Test_data = pd.concat([X_test_scaled_data, pd.DataFrame(y_test.values, columns=y_test.columns)], axis=1)
#     test_data = multiple_dataloader.CustomDataset(Test_data)
#     test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
#     model.eval()
#     test_loss = 0.0
#     avg_values=[]
#     all_pred=[]
#     all_output=[]
#     with torch.no_grad():
#         for batch_idx, (features, target) in enumerate(test_loader):
#             output, _ = model(features)
#             target = target.float()
#             output = output_scaler.inverse_transform(output.detach().numpy())
#
#             error=np.absolute(np.subtract(output,target.detach().numpy()))
#             all_pred.append(output.tolist())
#             all_output.append(target.detach().numpy().tolist())
#
#             avg_values.append((np.average(error,axis=0)).tolist())
#             # print(output)
#             # loss = criterion(output, target)
#             #
#             #
#             # test_loss += loss.item()
#
#             # print(output)
#             # print(output[:, 0]) #output of column index 0
#             #print("------------------------------------------------------")
#             # loss_hip=criterion(output[:,0], target[:,0])
#             #hip_loss += sum(output[:, 0])
#             # hip_loss+=loss_hip.item()
#             # writer.add_scalar('train', loss, i)
#             # writer.flush()
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             # if batch_idx==4:
#             #     print("size of output:",output.shape)
#             # train_loss += loss.item()
#             # print(train_loss)
#
#     avg_values=np.array(avg_values)
#     avg_values=avg_values.mean(axis=0)
#     twoD_pred=[elem for twod in all_pred for elem in twod]
#     twoD_output=[elem for twod in all_output for elem in twod]
#
#     print(twoD_output)
#     print(twoD_pred)
#     #avg_error = output_scaler.inverse_transform(np.array(avg_values))
#     print("avg_values",avg_values/10)


#Split and scale data

# data, total_features,output = handle_data()
# features = data.columns.tolist()
#
# target = features[-5:]  # get last column
#
# # features = list(set(features) - set([target]))
# features = features[:-5]
#
# X = data[features]
# y = data[target]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
#
# X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
#
# # define scalar
#
# scaler = preprocessing.StandardScaler()
# y_train = pd.DataFrame(y_train)
# y_eval = pd.DataFrame(y_eval)
# output_scaler = preprocessing.StandardScaler()
# scaler.fit(X_train.values)
# output_scaler.fit(y_train.values)
# X_train_scaled = scaler.transform(X_train.values)
# X_eval_scaled = scaler.transform(X_eval.values)
# #
#
#
# y_train_scaled = output_scaler.transform(y_train.values)
# y_eval_scaled = output_scaler.transform(y_eval.values)
#
# X_train_scaled_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_eval_scaled_data = pd.DataFrame(X_eval_scaled, columns=X_test.columns)
#
# # Train_data = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
# # Valid_data = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
# Train_data = pd.concat([X_train_scaled_data, pd.DataFrame(y_train_scaled, columns=y_train.columns)], axis=1)
# Valid_data = pd.concat([X_eval_scaled_data, pd.DataFrame(y_eval_scaled, columns=y_test.columns)], axis=1)
#
# # scaler.fit(Train_data)
# # Train_data_scaled = scaler.transform(Train_data.values)
# # Valid_data_scaled = scaler.transform(Valid_data.values)
# dump(scaler, open('multiple_scaler.pkl', 'wb'))
# dump(output_scaler, open('multiple_output_scaler.pkl', 'wb'))

# train_data = multiple_dataloader.CustomDataset(Train_data)
# eval_data = multiple_dataloader.CustomDataset(Valid_data)
#
# train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
# eval_loader = DataLoader(dataset=eval_data, batch_size=32, shuffle=True)

# model = Model(total_features, output)