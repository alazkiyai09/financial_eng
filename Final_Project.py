from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import time
from pandas import ExcelWriter
from pandas import ExcelFile

def preprocessing_data(filename1, filename2):
    start_time = time.time()
    df = pd.read_csv(filename1, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])

    dataset_train = df
    training_set = dataset_train.iloc[:, 1:2].values
    training_set_scaled = training_set
    X_train = []
    y_train = []
    for i in range(32, len(training_set)):
        X_train.append(training_set_scaled[i-32, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train, y_train = X_train.reshape(-1, 1), y_train.reshape(-1, 1)

    dataset_test = pd.read_csv(filename2, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    real_price = dataset_test.iloc[:, 1:2].values
    dataset_total = pd.concat((dataset_train['Price'], dataset_test['Price']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test)- 32:].values
    inputs = inputs.reshape(-1,1)
    X_test = []
    for i in range(32, 64):
        X_test.append(inputs[i - 32, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 1)

    total_time = time.time() - start_time
    return X_train, y_train, X_test, real_price, dataset_total, training_set, total_time

def KNN_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time):
    start_time = time.time()
    X_train, y_train = X_train.reshape(-1, 1), y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    params = {'n_neighbors':[2,3,5,7,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=3)
    model.fit(X_train,y_train)
    training_set_scaled = training_set
    knn_prediction = model.predict(X_test)
    rms=np.sqrt(np.mean(np.power((np.array(real_price)-np.array(knn_prediction)),2)))
    data_total = dataset_total.values
    data_total = data_total.reshape(-1,1)
    predicted_price = knn_prediction
    total_time = total_time + time.time() - start_time

    return predicted_price, rms, total_time

def SVM_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time):
    start_time = time.time()
    X_train, y_train = X_train.reshape(-1, 1), y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.00001)
    svr_rbf.fit(X_train, y_train)
    svm_prediction = svr_rbf.predict(X_test)
    svm_prediction = svm_prediction.reshape(-1, 1)
    rms=np.sqrt(np.mean(np.power((np.array(real_price)-np.array(svm_prediction)),2)))
    data_total = dataset_total.values
    data_total = data_total.reshape(-1,1)
    predicted_price = svm_prediction
    total_time = total_time + time.time() - start_time

    return predicted_price, rms, total_time

def LSTM_Prediction(filename1, filename2):
    np.random.seed(10)
    start_time = time.time()
    df = pd.read_csv(filename1, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data = df

    dataset_train = data
    training_set = dataset_train.iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(32, len(training_set)):
        X_train.append(training_set_scaled[i-32:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 70, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    #regressor.add(LSTM(units = 60, return_sequences = True))
    #regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, activation='sigmoid'))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 10, batch_size = 30)

    # Part 3 - Making the predictions and visualising the results

    dataset_test = pd.read_csv(filename2, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    real_price = dataset_test.iloc[:, 1:2].values

    dataset_total = pd.concat((dataset_train['Price'], dataset_test['Price']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test)- 32:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(32, 64):
        X_test.append(inputs[i - 32:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = regressor.predict(X_test)
    predicted_price = sc.inverse_transform(predicted_price)
    data_total = dataset_total.values
    data_total = data_total.reshape(-1,1)
    rms=np.sqrt(np.mean(np.power((np.array(predicted_price)-np.array(real_price)),2)))
    total_time = time.time() - start_time

    return predicted_price, rms, real_price, total_time

def Plot_Graph(data_total, predicted_price1, predicted_price2, predicted_price3, label_name):
    plt.plot(predicted_price1, color = 'blue', label = 'Predicted KNN '+ label_name, fontsize=18)
    plt.plot(predicted_price2, color = 'green', label = 'Predicted SVM '+ label_name, fontsize=18)
    plt.plot(predicted_price3, color = 'yellow', label = 'Predicted LSTM '+ label_name, fontsize=18)
    plt.plot(data_total, color = 'red', label = 'Real ' + label_name, fontsize=18)
    plt.title(label_name + ' Price Prediction', fontsize=20)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel(label_name + ' Price', fontsize=18)
    plt.legend()
    plt.show()

def Plot_RMSE(rms1, rms2, rms3):
    plt.plot(rms1, color = 'blue', label = 'RMSE KNN')
    plt.plot(rms2, color = 'green', label = 'RMSE SVM')
    plt.plot(rms3, color = 'yellow', label = 'RMSE LSTM')
    plt.title('RMSE From 3 Method')
    plt.xlabel('Method')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

def Plot_Time(time1, time2, time3):
    plt.plot(time1, color = 'blue', label = 'Time KNN')
    plt.plot(time2, color = 'green', label = 'Time SVM')
    plt.plot(time3, color = 'yellow', label = 'Time LSTM')
    plt.title('Time Execution From 3 Method')
    plt.xlabel('Method')
    plt.ylabel('Time')
    plt.legend()
    plt.show()


def Plot_All_Data(filename, color, label_name):
    dataset_1 = pd.read_csv(filename, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    price_1 = dataset_1.iloc[:, 1:2].values
    plt.plot(price_1, color = color , label = label_name)
    plt.xlabel('Time')
    plt.ylabel('Price ' + label_name)
    plt.legend()
    plt.show()


def Volatility(filename1, filename2, filename3, filename4):
    df1 = pd.read_csv(filename1, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data1 = df1.sort_index(ascending=False, axis=0)
    df2 = pd.read_csv(filename2, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data2 = df2.sort_index(ascending=False, axis=0)
    df3 = pd.read_csv(filename3, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data3 = df3.sort_index(ascending=False, axis=0)
    df4 = pd.read_csv(filename4, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data4 = df4.sort_index(ascending=False, axis=0)
    new_data = pd.DataFrame(index=range(0, 365),columns=['Change 1', 'Change 2', 'Change 3'])
    data_snp = pd.DataFrame(index=range(0, 250),columns=['Change 4'])
    for i in range(0,365):
        new_data['Change 1'][i] = float(data1['Change'][len(data1) - 365 + i-1])
        new_data['Change 2'][i] = float(data2['Change'][len(data2) - 365 + i-1])
        new_data['Change 3'][i] = float(data3['Change'][len(data3) - 365 + i-1])

    for i in range(0, 250):
        data_snp['Change 4'][i] = float(data4['Change'][i])

    bitcoin = np.array(new_data['Change 1'])
    etherum = np.array(new_data['Change 2'])
    xrp = np.array(new_data['Change 3'])
    snp500 = np.array(data_snp['Change 4'])
    std_bitcoin = np.std(bitcoin)*np.sqrt(len(bitcoin))
    std_etherum = np.std(etherum)*np.sqrt(len(etherum))
    std_xrp = np.std(xrp)*np.sqrt(len(xrp))
    std_snp500 = np.std(snp500)*np.sqrt(len(snp500))

    return std_bitcoin, std_etherum, std_xrp, std_snp500

def Comparison_Data(filename, data_actual, predicted_price1, predicted_price2, predicted_price3, label_name):
    df = pd.read_csv(filename, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    data_comparison = pd.DataFrame(index=range(0,32),columns=['Date', 'Actual Price '+label_name, 'Price KNN', 'RMSE KNN', 'Price SVM', 'RMSE SVM', 'Price LSTM', 'RMSE LSTM'])
    predicted_price1, predicted_price2, predicted_price3, data_actual = predicted_price1.ravel(), predicted_price2.ravel(), predicted_price3.ravel(), data_actual.ravel()
    for i in range(0, len(data_comparison)):
        data_comparison['Date'][i] = df['Date'][i]
        data_comparison['Actual Price '+label_name][i] = data_actual[i]
        data_comparison['Price KNN'][i] = predicted_price1[i]
        data_comparison['RMSE KNN'][i] = np.sqrt(np.mean(np.power((np.array(predicted_price1[i])-np.array(data_actual[i])),2)))
        data_comparison['Price SVM'][i] = predicted_price2[i]
        data_comparison['RMSE SVM'][i] = np.sqrt(np.mean(np.power((np.array(predicted_price2[i])-np.array(data_actual[i])),2)))
        data_comparison['Price LSTM'][i] = predicted_price3[i]
        data_comparison['RMSE LSTM'][i] = np.sqrt(np.mean(np.power((np.array(predicted_price3[i])-np.array(data_actual[i])),2)))
    data_comparison.to_excel('Data Comparison '+label_name+'.xlsx', index=False)

def main():
    filename1 = 'Bitcoin Historical Data Train.csv'
    filename2 = 'Ethereum Historical Data Train.csv'
    filename3 = 'XRP Historical Data Train.csv'
    filename4 = 'Bitcoin Historical Data Test.csv'
    filename5 = 'Etherum Historical Data Test.csv'
    filename6 = 'XRP Historical Data Test.csv'
    filename7 = 'SnP500 Historical Data Train.csv'

    std_bitcoin, std_etherum, std_xrp, std_snp500 = Volatility(filename1, filename2, filename3, filename7)
    print("STD Bitcoin:", std_bitcoin)
    print("STD Etherum:", std_etherum)
    print("STD XRP:", std_xrp)
    print("STD S&P500:", std_snp500)
    rms1 = []
    rms2 = []
    rms3 = []
    time1 = []
    time2 = []
    time3 = []
    #df = pd.read_csv(filename2, delimiter=';', names = ['Date', 'Price','Open','High','Low','Change'])
    #data_comparison = pd.DataFrame(index=range(0,32),columns=['Date', 'Actual Price Bitcoin', 'Predicted Price Bitcoin', 'Actual Price Etherum', 'Predicted Price Etherum', 'Actual Price XRP', 'Predicted Price XRP'])
    #data_comparison['Date'] = df['Date']

    #bitcoin
    '''
    predicted_price = []
    X_train, y_train, X_test, real_price, dataset_total, training_set, total_time = preprocessing_data(filename1, filename4)
    predicted_price1, temp2, temp1 = KNN_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms1.append(temp2)
    time1.append(temp1)
    predicted_price2, temp2, temp1 = SVM_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms2.append(temp2)
    time2.append(temp1)
    predicted_price3, temp2, data_actual, temp1 = LSTM_Prediction(filename1, filename4)
    rms3.append(temp2)
    time3.append(temp1)
    Plot_Graph(data_actual, predicted_price1, predicted_price2, predicted_price3, "Bitcoin")
    Comparison_Data(filename4, data_actual, predicted_price1, predicted_price2, predicted_price3, "Bitcoin")
    
    #etherum
    predicted_price = []
    X_train, y_train, X_test, real_price, dataset_total, training_set, total_time = preprocessing_data(filename2, filename5)
    predicted_price1, temp2, temp1 = KNN_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms1.append(temp2)
    time1.append(temp1)
    predicted_price2, temp2, temp1 = SVM_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms2.append(temp2)
    time2.append(temp1)
    predicted_price3, temp2, data_actual, temp1 = LSTM_Prediction(filename2, filename5)
    rms3.append(temp2)
    time3.append(temp1)
    Plot_Graph(data_actual, predicted_price1, predicted_price2, predicted_price3, "Etherum")
    Comparison_Data(filename5, data_actual, predicted_price1, predicted_price2, predicted_price3, "Etherum")
    '''
    #xrp
    predicted_price = []
    X_train, y_train, X_test, real_price, dataset_total, training_set, total_time = preprocessing_data(filename3, filename6)
    predicted_price1, temp2, temp1 = KNN_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms1.append(temp2)
    time1.append(temp1)
    predicted_price2, temp2, temp1 = SVM_Prediction(X_train, y_train, X_test, real_price, dataset_total, training_set, total_time)
    rms2.append(temp2)
    time2.append(temp1)
    predicted_price3, temp2, data_actual, temp1 = LSTM_Prediction(filename3, filename6)
    rms3.append(temp2)
    time3.append(temp1)
    Plot_Graph(data_actual, predicted_price1, predicted_price2, predicted_price3, "XRP")
    Comparison_Data(filename6,data_actual, predicted_price1, predicted_price2, predicted_price3, "XRP")

    Plot_RMSE(rms1, rms2, rms3)
    Plot_Time(time1, time2, time3)

    Plot_All_Data(filename1, 'red', 'Bitcoin')
    Plot_All_Data(filename2, 'green', 'Etherum')
    Plot_All_Data(filename3, 'blue', 'XRP')

    print("RMSE KNN", rms1)
    print("RMSE SVM", rms2)
    print("RMSE LSTM", rms3)
    print("Time KNN", time1)
    print("Time SVM", time2)
    print("Time LSTM", time3)

if __name__ == '__main__':
    main()
