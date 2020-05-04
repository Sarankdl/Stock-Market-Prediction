import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pickle
from datetime import date

def get_data(name, start_date, end_date):
    stock_df = yf.download(name,
                           start_date,
                           end_date,
                           progress=False)
    return stock_df


tqqq_train_data = get_data('TQQQ', '2010-01-01', '2019-12-31')
tqqq_training_set = pd.DataFrame(tqqq_train_data).iloc[:,1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
tqqq_training_scaled = sc.fit_transform(tqqq_training_set)

X_train = []
y_train = []

for i in range(60, len(tqqq_train_data)):
    X_train.append(tqqq_training_scaled[i-60:i, 0])
    y_train.append(tqqq_training_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Initializing the RRN
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Forth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

with open("stock_model_1.pickle","wb") as f:
    pickle.dump(regressor,f)

#pickle_in = open("stock_model_1.pickle", "rb")
#regressor = pickle.load(pickle_in)

# Getting the real stock price
tqqq_data_set = get_data('TQQQ', '2020-01-01', date.today())
tqqq_testing_set = pd.DataFrame(tqqq_data_set).iloc[:,1:2].values

# Getting the predicted stock price
dataset_total = tqqq_train_data['Open']
inputs = dataset_total[len(dataset_total) - len(tqqq_data_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 60 + len(tqqq_data_set)):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(tqqq_training_set, color = 'red', label = 'Real TQQQ Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted TQQQ Stock Price')
plt.title('TQQQ Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
