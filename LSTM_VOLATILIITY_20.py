import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
import yfinance as yf
import tensorflow as tf
import plotly.graph_objects as go
import plotly.io as pio

VIX_df = yf.download('^VIX')
VIX_df['Date'] = VIX_df.index.tolist()
df = VIX_df.iloc[:-20]
print(df.shape)
num_shape = int(len(df) * 0.95)

train = df.iloc[:num_shape, 3:4].values
test = df.iloc[num_shape:, 3:4].values
# print(train)
# print(test)

sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []

#Price on next day
y_train = []

window = 45
future_target = 20

for i in range(window, num_shape):
    # print(i)
    # print(i-window)
    # print(i-future_target)
    X_train_ = np.reshape(train_scaled[i-window:i-future_target, 0], (window-future_target, 1))
    X_train.append(X_train_)
    y_train.append(train_scaled[i-future_target:i, 0])
    # print(X_train_)
    # print(train_scaled[i-future_target:i, 0])

X_train = np.stack(X_train)
y_train = np.stack(y_train)

# NEIRON   =======================================================================================

multi_step_model = tf.keras.models.Sequential()

multi_step_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50, return_sequences = True))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50, return_sequences = True))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(tf.keras.layers.Dense(20))

# print('X_train')
# print(X_train)
# print('y_train')
# print(y_train)
# print('test')
print(y_train.shape)
print(X_train.shape)


multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# multi_step_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
multi_step_model.fit(X_train, y_train, epochs = 10, batch_size = 32);


sc = MinMaxScaler(feature_range = (0, 1))
test_scaled = sc.fit_transform(test)
print(np.array([test_scaled[-40:]]).shape)
predict = multi_step_model.predict(np.array([test_scaled[-(window-future_target):]]))


predict = sc.inverse_transform(predict)

diff = predict - VIX_df['Close'][-20:]

print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))

plt.figure(figsize=(20,7))
plt.plot(VIX_df['Date'][-350:].values, VIX_df['Close'][-350:], color = 'red', label = 'Real Stock Price')
plt.plot(VIX_df['Date'][-20:].values, predict[0], color = 'blue', label = 'Predicted Stock Price')
plt.xticks(VIX_df['Date'][-350:].values)
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()