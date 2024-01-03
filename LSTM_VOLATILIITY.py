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
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import tensorflow as tf

VIX_df = yf.download('^VIX')
VIX_df['Date'] = VIX_df.index.tolist()
df = VIX_df.iloc[:-20]
print(df.shape)
num_shape = 8000

train = df.iloc[:num_shape, 3:4].values
test = df.iloc[num_shape:, 3:4].values
print(train)
print(test)

sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []

#Price on next day
y_train = []

window = 60

for i in range(window, num_shape):
    X_train_ = np.reshape(train_scaled[i-window:i, 0], (window, 1))
    X_train.append(X_train_)
    y_train.append(train_scaled[i, 0])
X_train = np.stack(X_train)
y_train = np.stack(y_train)

# Initializing the Recurrent Neural Network
model = Sequential()
#Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
#Units - dimensionality of the output space

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 30, batch_size = 32);

print('X_train')
print(X_train)
print('y_train')
print(y_train)

# Prediction

df_volume = np.vstack((train, test))

inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

num_2 = df_volume.shape[0] - num_shape + window

X_test = []

for i in range(window, num_2):
    X_test_ = np.reshape(inputs[i - window:i, 0], (window, 1))
    X_test.append(X_test_)

X_test = np.stack(X_test)


predict = model.predict(X_test)
predict = sc.inverse_transform(predict)

diff = predict - test

print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))

pred_ = predict[-1].copy()
prediction_full = []
window = 60
df_copy = df.iloc[:, 3:4][1:].values

for j in range(20):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:num_shape]
    test_ = df_[num_shape:]

    df_volume_ = np.vstack((train_, test_))

    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
    inputs_ = inputs_.reshape(-1, 1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(window, num_2):
        X_test_3 = np.reshape(inputs_[k - window:k, 0], (window, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2)
    predict_ = model.predict(X_test_)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]

print('prediction_full')
print(prediction_full)


# regression line
reg = LinearRegression().fit(np.vstack(VIX_df['Date'][-20:].values.tolist()), prediction_full)
regression_data = reg.predict(np.vstack(VIX_df['Date'][-20:].values.tolist()))



DF =pd.DataFrame(
    {
        'Date': VIX_df['Date'][-20:].values,
        'Predict': prediction_full,
    }
)

max_idx = argrelextrema(np.array(DF['Predict'].values), np.greater, order=1)
min_idx = argrelextrema(np.array(DF['Predict'].values), np.less, order=1)


DF['peaks'] = np.nan
DF['lows'] = np.nan

for i in max_idx:
    DF['peaks'][i] = DF['Predict'][i]
for i in min_idx:
    DF['lows'][i] = DF['Predict'][i]

print(DF)

print(DF[DF['peaks']>= VIX_df['Close'][-20]])
print(DF[DF['lows']<= VIX_df['Close'][-20]])
print(len(DF[DF['peaks']>= VIX_df['Close'][-20]]))
print(len(DF[DF['lows']<= VIX_df['Close'][-20]]))
peaks_len = len(DF[DF['peaks']>= VIX_df['Close'][-19]])
lows_len = len(DF[DF['lows']<= VIX_df['Close'][-19]])


# ------------------------------ LSTM 20 ------------------------------

future_target = 20

X_train_multi = []
#Price on next day
y_train_multi = []

for i in range(window, num_shape):
    # print(i)
    # print(i-window)
    # print(i-future_target)
    X_train_ = np.reshape(train_scaled[i-window:i-future_target, 0], (window-future_target, 1))
    X_train_multi.append(X_train_)
    y_train_multi.append(train_scaled[i-future_target:i, 0])
    # print(X_train_)
    # print(train_scaled[i-future_target:i, 0])

X_train_multi = np.stack(X_train_multi)
y_train_multi = np.stack(y_train_multi)

# NEIRON   ======

multi_step_model = tf.keras.models.Sequential()

multi_step_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_multi.shape[1], 1)))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50, return_sequences = True))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50, return_sequences = True))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(LSTM(units = 50))
multi_step_model.add(Dropout(0.2))

multi_step_model.add(tf.keras.layers.Dense(20))


multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# multi_step_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
multi_step_model.fit(X_train_multi, y_train_multi, epochs = 30, batch_size = 32);


sc = MinMaxScaler(feature_range = (0, 1))
test_scaled = sc.fit_transform(test)
# print(np.array([test_scaled[-40:]]).shape)
predict_20 = multi_step_model.predict(np.array([test_scaled[-(window-future_target):]]))


predict_20 = sc.inverse_transform(predict_20)

print(predict_20)

# ------------------------------ visualization ------------------------------

fig = make_subplots(rows=1, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.01, x_title=f'Peaks: {peaks_len}, Lows: {lows_len}')

fig.add_trace(
    go.Scatter(
        x=VIX_df['Date'][-20:].values,
        y=predict_20[0],
        name="Predict Chain",
        line=dict(width=0.5),


    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=VIX_df['Date'][-80:].values,
        y= VIX_df['Close'][-80:],
        name="VIX",
        line=dict(width=0.5, color='green'),
        # stackgroup='one'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=VIX_df['Date'][-20:].values,
        y=prediction_full,
        line_shape='spline',
        name="FLAT Predict",
        line=dict(width=0.5),


    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=VIX_df['Date'][-20:].values,
        y=prediction_full,
        name="Predict",
        line=dict(width=0.5),


    ),
    row=1, col=1
)





fig.add_trace(
    go.Scatter(
        x=VIX_df['Date'][-20:].values,
        y=regression_data,
        name="Predict",
        line=dict(width=0.5, color='red'),


    ),
    row=1, col=1
)






fig.add_trace(
    go.Scatter(
        mode='markers',
        x=DF['Date'],
        y=DF['peaks'],
        marker=dict(
            size=10,
            symbol='triangle-up'
        ),
        name='Peaks'
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        mode='markers',
        x=DF['Date'],
        y=DF['lows'],
        marker=dict(
            size=10,
            symbol='triangle-down'
        ),
        name='Lows'
    ),
    secondary_y=False
)



fig.show()

