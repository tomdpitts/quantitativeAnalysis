"""
Created on Thu Sep 17 10:34:48 2020

@author: Tom
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# parameters to easily adjust how to slice up datasets 

start = -3000
cutoff = -240
days = 30

# dataset_train = pd.read_csv('asx.csv')
# dataset_train = dataset_train.dropna().reset_index(drop=True)

train_data_dict = {}
test_data_dict = {}

# import filenames

import os

path = './data'

folder = os.fsencode(path)

filenames = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.csv') ): # whatever file types you're using...
        filenames.append(filename[:-4])

filenames.sort()

print(filenames)

lengths = []

for i in filenames:
    
    # start by getting the data
    
    x = pd.read_csv('./data/{}.csv'.format(i))
    # theoretically, the neural net should learn to ignore 0 values
    x = x.fillna(0)
    
    lengths.append(len(x))
    
lengths = sorted(lengths)
# set start to equal value of 
#start = -(lengths[0])

def timeDiff(x):
    x = np.array(x)
    output = [0]*len(x)
    
    for k in range(1, len(x)):
        output[k] = x[k]-x[k-1]
    
    output = output[1:]
    
    output = pd.DataFrame(output)
        
    return output

for i in filenames:
    print(i)
    
    # start by getting the data 
    
    x = pd.read_csv('./data/{}.csv'.format(i))
    # theoretically, the neural net should learn to ignore 0 values
    #x = x.fillna(0)
    
    price_train = x.iloc[start:cutoff, 4:5]
    vol_train = x.iloc[start:cutoff, 5:6]
    
    price_test = x.iloc[cutoff:, 4:5]
    vol_test = x.iloc[cutoff:, 5:6]
    
    # convert to time difference series
    
    price_diff_train = timeDiff(price_train)
    vol_diff_train = timeDiff(vol_train)
    
    price_diff_test = timeDiff(price_test)
    vol_diff_test = timeDiff(vol_test)
    
    
    # scale the data to the (0,1) range
    
    price_scaler = MinMaxScaler(feature_range=(-1,1))
    vol_scaler = MinMaxScaler(feature_range=(-1,1))
    

    scaled_price_data = price_scaler.fit_transform(price_diff_train)
    scaled_vol_data = vol_scaler.fit_transform(vol_diff_train)
    
    scaled_price_test_data = price_scaler.transform(price_diff_test)
    scaled_vol_test_data = vol_scaler.transform(vol_diff_test)
    
    
    # turn 1d vector into 2d matrix to capture a set length of days of data e.g. 60 days

    price_train_array = []
    vol_train_array = []
    
    price_test_array = []
    vol_test_array = []
    
    for j in range(days, len(scaled_price_data)):
        
        price_train_array.append(scaled_price_data[j-days:j, 0])
        vol_train_array.append(scaled_vol_data[j-days:j, 0])
        
    for y in range(days, len(scaled_price_test_data)):
        
        price_test_array.append(scaled_price_test_data[y-days:y, 0])
        vol_test_array.append(scaled_vol_test_data[y-days:y, 0])
        
 
    # convert to np array
        
    price_train_array = np.array(price_train_array)
    vol_train_array = np.array(vol_train_array)
    
    price_test_array = np.array(price_test_array)
    vol_test_array = np.array(vol_test_array)
    
    
    
    # store these flat tables in a dict, one for train, one for test
    
    train_data_dict['{}_price_train'.format(i)] = price_train_array
    train_data_dict['{}_vol_train'.format(i)] = vol_train_array
    
    test_data_dict['{}_price_test'.format(i)] = price_test_array
    test_data_dict['{}_vol_test'.format(i)] = vol_test_array
    

lister_train = []
lister_test = []

for k, (key,value) in enumerate(train_data_dict.items()):
    lister_train.append(value)
    
for k, (key,value) in enumerate(test_data_dict.items()):
    lister_test.append(value)
    
# create final 3D tensor composed of stacked 2D timestepped tables, one per feature
    
X_train = np.stack(lister_train, axis = 2)
X_test = np.stack(lister_test, axis = 2)



# build target y

y = pd.read_csv('ASX.csv')
y = y.interpolate()

y_train = y.iloc[start:cutoff, 4:5]
y_test = y.iloc[cutoff: , 4:5]

y_train = timeDiff(y_train)
y_test = timeDiff(y_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_sc = MinMaxScaler(feature_range=(-1,1))
y_sc2 = MinMaxScaler(feature_range=(-1,1))

y_train = y_sc.fit_transform(y_train)
y_test = y_sc.transform(y_test)


y_train = y_train[days:]
y_test = y_test[days:]



# build LSTM Recurrent Neural Network

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 32, return_sequences = True, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 32, return_sequences = True, activation = 'relu'))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 16))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))

epochs = 5

from tensorflow import keras
#opt = keras.optimizers.SGD(learning_rate=0.1)
#opt = keras.optimizers.Adam(learning_rate=0.01)

regressor.compile(loss='mean_squared_error', optimizer='adam')

history = regressor.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = epochs, shuffle = False)

# Plot loss function by epoch

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochRange = range(1, epochs+1)
plt.plot(epochRange, loss_train, 'g', label='Training loss')
plt.plot(epochRange, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = y_sc.inverse_transform(predicted_stock_price)

real_stock_price = y_test

# Since stock prices follow Wiener Process, prediction is almost perfectly constant as an average value (i.e. best prediction available)
    

plt.plot(real_stock_price, color = 'red', label = 'ASX50 Daily Change')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ASX50 Value')
plt.title('ASX50 Value Prediction')
plt.xlabel('Time')
plt.ylabel('ASX50 Value')
plt.legend()
plt.show()

# Simulate 'reversion' strategy, for different cutoff points 

profit = 0
profit_tracker = []

profit2 = 0
profit_tracker2 = []

profit3 = 0
profit_tracker3 = []

# TODO - factor in trading costs

for g in range(1, abs(cutoff+days)-1):
    
    mid = predicted_stock_price[g]
    
    yester = real_stock_price[g-1]
    today = real_stock_price[g]
    
    if yester < mid:
        profit += today
        
    else:
        profit -= today
        
        
    if yester < 1:
        profit2 += today
    else:
        profit2 -= today
        
    if yester < -0.5:
        profit3 += today
    else:
        profit3 -= today

    
    profit_tracker.append(float(profit))
    profit_tracker2.append(float(profit2))
    profit_tracker3.append(float(profit3))
    

plt.plot(profit_tracker, color = 'green', label = 'Cumulative Profit1')
plt.title('Profit tracker')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show() 

plt.plot(profit_tracker2, color = 'yellow', label = 'Cumulative Profit2')
plt.title('Profit tracker')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show() 

plt.plot(profit_tracker3, color = 'orange', label = 'Cumulative Profit3')
plt.title('Profit tracker')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show() 
