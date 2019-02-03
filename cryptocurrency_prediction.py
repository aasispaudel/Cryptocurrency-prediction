# -*- coding: utf-8 -*-

# The zip file of the data set can be found with this program ##
#_____________________________________________________________________________
filepath = './crypto_data'

import pandas as pd

# These are the different cryptocurrencies we are working with:
#   Bitcoin cash, Bitcoin, Ethereum, Litecoin

#   ==================================================================     ##

# We will be approaching this problem as classification problem
# We will take the cryptocurrency closing value and their volume
# to train the model   We will take last 60 minutes of all the above
# mentioned values of all four type of cryptocurrencies and try to predict
# the closing exchange value of next three minutes

#   =================================================================     ##


ratios = ['BCH', 'BTC', 'ETH', 'LTC']

main_df = pd.DataFrame()

#   =================================================================== ##

# Here we take our data from four csv files for respective cryptocurrencies
# And, join them inside same dataframe so we can work on the preprocessing

#  ==================================================================== ##

for ratio in ratios:
    df = pd.read_csv(f'{filepath}/{ratio}-USD.csv', 
                     names=['time', '_', '_', '_', f'{ratio}_close', f'{ratio}_volume'],
                    usecols=['time', f'{ratio}_close', f'{ratio}_volume'])
        
    df.set_index('time', inplace=True)
        
    if main_df.empty: # See if the dataframe is empty
        main_df = df
        continue
    
    main_df = main_df.join(df)
                 
main_df.fillna(method='ffill', inplace=True)
main_df.dropna(inplace=True)        

main_df.head()



PRECEDE_LEN = 60  # (in minutes) The time we want to consider for our training data
FOLLOW_LEN = 3    # (in minutes) The time in the future when we want to predict the value

# The cryptocurrency we want to predict (Change it to one of the options in ratios to predict others)
LUCKY_RATIO = 'LTC'

# ====================================================================================  ##########

# Let's create a target for our value to train on
# For this we will create a column 'future' which will contain closing value 3 minutes in the future
# Then we will compare and make it a buy (1) if the 'future' value is bigger than 'LTC_close' value
# else sell (0)

# ====================================================================================== #########

main_df['future'] = main_df[f'{LUCKY_RATIO}_close'].shift(-FOLLOW_LEN)
main_df.dropna(inplace=True)

main_df['target'] = list(map(lambda x, y: int(float(x) > float(y)), main_df['future'], main_df[f'{LUCKY_RATIO}_close']))

main_df.drop('future', 1, inplace=True) # We no longer need it

# main_df.head()

# Do this if you want to split your validation data beforehand (mark *** :: see below)
# main_df_validation = main_df[int(len(main_df.index)*0.95):]
# main_df_train = main_df[:int(len(main_df)*0.95)]


#  ============================================================================ ##

# Now that we have set our frame with all the values that we need, let us preprocess the data
# The first thing to remember is that we are using different cryptocurrencies
# So the best thing to do is to use percent change for each of our column
# Then we can normalize or scale our data and get it ready to pass into the model

# =========================================================================== ##

import numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn import preprocessing

def preprocess_data(df):    
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            
    df.dropna(inplace=True)
    
    for col in df.columns:
        if col!= 'target':
            df[col] = (preprocessing.scale(df[col].values))
            
    df.dropna(inplace=True)

    # Now, that we have scaled our data, let's create a collection of
    # previous set of values (in this case 60 values) with the corresponding target
    # of next three minute
    training_unit = deque(maxlen=PRECEDE_LEN)
    training_units = []
    for data in df.values:
        training_unit.append(data[:-1])
        
        if len(training_unit) == PRECEDE_LEN:
            training_units.append((np.array(training_unit), data[-1]))
            
    random.shuffle(training_units)

    # Balancing is the important part of working with financial data
    # It might not be the biggest of problems if the ratio is like 60:40
    # But, if the data is highly unbalanced, you ought to balance it
    # Anyways, let's do our job
    buys = [i for i in training_units if i[-1] == 1]
    sells = [i for i in training_units if i[-1] == 0]

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!
    split_point = min(len(buys), len(sells))
    
    buys = buys[:split_point]
    sells = sells[:split_point]
    
    training_units = buys + sells
    
    random.shuffle(training_units)
    
    # Separate x_train, and y_train and return the value
    X = []
    y = []
    for seq, target in training_units:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
    
    return np.array(X), y

x_train, y_train = preprocess_data(main_df)

# Do this if you have separate validation and training data (from ***)
# x_train, y_train = preprocess_data(main_df_train)
# x_val, y_val = preprocess_data(main_df_validation)


#  ==================================================================== ##

# Finally, we have data ready to throw into the model
# Let us make our model and start training the data

from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, CuDNNLSTM

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=x_train.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))

# model.summary()

from keras.optimizers import Adam

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

from keras.utils import to_categorical
y_train = to_categorical(y_train)

# y_val = to_categorical(y_val) # If you have separate validation data (from ***)

NO_EPOCHS = 10

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NO_EPOCHS, batch_size=64)

