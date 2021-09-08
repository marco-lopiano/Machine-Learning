import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import deque
import numpy as np
import random

import sklearn
from sklearn.preprocessing import MinMaxScaler

'''
VERY HELPFUL RESOURCE:
https://pythonprogramming.net/normalizing-sequences-deep-learning-python-tensorflow-keras/?completed=/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/
'''

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 1

def create_target(df, FUTURE_PERIOD_PREDICT):
    """ Create column data that will be used as when training the model"""
    df['future'] = df['Close'].shift(-FUTURE_PERIOD_PREDICT)
    classify = lambda x,y : 1 if x < y else 0
    df['target'] = list(map(classify, df['Close'], df['future']))
    df.drop('future', axis=1, inplace=True)
    df.dropna(inplace=True)

    return df

def scale_data(data):
    """Scale data 0-1 based on min and max in training set"""
    sc = MinMaxScaler()
    sc.fit(data)
    train_sc = sc.transform(data)

    return data

def train_and_validation(df):
    times = sorted(df.index.values)
    last_5pct = sorted(df.index.values)[-int(0.05*len(times))]

    validation_df = df[(df.index >= last_5pct)]
    train_df = df[(df.index < last_5pct)]

    return train_df, validation_df

# TODO: MAKE THIS METHOD WORK
def preprocess_df(df, SEQ_LEN, balance=False):

    pd.set_option('chained',None)
    for col in df.columns:
        if col!='target':
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df[col] = sklearn.preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    if balance:
        random.shuffle(buys)
        random.shuffle(sells)

        lower = min(len(buys), len(sells))

        buys = buys[:lower]
        sells = sells[:lower]

        sequential_data = buys+sells
        random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

    test


df = pd.read_csv("data/aapl_hist.csv")
df.set_index('Date', inplace=True)

df = create_target(df, FUTURE_PERIOD_PREDICT)

train_df, test_df = train_and_validation(df)

# data ready for training
train_x, train_y = preprocess_df(train_df, SEQ_LEN, balance=True)
test_x, test_y = preprocess_df(test_df, SEQ_LEN, balance=False)
