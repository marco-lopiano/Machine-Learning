import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad

'''
HUGE HELP FROM:
https://www.youtube.com/watch?v=IAzuLSa-sb0
https://github.com/MichaelAllen1966/2004_titanic/tree/master/jupyter_notebooks
'''

def scale_data(X_train, X_test):
    """Scale data 0-1 based on min and max in training set"""
    sc = MinMaxScaler()
    sc.fit(X_train)
    train_sc = sc.transform(X_train)
    test_sc = sc.transform(X_test)

    return train_sc, test_sc

def model_create(number_features,
                hidden_layer=1,
                neurons=32,
                activation_func='relu',
                dropout=0.0,
                learning_rate=0.001):

    ''' Create a Tensorflow model '''
    model = Sequential()

    for i in range(hidden_layer):
        model.add(Dense(neurons, input_dim=number_features, activation=activation_func))
        if dropout > 0.0:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=learning_rate)
    rms_prop = RMSprop(learning_rate=learning_rate, momentum=0.5)
    sgd = RMSprop(learning_rate=learning_rate, momentum=0.5)
    adagrad = Adagrad(learning_rate=0.001)

    opt = adam

    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model

def plot_performance(model, history_dict, save=False, my_dpi=126):
    ''' Create a plot of the model performance, like Tensorboard '''

    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    ax1 = fig.add_subplot(121)

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(acc_values) + 1)

    ax1.plot(epochs, acc_values, 'b', label='Training acc')
    ax1.plot(epochs, val_acc_values, 'r', label='Test accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    ax2 = fig.add_subplot(122)
    stringlist = []

    # adding custom info
    loss_info = f"loss: {history_dict['loss'][-1]:0.4f}"
    acc_info = f"accuracy: {acc_values[-1]:0.4f}"
    val_loss_info = f"val loss: {history_dict['val_loss'][-1]:0.4f}"
    val_acc_info = f"val accuracy: {val_acc_values[-1]:0.4f}"
    epoch_info = f"epochs: {len(acc_values)}"

    summary = model.summary(print_fn=lambda x: stringlist.append(x))
    stringlist.append('\n')
    stringlist.append(loss_info)
    stringlist.append(acc_info)
    stringlist.append(val_loss_info)
    stringlist.append(val_acc_info)
    stringlist.append(epoch_info)

    short_model_summary = "\n".join([ i for i in stringlist if '=' not in i ])

    txt = ax2.text(0, 0.6, short_model_summary, horizontalalignment='left',verticalalignment='center',transform = ax2.transAxes)
    txt.set_clip_on(False)
    ax2.axis('off')

    if save:
        from datetime import datetime
        import os

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

        p = f'performance_history/model_performance_{dt_string}'
        folder = os.mkdir(p)

        plt.savefig(f'{p}/plot.pdf')
        with open(f'{p}/history.json', 'w') as fp:
            json.dump(history_dict, fp, indent=4)
        return True

    plt.show()

def train_and_fit(X_train_sc, y_train, X_test_sc, y_test, number_features, EPOCHS=250, HIDDEN_LAYERS=2, NEURONS=128, DROPOUT=0.3, save=False):

    model = model_create(number_features,
                    hidden_layer=HIDDEN_LAYERS,
                    neurons=NEURONS,
                    activation_func='relu',
                    dropout=DROPOUT,
                    learning_rate=0.003)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    history = model.fit(X_train_sc,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=64,
                        validation_data=(X_test_sc, y_test),
                        verbose=1,
                        callbacks=[early_stopping_cb])


    if save:
        n_models = len(os.listdir('models'))
        if n_models==0:
            num = 1
        else:
            num = n_models+1

        model.save(f'models/titanic_model_{num}.model')

    return [model, history.history]


if __name__ == '__main__':

    STORED_MODEL = True

    if not STORED_MODEL:
        # Loading data and converting it to nupy array
        data = pd.read_csv('clean.csv')
        data = data.astype(float)
        X = data.drop('Survived',axis=1)
        y = data['Survived']
        X_np = X.values
        y_np = y.values

        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size = 0.25)
        X_train_sc, X_test_sc = scale_data(X_train, X_test)

        number_features = X_train_sc.shape[1]

        hist = train_and_fit(X_train_sc, y_train, X_test_sc, y_test, number_features, EPOCHS=250, HIDDEN_LAYERS=2, NEURONS=128, DROPOUT=0.5, save=True)
        plot_performance(hist[0], hist[1], save=True)

        del hist[0]

    else:
        # CONTROLS
        ITER = 300

        # Loading data and converting it to nupy array
        raw = pd.read_csv('full_data.csv')
        data = pd.read_csv('clean.csv')
        data = data.astype(float)
        X = data.drop('Survived',axis=1)
        y = data['Survived']
        X_np = X.values
        y_np = y.values

        # scaling inputs because the model has been trained on norm data
        sc = MinMaxScaler()
        sc.fit(X_np)
        X_np = sc.transform(X_np)

        models = os.listdir("models")
        m = sorted(models)[-1]
        path = os.path.join('models', m)

        # the actual loaded model
        testing_model = load_model(path)

        RIGHTS = 0
        WRONGS = 0
        DIED_FOR_SURVIDED = 0
        SURVIVED_FOR_DIED = 0

        for i in range(ITER):

            print('*'*75)
            NUM = random.randint(0,len(X_np)-1)

            predictions = testing_model.predict(X_np, verbose=3)

            pass_name = raw.values[NUM][2]
            correct_ = y_np[NUM]
            out_ = predictions[NUM]

            if out_ < 0.5:
                out_ = 0
            else:
                out_ = 1

            if out_ == 0:
                res_ = 'died'
            elif out_ == 1:
                res_ = 'survived'

            print(f"input array is: {X_np[NUM]}")
            print(f"correct label is {correct_}")
            print(f"model prediction is: {out_}")

            if correct_ == out_:
                RIGHTS+=1
                print(f"Model prediction is correct: {pass_name} {res_} in the Titanic accident")
            else:
                WRONGS+=1
                if res_ == 'died':
                    DIED_FOR_SURVIDED+=1
                    exp = 'survided'
                else:
                    SURVIVED_FOR_DIED+=1
                    exp = 'died'
                print(f"Model prediction is wrong: {pass_name} {exp} but the model predicted he/she {res_}")

        print('\n')
        print(f'Evaluating model {m}')
        print(f'{"-"*3} Model has been {RIGHTS} times right - {(RIGHTS/ITER)*100}%')
        print(f'{"-"*3} Model has been {WRONGS} times wrong - {(WRONGS/ITER)*100}%')
        print(f'{"-"*6} {DIED_FOR_SURVIDED} times models said passenger DIED but instead SURVIVED')
        print(f'{"-"*6} {SURVIVED_FOR_DIED} times models said passenger SURVIVED but instead DIED')
