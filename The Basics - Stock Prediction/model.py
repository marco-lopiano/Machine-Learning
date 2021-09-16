import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

import os
import random
import numpy as np

def build_sequential_model(train_x, hidden_layers=1):
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())


    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    #model.add(LSTM(128))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model

# MAYBE REFINE IT IN THE FUTURE
def build_functional_model(train_x):

    model_input = Input(shape=train_x.shape[1:], name="ticker")
    x = LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True)(model_input)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    model_output = Dense(2, activation='softmax')(x)

    model = Model(model_input, model_output)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    print(model.summary())

    return model

def model_train(model, train_x, train_y, validation_x, validation_y, BATCH_SIZE, EPOCHS):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    tensorboard = TensorBoard(log_dir=f"logs/{dt_string}")
    filepath = f"RNN_Final-{dt_string}"

    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
    early_stopping_cb = EarlyStopping(patience=25, restore_best_weights=True)
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint, early_stopping_cb])

    results = model.evaluate(validation_x, validation_y, batch_size=128)
    print(f"test loss: {results[0]}")
    print(f"test acc: {results[1]}")

    return history

def load_latest_model():
    models = sorted(os.listdir("models"), reverse=True)
    latest = models[0]
    return load_model(os.path.join("models", latest))

def evaluate_prediction(test_x, test_y, MAIN_RANGE=1, SUB_RANGE=20):
    saved = load_latest_model()

    OVERALL_RIGHTS = 0
    OVERALL_WRONGS = 0
    for main in range(0, MAIN_RANGE):
        predictions = saved.predict(test_x)

        RIGHTS = 0
        WRONGS = 0
        for sub in range(0, SUB_RANGE-1):
            NUM = random.randint(0, len(predictions))
            prediction = np.argmax(predictions[NUM])
            value = test_y[NUM]

            if int(prediction) == int(value):
                RIGHTS+=1
            else:
                WRONGS+=1

        OVERALL_RIGHTS+=RIGHTS
        OVERALL_WRONGS+=WRONGS

    FULL_RANGE = MAIN_RANGE*SUB_RANGE
    print(f'{"-"*3} Model has been {OVERALL_RIGHTS} times right - {(OVERALL_RIGHTS/FULL_RANGE)*100}%')
    print(f'{"-"*3} Model has been {OVERALL_WRONGS} times wrong - {(OVERALL_WRONGS/FULL_RANGE)*100}%')
