import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras

import keras_tuner as kt
from keras_tuner.tuners import Hyperband
from keras.layers import LSTM
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

import preproccessing as prep

NUM_CLASSES = 2
EPOCHS = 15
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5 #if the accuracy does not increase after this many epochs -> break and continue with next
BN_AXIS= -1 #axis=-1 implies channel last ordering[rows][cols][channels].
NUM_CLASSES = 2 # leak or no leak
DATA = prep.Dataloader(train_path='train.json', test_path='test.json')
PREPPER = prep.LSTMPrepper()
INP_SHAPE = PREPPER.max_len


def lstm_builder(tuner):
    input = Input(shape=(INP_SHAPE))
    model = Embedding(input_dim=3461, output_dim=INP_SHAPE, input_length=140)(input)
    model = SpatialDropout1D(tuner.Float("spatial_drop", min_value=0, max_value=0.25, step=0.05))(model)
    model = Bidirectional(LSTM(units=tuner.Int("lstm_units", min_value=32, max_value=240, step=16), return_sequences=True, recurrent_dropout=tuner.Float("rec_drop", min_value=0.05, max_value=0.3, step=0.05)))(model)
    output = TimeDistributed(Dense(units=len(DATA.unique_tags), activation="softmax"))(model)
    model = Model(input, output)
    #model.summary()
    lr = tuner.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  #before categorical crossentropy
    return model


def lstm1_model_builder(tuner):
    model = keras.Sequential()
    model.add(keras.layers.Input(INP_SHAPE))
    forward_layer = LSTM(tuner.Int("lstm_f_1", min_value=32, max_value=176, step=16),
                         recurrent_dropout=tuner.Float("drop_f_1", min_value=0.2, max_value=0.3, step=0.05),
                         return_sequences=True)
    backward_layer = LSTM(tuner.Int("lstm_b_1", min_value=16, max_value=128, step=16),
                          recurrent_dropout=tuner.Float("drop_b_1", min_value=0.2, max_value=0.3, step=0.05),
                          return_sequences=True, go_backwards=True)
    model.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    lr = 0.0001#tuner.Choice("learning_rate", values=[1e-3, 1e-4])
    # model.add()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # Compile the model
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=[
                      "sparse_categorical_accuracy"])  # optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Return the model
    return model

if __name__ == '__main__':
    tuner = kt.Hyperband(lstm_builder, objective="sparse_categorical_accuracy", max_epochs=EPOCHS, factor=3,
                         seed=42, directory='./',
                         project_name='my_lstm_tuner')
    earlyStopper = EarlyStopping(monitor="sparse_categorical_accuracy", patience=EARLY_STOPPING_PATIENCE,
                                 restore_best_weights=True)
    tuner.search(PREPPER.x_train_padded, PREPPER.y_train_padded, validation_data=(PREPPER.x_test_padded, PREPPER.y_test_padded), batch_size=BATCH_SIZE,
                 callbacks=[earlyStopper], epochs=EPOCHS)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    # save the best parameters
    with open(os.path.join('Hyperparameters','lstm_params.pkl'), 'wb') as f:
        pickle.dump(best_hps, f)
        f.close()