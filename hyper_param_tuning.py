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

import lstm_baseline
import preproccessing as prep
import lstm_baseline as lb

EPOCHS = 15
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5 #if the accuracy does not increase after this many epochs -> break and continue with next
BN_AXIS= -1 #axis=-1 implies channel last ordering[rows][cols][channels].
LSTM_BASE = lb.LSTM_Base(run_training=False)
DATA = LSTM_BASE.data
PREPPER = LSTM_BASE.prepper
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
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[lb.get_f1])  #before categorical crossentropy
    # to do check accuracies
    return model


if __name__ == '__main__':
    tuner = kt.Hyperband(lstm_builder, objective=kt.Objective('get_f1', direction="max"), max_epochs=EPOCHS, factor=3, distribution_strategy=tf.distribute.MirroredStrategy(),
                         seed=42, directory='./', overwrite=True,
                         project_name='my_lstm_tuner')
    earlyStopper = EarlyStopping(monitor='get_f1', patience=EARLY_STOPPING_PATIENCE,
                                 restore_best_weights=True)
    tuner.search(PREPPER.x_train_padded, np.array(PREPPER.y_train), validation_data=(PREPPER.x_test_padded, np.array(PREPPER.y_test)), batch_size=BATCH_SIZE,
                 callbacks=[earlyStopper], epochs=EPOCHS)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    # save the best parameters
    with open(os.path.join('Hyperparameters','lstm_params.pkl'), 'wb') as f:
        pickle.dump(best_hps, f)
        f.close()