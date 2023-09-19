import os

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

#from livelossplot.tf_keras import PlotLossesCallback

import preproccessing as prep
from torchmetrics.classification import F1Score


def get_f1(y_true, y_pred): #taken from old keras source code
    #f1_score = 0.0
    #try:
        if np.shape(y_true) == np.shape(y_pred):
            # Count positive samples.
            c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
            c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

            # If there are no true samples, fix the F1 score at 0.
            if c3 == 0:
                f1_score = 0.0

            # How many selected items are relevant?
            precision = c1 / c2

            # How many relevant items are selected?
            recall = c1 / c3

            # Calculate f1_score
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
    #except:
        #f1_score = 0.0
     #   print(np.shape(y_true))
      #  print(np.shape(y_pred))
        return f1_score

class LSTM_Base:
    def __init__(self, run_training=None):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json')
        self.prepper = prep.LSTMPrepper()
        self.model = self.make_model()
        if run_training == True:
            self.train()

    def make_model(self):
        max_len = self.prepper.max_len
        input = Input(shape=(max_len))
        model = Embedding(input_dim=5000, output_dim=max_len, input_length=140, mask_zero=True)(input)
        model = SpatialDropout1D(0.15)(model)
        model = Bidirectional(LSTM(units=80, return_sequences=True, recurrent_dropout=0.05))(model)
        output = TimeDistributed(Dense(units=len(self.data.unique_tags), activation="softmax"))(model)
        model = Model(input, output)
        model.summary()
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[get_f1])
        return model

    def train(self):
        early_stop = EarlyStopping(monitor='get_f1', patience=10, verbose=0, mode='max', restore_best_weights=False)   #monitor='val_accuracy'
        callbacks = [early_stop]    #PlotLossesCallback() for plotting
        history=self.model.fit(self.prepper.x_train_padded, np.array(self.prepper.y_train), validation_split=0.2, batch_size=32, epochs=50, verbose=1, callbacks=callbacks)
        #self.model.evaluate(self.prepper.x_test_padded, np.array(self.prepper.y_test))
        results = self.model.predict(self.prepper.x_test_padded)
        f1 = get_f1(np.array(self.prepper.y_test), results)
        print(f1)
        self.model.save_weights(os.path.join('models', 'lstm_base.h5'))
#lstm_base = LSTM_Base(run_training=True)