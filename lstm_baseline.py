import os

import numpy as np
import tensorflow as tf
import keras
from keras import backend as kb
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from livelossplot.tf_keras import PlotLossesCallback

import preproccessing as prep


def get_f1(y_true, y_pred): #taken from old keras source code
    try:
        if np.shape(y_true) == np.shape(y_pred):
            epsilon = kb.epsilon()  # avoid division by 0
            tp_clipped = kb.clip(y_true * y_pred, 0, 1)
            tp_rounded = kb.round(tp_clipped)
            true_positives = kb.sum(tp_rounded)
            possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
            predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + epsilon)
            recall = true_positives / (possible_positives + epsilon)
            f1_score = 2*(precision*recall)/(precision+recall+epsilon)
        else:
            f1_score = 0.8
    except:
        f1_score = 0.5
        print(np.shape(y_true))
        print(np.shape(y_pred))
    return f1_score

class LSTM_Base:
    def __init__(self):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json')
        self.prepper = prep.LSTMPrepper()
        self.model = self.make_model()
        self.train()

    def make_model(self):
        max_len = self.prepper.max_len
        input = Input(shape=(max_len))
        model = Embedding(input_dim=3461, output_dim=max_len, input_length=140)(input)
        model = SpatialDropout1D(0.1)(model)
        model = Bidirectional(LSTM(units=150, return_sequences=True, recurrent_dropout=0.1))(model)
        output = TimeDistributed(Dense(units=len(self.data.unique_tags), activation="softmax"))(model)
        model = Model(input, output)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[get_f1])
        return model

    def train(self):
        early_stop = EarlyStopping(monitor='get_f1', patience=1, verbose=0, mode='max', restore_best_weights=False)   #monitor='val_accuracy'
        callbacks = [early_stop]    #PlotLossesCallback() for plotting
        history=self.model.fit(self.prepper.x_train_padded,np.array(self.prepper.y_train),validation_split=0.2,batch_size=32,epochs=100,verbose=1,callbacks=callbacks)
        self.model.evaluate(self.prepper.x_test_padded, np.array(self.prepper.y_test))
        self.model.save_weights(os.path.join('models', 'lstm_base.h5'))
lstm_base = LSTM_Base()