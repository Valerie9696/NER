import os

import numpy as np
import tensorflow as tf
import keras
from keras.backend import eval
from keras import backend as kb
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense, Masking
from keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional, Flatten, Activation, Embedding, Concatenate, Input, Dense, Dropout, MaxPool2D
from keras.layers import Reshape, Flatten, Conv1D, MaxPool1D, Embedding,BatchNormalization, LSTM, Conv2D

#from livelossplot.tf_keras import PlotLossesCallback

import preproccessing as prep


def get_f1(y_true, y_pred): #taken from old keras source code
    f1_score = 0.6
    try:
        print(np.shape(y_true))
        print(np.shape(y_pred))
        if np.shape(y_true) == np.shape(y_pred):
            epsilon = kb.epsilon()  # avoid division by 0
            tp_clipped = kb.clip(y_true * y_pred, 0, 1)
            tp_rounded = kb.round(tp_clipped)
            true_positives = kb.sum(tp_rounded)
            print(true_positives)
            possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
            predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + epsilon)
            recall = true_positives / (possible_positives + epsilon)
            f1_score = 2*(precision*recall)/(precision+recall+epsilon)
        else:
            #print(np.shape(y_true), np.shape(y_pred))
            f1_score = 0.8
    except:
        f1_score = 0.5
        print(np.shape(y_true))
        print(np.shape(y_pred))
    return f1_score


class LSTM_Base:
    def __init__(self, run_training=None):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json')
        self.prepper = prep.LSTMPrepper()
        self.model = self.make_model()
        if run_training:
            self.train()

    def make_model(self):
        max_len = self.prepper.max_len
        dropout = 0.25
        input = Input(shape=((max_len),))
        a = [self.prepper.embedding]
        b = len(self.prepper.tokenizer.word_index) + 1
        print(len(self.prepper.tokenizer.word_index))
        emb = Embedding(input_dim=len(self.prepper.tokenizer.word_index) + 1, output_dim=100, weights=[self.prepper.embedding], input_length=140, mask_zero=True)(input)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(emb)
        drop1 = Dropout(dropout)(lstm1)
        #mask = Masking(mask_value=0.)(drop1)
        lstm2 = Bidirectional(LSTM(128, return_sequences=True))(drop1)
        #drop2 = Dropout(dropout)(lstm2)
        #lstm2 = Bidirectional(LSTM(128, return_sequences=False))(drop2)
        td = TimeDistributed(Dense(units=37, activation="softmax"))(lstm2)
        #batch1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                    #beta_initializer='zeros',
                                    #gamma_initializer='ones', moving_mean_initializer='zeros',
                                    #moving_variance_initializer='ones', beta_regularizer=None,
                                    #gamma_regularizer=None, beta_constraint=None,
                                    #gamma_constraint=None)(lstm2)
        #dense3 = Dense(181, activation='relu')(lstm2)
        #output = TimeDistributed(Dense(units=len(self.data.unique_tags), activation="softmax"))(dense3)
        #drop4 = Dropout(0.5)(dense3)
        #dense4 = Dense(128, activation='relu')(drop4)
        #drop5 = Dropout(0.5)(dense4)
        #dense7 = Dense(128, activation='relu')(drop5)
        #output = Dense(len(self.data.unique_tags), activation='sigmoid')(td)
        model = Model(input, td)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[get_f1])
        return model

    def train(self):
        early_stop = EarlyStopping(monitor='get_f1', patience=5, verbose=0, mode='max', restore_best_weights=False)   #monitor='val_accuracy'
        callbacks = [early_stop]    #PlotLossesCallback() for plotting
        history=self.model.fit(self.prepper.x_train_padded, np.array(self.prepper.y_train), validation_split=0.2, batch_size=32, epochs=100,verbose=1,callbacks=callbacks)
        self.model.evaluate(self.prepper.x_test_padded, np.array(self.prepper.y_test))
        self.model.save_weights(os.path.join('models', 'lstm_base.h5'))
        result = self.model.predict(self.prepper.x_test_padded)
        all_tags = tf.argmax(result, axis=-1)
        for tags in all_tags:
            t = eval(tags[0]) #tf.cast(tags[0], tf.int32)
            labels = [self.prepper.dl.lookup_table[eval(tag)] for tag in tags]
            #print(tags)
            #print(labels)
        print(labels)
#lstm_base = LSTM_Base(run_training=True)