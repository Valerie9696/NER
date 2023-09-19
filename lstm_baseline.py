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
from sklearn.metrics import f1_score as f1score
#from livelossplot.tf_keras import PlotLossesCallback

import preproccessing as prep
EPOCHS = 6
tf.compat.v1.enable_eager_execution()
def get_f1(y_true, y_pred):
    f1 = 0.0
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    pred, true = [], []
    for line in y_pred:
        for probs in line:
            pred.append(np.argmax(probs))
    for line in y_true:
        for probs in line:
            true.append(np.argmax(probs))
    labels = [float(i) for i in range(0, 38)]
    f1 = f1score(y_true=true, y_pred=pred, average='micro', labels=labels)
    #print(labels)
    #print(f1)
    return f1


class LSTM_Base:
    def __init__(self, run_training=None):
        self.data = prep.Dataloader(train_path='train.json', test_path='test.json')
        self.prepper = prep.LSTMPrepper()
        self.model = self.make_model()
        if run_training:
            self.train()

    def make_model(self):
        max_len = self.prepper.max_len
        dropout = 0.1
        input = Input(shape=((max_len),))
        emb = Embedding(input_dim=len(self.prepper.tokenizer.word_index) + 1, output_dim=100, weights=[self.prepper.embedding], input_length=140, mask_zero=True)(input)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(emb)
        drop1 = Dropout(dropout)(lstm1)
        td = TimeDistributed(Dense(units=37, activation="softmax"))(drop1)
        model = Model(input, td)
        model.summary()
        adam = keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])#, get_f1])#, run_eagerly=True)
        return model

    def train(self):
        early_stop = EarlyStopping(monitor='accuracy', patience=5, verbose=0, mode='max', restore_best_weights=False)   #monitor='val_accuracy'
        callbacks = [early_stop]    #PlotLossesCallback() for plotting
        history=self.model.fit(self.prepper.x_train_padded, np.array(self.prepper.y_train), validation_split=0.2, batch_size=32, epochs=EPOCHS,verbose=1,callbacks=callbacks)
        self.model.evaluate(self.prepper.x_test_padded, np.array(self.prepper.y_test))
        self.model.save_weights(os.path.join('models', 'lstm_base.h5'))
        result = self.model.predict(self.prepper.x_test_padded)
        pred, true = [], []
        for line in result:
            for probs in line:
                pred.append(np.argmax(probs))
        for line in self.prepper.y_test:
            for probs in line:
                true.append(np.argmax(probs))
        labels = [float(i) for i in range(0, 38)]
        print(true)
        print(pred)
        f1 = f1score(y_true=true, y_pred=pred, average='micro', labels=labels)
        print(f1)
        all_tags = tf.argmax(result, axis=-1)
        #for tags in all_tags:
         #   t = eval(tags[0])
          #  labels = [self.prepper.dl.lookup_table[eval(tag)] for tag in tags]
        #print(labels)
#lstm_base = LSTM_Base(run_training=True)