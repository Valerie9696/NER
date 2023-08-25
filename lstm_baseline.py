import numpy as np
import tensorflow as tf
import keras
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback

import preproccessing as prep

class LSTM_Base:
    def __init__(self):
        self.data = prep.Dataloader()
        self.prepper = prep.LSTMPrepper()
        self.input = Input(shape=(104))
        self.model = self.make_model()
        self.train()

    def make_model(self):
        model = Embedding(input_dim=len(self.data.vocabulary),output_dim=104,input_length=140)(input)
        model = SpatialDropout1D(0.1)(model)
        model=Bidirectional(LSTM(units=150,return_sequences=True, recurrent_dropout=0.1))(model)
        output = TimeDistributed(Dense(units=len(self.data.unique_tags), activation="softmax"))(model)
        model=Model(input,output)
        model.summary()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model

    def train(self):
        early_stop = EarlyStopping(monitor='val_accuracy', patience=1, verbose=0, mode='max',
                                   restore_best_weights=False)
        callbacks = [PlotLossesCallback(), early_stop]
        history=self.model.fit(self.prepper.x_train_padded,np.array(self.prepper.y_train_padded),validation_split=0.2,batch_size=32,epochs=2,verbose=1,callbacks=callbacks)
        self.model.evaluate(self.prepper.x_test_padded,np.array(self.prepper.y_test_padded))
        self.model.save_weights('final_model.h5')