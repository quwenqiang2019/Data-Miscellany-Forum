import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Dropout, Attention, Flatten, Layer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()



"""
由于Keras目前还没有现成的Attention层可以直接使用，我们需要自己来构建一个新的层函数。
Keras自定义的函数主要分为四个部分，分别是：
init：初始化一些需要的参数
bulid：具体来定义权重是怎么样的
call：核心部分，定义向量是如何进行运算的
compute_output_shape：定义该层输出的大小
"""

class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

reviews = pd.read_csv('E:\data\IMDB Dataset.csv')
print(reviews.head())

reviews['sentiment'] = np.where(reviews['sentiment'] == 'positive', 1, 0)
sentences = reviews['review'].to_numpy()
print(sentences)
labels = reviews['sentiment'].to_numpy()
print(labels)


X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25)
print("Training Data Input Shape: ", X_train.shape)
print("Training Data Output Shape: ", y_train.shape)
print("Testing Data Input Shape: ", X_test.shape)
print("Testing Data Output Shape: ", y_test.shape)


vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)


sequence_length = 200
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')

embedding_dim = 16
lstm_out = 32
input_ = Input(shape=[sequence_length])
layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(input_)
lstm = LSTM(units=lstm_out, return_sequences=True)(layer)
layer = Dense(128, activation='relu')(lstm)
layer = Dropout(0.2)(layer)
## 注意力机制
attention = AttentionLayer(attention_size=50)(layer)
output = Dense(1, activation='sigmoid')(attention)
model = Model(inputs=input_, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
callbacks = [EarlyStopping(patience=2), model_checkpoint_callback]
history = model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test), callbacks=callbacks)

metrics_df = pd.DataFrame(history.history)
print(metrics_df)