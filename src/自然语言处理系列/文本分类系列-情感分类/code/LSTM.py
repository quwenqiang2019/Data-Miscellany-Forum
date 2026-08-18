import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint


reviews = pd.read_csv('E:\data\IMDB Dataset.csv')
print(reviews.head())

# 将影评情感转为0和1的数值，并将影评和情感转化为numpy数组，最后划分训练集和测试集。
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


# 构建分词器，构建单词索引，将字符串转化成整数索引组成的列表
vocab_size = 10000
oov_tok = "<OOV>" # 参数oov_tok指定一个特殊的标记，用于表示在词汇表中未出现的词语。
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
# 将整数列表转化为二维数值张量，相同的操作对测试集再执行一遍。
sequence_length = 200
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')


embedding_dim = 16
lstm_out = 32
model = Sequential()
# 定义Embedding词嵌入层。这是因为深度学习模型只能处理数值张量类型的数据。16表示的是词向量维度。
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))
# 定义一个双向LSTM层，定义一个relu激活的全连接层，定义一个使用sigmoid激活的输出层。
model.add(Bidirectional(LSTM(lstm_out)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型自动判断迭代次数，防止过拟合。
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
callbacks = [EarlyStopping(patience=2), model_checkpoint_callback]
history = model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test), callbacks=callbacks)

metrics_df = pd.DataFrame(history.history)
print(metrics_df)