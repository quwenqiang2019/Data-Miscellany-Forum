import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Dropout, Attention, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Input, SimpleRNN
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

reviews = pd.read_csv('E:\data\IMDB Dataset.csv')
print(reviews.head())
# 假设 reviews 已经读进来
pos_df = reviews[reviews['sentiment']=="positive"]
neg_df = reviews[reviews['sentiment']=="negative"]

# 每类各保留 N 条，例如 5000
N = 10000
pos_df = pos_df.sample(n=N, random_state=42)
neg_df = neg_df.sample(n=N, random_state=42)

# 合并并打乱
reviews = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

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
test_sequences = tokenizer.texts_to_sequences(X_test)

sequence_length = 200
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')

embedding_dim = 16
lstm_out = 32


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))
model.add(SimpleRNN(lstm_out))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# #训练模型
# model.fit(train_padded, y_train, epochs=5, batch_size=128)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
history = model.fit(train_padded, y_train, epochs=100, validation_data=(test_padded, y_test),
                    callbacks=[early_stopping, model_checkpoint], batch_size=64)

# 在测试集上最终评估
test_loss, test_acc = model.evaluate(test_padded, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")


# 预测测试集的概率
y_pred_prob = model.predict(test_padded)
y_pred_prob = y_pred_prob.flatten()

# 将预测的概率转换为类别
y_pred = np.where(y_pred_prob > 0.5, 1, 0)

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()