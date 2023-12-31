# -*- coding: utf-8 -*-
"""Combination

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Z7YiW-43_g2jkeFA0qs4M4a7aUqqQi_
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import tensorflow as tf

bert_cnn = pd.read_csv('drive/My Drive/corona/BERT_CNN_Labeled_data.csv')
print(len(bert_cnn))
bert_cnn_train = bert_cnn[:2489]
print(len(bert_cnn_train))
bert_cnn_valid = bert_cnn[2489: 2766]
bert_cnn_valid = bert_cnn_valid.reset_index(drop=True)
print(len(bert_cnn_valid))
bert_cnn_test = bert_cnn[2766:]
bert_cnn_test = bert_cnn_test.reset_index(drop=True)
print(len(bert_cnn_test))

svm_train_valid = pd.read_csv('drive/My Drive/corona/svm_common_dataset_result.csv')
svm_train = svm_train_valid[:2489]
svm_train = svm_train.reset_index(drop=True)
svm_valid = svm_train_valid[2489: 2766]
svm_valid = svm_valid.reset_index(drop=True)
svm_test = pd.read_csv('drive/My Drive/corona/test_result.csv')

# print(svm_train)
# print(bert_cnn_train)
# print(svm_valid)
# print(bert_cnn_valid)
print(svm_test)
print(bert_cnn_test)

y_train = bert_cnn_train["real"]
y_valid = bert_cnn_valid["real"]
y_test = bert_cnn_test["real"]

X_train = []
X_valid = []
X_test = []

for index, row in bert_cnn_train.iterrows():
  X_train.append([row["predicted"], svm_train.loc[index]["predicted"]])

for index, row in bert_cnn_valid.iterrows():
  X_valid.append([row["predicted"], svm_valid.loc[index]["predicted"]])

for index, row in bert_cnn_test.iterrows():
  X_test.append([row["predicted"], svm_test.loc[index]["predicted"]])

X_train = np.asarray(X_train, dtype=np.float32)
X_valid = np.asarray(X_valid, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)
y_valid = np.asarray(y_valid, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

# from sklearn.model_selection import train_test_split
# # df = pd.read_csv('drive/My Drive/corona/bert_cnn_svm_merged_data.csv')
# df = df.sample(frac=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

def build_model():
    input = tf.keras.layers.Input(shape=(2,), name="input")
    print("input", input.shape)
    dense = tf.keras.layers.Dense(1024, activation="relu")(input)
    print("dense", dense.shape)
    pred = tf.keras.layers.Dense(4, activation="softmax")(dense)
    model = tf.keras.models.Model(inputs=input, outputs=pred)
    # adam = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model

model = build_model()

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, batch_size=400)

from sklearn.metrics import accuracy_score
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

from sklearn.metrics import accuracy_score
score = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
#
# kfold = KFold(n_splits=10, shuffle=True, random_state=7)
#
# test_acc = []
# for train, test in kfold.split(X, y):
#     model = build_model()
#     model.fit(X[train], y[train], epochs=30, batch_size=400, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X[test], y[test], verbose=0)
#     print('Test accuracy:', scores[1])
#     test_acc.append(scores[1])
#
# print("Test Accuracy: %.2f (+/- %.2f)" % (np.mean(test_acc), np.std(test_acc)))
# print(test_acc)