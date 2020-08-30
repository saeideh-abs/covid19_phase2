# -*- coding: utf-8 -*-
"""BERT_CNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l3hrXU5HYfTUD-dEIHZftUR5G-n_23bX
"""

# !pip install tensorflow==1.15.0
# !pip install bert-tensorflow==1.0.1

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import math

train = pd.read_csv('drive/My Drive/corona/eini/polarity_no_multi_label_train.csv')
test = pd.read_csv('drive/My Drive/corona/eini/polarity_no_multi_label_test.csv')
valid = pd.read_csv('drive/My Drive/corona/eini/polarity_no_multi_label_valid.csv')
data = pd.concat([train, valid, test]).reset_index(drop=True)

label_list = [0, 1, 2, 3]
sentiment_labels = ['خنثی', 'منفی', 'مثبت', 'پست عینی']

DATA_COLUMN = "Content"
LABEL_COLUMN = "label"
MAX_SEQ_LENGTH = 64

def multi_label_to_one_label(labels):
    one_labels = np.zeros(labels.shape, dtype=int)
    for i in range(labels.shape[0]):
        label_list = np.where(labels[i, :] > 0)[0]
        if len(label_list) > 1:
            random_one_label_tag = np.random.choice(label_list)
            one_labels[i, :] = 0
            one_labels[i, random_one_label_tag] = 1
        else:
            one_labels[i, :] = labels[i, :]
    final_labels = np.zeros(labels.shape[0])
    final_labels = np.argwhere(one_labels > 0)[:, 1]
    return final_labels

train["label"] = multi_label_to_one_label(np.array(train[sentiment_labels]))
test["label"] = multi_label_to_one_label(np.array(test[sentiment_labels]))
valid["label"] = multi_label_to_one_label(np.array(valid[sentiment_labels]))

data["label"] = multi_label_to_one_label(np.array(data[sentiment_labels]))
data = data[["Content", "label", "Post Id"]]
print(len(data))
print(set(list(data["label"])))

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                text_a=x[DATA_COLUMN],
                                                                                text_b=None,
                                                                                label=x[LABEL_COLUMN]), axis=1)

valid_InputExamples = valid.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                              text_a=x[DATA_COLUMN],
                                                                              text_b=None,
                                                                              label=x[LABEL_COLUMN]), axis=1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                              text_a=x[DATA_COLUMN],
                                                                              text_b=None,
                                                                              label=1), axis=1)

data_InputExamples = data.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                              text_a=x[DATA_COLUMN],
                                                                              text_b=None,
                                                                              label=1), axis=1)

# This is a path to multilingual cased version of BERT
# BERT_MODEL_HUB = "multi_cased_L-12_H-768_A-12"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
valid_features = bert.run_classifier.convert_examples_to_features(valid_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
data_features = bert.run_classifier.convert_examples_to_features(data_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)

    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    output_layer = bert_outputs["pooled_output"] 
    sequence_outputs = bert_outputs["sequence_output"]

    hidden_size = output_layer.shape[-1].value
    print("hidden_size", hidden_size)

    # Create our own layers to tune for politeness data.
    with tf.variable_scope("loss"):
        l_reshape = tf.reshape(sequence_outputs, [-1, MAX_SEQ_LENGTH, hidden_size])
        print("l_reshape", l_reshape.shape)
        l_cov1 = tf.layers.conv1d(                          
              inputs=l_reshape,
              filters=1024,
              kernel_size=4,
              activation=tf.nn.relu)
        print("l_cov1", l_cov1.shape)
        l_pool1 = tf.layers.max_pooling1d(inputs=l_cov1, pool_size=4, strides=1)
        print("l_pool1", l_pool1)
        l_flat = tf.layers.flatten(inputs=l_pool1)
        print("l_flat", l_flat)

        # Dropout helps prevent overfitting
        droup_out1 = tf.nn.dropout(l_flat, keep_prob=0.9)
        print("droup_out1", droup_out1)
        log_probs = tf.layers.dense(inputs=droup_out1, units=num_labels, activation=tf.nn.log_softmax)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.

        if is_predicting:
            return (predicted_labels, log_probs, sequence_outputs, output_layer)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs, sequence_outputs, output_layer)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, probs, sequence_outputs, output_layer) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)

            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])

            tf.identity(loss, name='train_loss')
            tf.summary.scalar('train_loss', loss)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                #         f1_score = tf.contrib.metrics.f1_score(
                #             label_ids,
                #             predicted_labels)
                #         auc = tf.metrics.auc(
                #             label_ids,
                #             predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                #         true_pos = tf.metrics.true_positives(
                #             label_ids,
                #             predicted_labels)
                #         true_neg = tf.metrics.true_negatives(
                #             label_ids,
                #             predicted_labels)
                #         false_pos = tf.metrics.false_positives(
                #             label_ids,
                #             predicted_labels)
                #         false_neg = tf.metrics.false_negatives(
                #             label_ids,
                #             predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    #             "f1_score": f1_score,
                    #             "auc": auc,
                                "precision": precision,
                                "recall": recall,
                    #             "true_positives": true_pos,
                    #             "true_negatives": true_neg,
                    #             "false_positives": false_pos,
                    #             "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, probs, sequence_outputs, output_layer) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': probs,
                'labels': predicted_labels,
                'sequence_outputs':sequence_outputs,
                'output_layer':output_layer
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 8.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
print("num_train_steps", num_train_steps)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    # model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

# Create an input function for validation. drop_remainder = True for using TPUs.
valid_input_fn = run_classifier.input_fn_builder(
    features=valid_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

print(f'Beginning Training!')

current_time = datetime.now()

tensors_to_log = {
    'train_accuracy': 'train_accuracy',
    'train_loss': 'train_loss'
}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

# train
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print("Training took time ", datetime.now() - current_time)

#validation
estimator.evaluate(input_fn=valid_input_fn, steps=None)

from sklearn.metrics import accuracy_score

# Create an input function for test. drop_remainder = True for using TPUs.
predict_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

#prediction
predictions = estimator.predict(predict_input_fn)

# evaluate prediction results
pred_labels = []
for post_id, content, pred in zip(test["Post Id"], test["Content"], predictions):
  pred_labels.append(pred['labels'])
print(pred_labels)
# print evaluation metrics value
print("test accuracy:", accuracy_score(test[LABEL_COLUMN].tolist(), pred_labels))

# Create an input function for test. drop_remainder = True for using TPUs.
predict_input_fn = run_classifier.input_fn_builder(features=valid_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

#prediction
predictions = estimator.predict(predict_input_fn)

pred_labels = []
for post_id, real, pred in zip(data["Post Id"], data["label"], predictions):
  pred_labels.append([post_id, sentiment_labels[pred['labels']], real])
print(len(pred_labels))
for i in range(len(pred_labels)):
  if pred_labels[i][1] == 'مثبت':
    pred_labels[i][1] = 0
  elif pred_labels[i][1] == 'منفی':
    pred_labels[i][1] = 1
  elif pred_labels[i][1] == 'خنثی':
    pred_labels[i][1] = 2
  else:
    pred_labels[i][1] = 3

df = pd.DataFrame(pred_labels, columns=["postId", "predicted", "real"] )
df.to_csv("BERT_CNN_Labeled_valid.csv")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

MAX_SEQ_LENGTH = 64
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 8.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
SPLIT = 10
SEED = 7
LABEL_COLUMN = "label"
DATA_COLUMN = "Content"
np.random.seed(SEED)
kfold = StratifiedKFold(n_splits=SPLIT, shuffle=False, random_state=SEED)
test_recall = []
test_precision = []
test_acc = []
tokenizer = create_tokenizer_from_hub_module()
for train_idx, test_idx in kfold.split(data, data[LABEL_COLUMN].to_numpy()):
  train_InputExamples = data.iloc[train_idx].apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                text_a=x[DATA_COLUMN],
                                                                                text_b=None,
                                                                                label=x[LABEL_COLUMN]), axis=1)

  test_InputExamples = data.iloc[test_idx].apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                text_a=x[DATA_COLUMN],
                                                                                text_b=None,
                                                                                label=1), axis=1)
  # # Convert our train and test features to InputFeatures that BERT understands.
  train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
  test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
  num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
  print("num_train_steps", num_train_steps)
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
  # # Specify output directory and number of checkpoint steps to save
  run_config = tf.estimator.RunConfig(save_summary_steps=SAVE_SUMMARY_STEPS, save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
  model_fn = model_fn_builder(num_labels=len(label_list), learning_rate=LEARNING_RATE, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE})
  # # Create an input function for training. drop_remainder = True for using TPUs.
  train_input_fn = bert.run_classifier.input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=True,
      drop_remainder=False)

  current_time = datetime.now()
  tensors_to_log = {
      'train_accuracy': 'train_accuracy',
      'train_loss': 'train_loss'
  }
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

  # train
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  valid_input_fn = run_classifier.input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=False)

  # # Create an input function for test. drop_remainder = True for using TPUs.
  predict_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)

  # evaluate prediction results
  pred_labels = []
  test_text_list = data.iloc[test_idx][DATA_COLUMN].tolist()

  for sentence, pred in zip(test_text_list, predictions):
    pred_labels.append(pred['labels'])

  # print evaluation metrics value
  test_acc.append(accuracy_score(data.iloc[test_idx][LABEL_COLUMN].tolist(), pred_labels))
  print('=' * 100)
  print("Training took time ", datetime.now() - current_time)
  print(estimator.evaluate(input_fn=valid_input_fn, steps=None))
  print('-' * 100)
  print("test accuracy:", accuracy_score(data.iloc[test_idx][LABEL_COLUMN].tolist(), pred_labels)) 
  print('=' * 100)

print("Test Accuracy: %.2f (+/- %.2f)" % (np.mean(test_acc), np.std(test_acc)))
print(test_acc)