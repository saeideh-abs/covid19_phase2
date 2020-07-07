from __future__ import unicode_literals
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from hazm import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



class Embedding():

    def __init__(self):
        self.polarity = ['مثبت', 'منفی', 'خنثی']
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    @staticmethod
    def tfidf_embedding(sentences_list, labels, test_size, vocab=None):
        x_train, x_test, y_train, y_test = train_test_split(sentences_list, labels, test_size=test_size)
        tfidf = TfidfVectorizer(vocabulary=vocab)
        term_doc_train = tfidf.fit_transform(raw_documents=x_train)
        term_doc_test = tfidf.transform(raw_documents=x_test)
        return term_doc_train, term_doc_test, y_train, y_test

    @staticmethod
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

    @staticmethod
    def svm_model(x_train, x_test, y_train, y_test, C_list, kernels_list, cls_weight):
        print("you have been enterd in svm classifier")
        param_grid = {'C': C_list, 'kernel': kernels_list}
        clf = GridSearchCV(SVC(class_weight=cls_weight, probability=True), param_grid)
        clf.fit(x_train, y_train)
        print("best estimator: ", clf.best_estimator_, clf.best_score_, clf.best_params_)
        predicted_labels = clf.predict(x_test)
        probability = clf.predict_proba(x_test)
        score = clf.score(x_test, y_test)
        print(score)
        return clf


    def seperate_content_lables(self, filename, content, label_fields):
        df = pd.read_csv(filename)
        content = df.loc[:, content]
        labels = df.loc[:, label_fields].to_numpy()
        return content, labels

    @staticmethod
    def hazm_sentences_tokenize(filename, field_name):
        df = pd.read_csv(filename)
        content = df.loc[:, field_name]
        normalizer = Normalizer(persian_numbers=False)
        normalize_content = []

        for elem in content:
            normalize_content.append(normalizer.normalize(elem))
        return content


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.dirname(path)
root_dir = os.path.dirname(parent_dir)

emotions_file = '{}/data/statistics/emotions_no_multi_label.csv'.format(root_dir)
polarity_file = '{}/data/statistics/polarity_no_multi_label.csv'.format(root_dir)
vocabs = pd.read_csv('../../data/vectors/IG_features_polarity.csv')['word']

# Embedding('{}/data/manual_tag/statistics/clean_labeled_data.csv'.format(root_dir))

embedding_instance = Embedding()
# emotion_contents, emotion_labels = embedding_instance.seperate_content_lables(emotions_file, 'Content',
#                                                                               embedding_instance.emotional_tags)
# term_doc_train, term_doc_test, train_labels, test_labels = Embedding.tfidf_embedding(emotion_contents, emotion_labels,
#                                                                                      0.1)
# final_train_labels = Embedding.multi_label_to_one_label(train_labels)
# final_test_labels = Embedding.multi_label_to_one_label(test_labels)
# Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels)

polarity_contents, polarity_labels = embedding_instance.seperate_content_lables(polarity_file, 'Content',
                                                                                embedding_instance.polarity)
term_doc_train, term_doc_test, train_labels, test_labels = Embedding.tfidf_embedding(polarity_contents, polarity_labels,
                                                                                     0.1)
final_train_labels = Embedding.multi_label_to_one_label(train_labels)
final_test_labels = Embedding.multi_label_to_one_label(test_labels)
# __________ classification part ___________
C = [0.5, 1, 10]
kernel = ['rbf', 'linear']

Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels, C_list=C, kernels_list=kernel,
                    cls_weight='balanced')
