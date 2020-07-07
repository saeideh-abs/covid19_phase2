from __future__ import unicode_literals

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance


class Embedding():

    def __init__(self):
        self.polarity = ['مثبت', 'منفی', 'خنثی']
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    @staticmethod
    def tfidf_embedding(sentences_list, labels, test_size):
        x_train, x_test, y_train, y_test = train_test_split(sentences_list, labels, test_size=test_size)
        tfidf = TfidfVectorizer()
        term_doc_train = tfidf.fit_transform(raw_documents=x_train)
        term_doc_test = tfidf.transform(raw_documents=x_test)
        return term_doc_train, term_doc_test, y_train, y_test

    def seperate_content_lables(self, filename, content, label_fields):
        df = pd.read_csv(filename)
        content = df.loc[:, content]
        labels = df.loc[:, label_fields].to_numpy()
        return content, labels

    def binary_relevance_classifiers(self, x_train, y_train, x_test, y_test):
        clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        print(clf.best_params_, clf.best_score_)
        print(prediction)
        return accuracy_score(y_test, prediction)


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.dirname(path)
root_dir = os.path.dirname(parent_dir)
print(root_dir)
data_file = '{}/data/manual_tag/clean_labeled_data.csv'.format(root_dir)

embedding_instance = Embedding()
contents, labels = embedding_instance.seperate_content_lables(data_file, 'Content', embedding_instance.polarity)

term_doc_train, term_doc_test, train_labels, test_labels = Embedding.tfidf_embedding(contents, labels, 0.1)

parameters = [
    # {
    #     'classifier': [MultinomialNB()],
    #     'classifier__alpha': [0.7, 1.0],
    # },
    # {
    #     'classifier': [SVC()],
    #     'classifier__kernel': ['rbf', 'linear'],
    # },
    {
        'classifier': [RandomForestClassifier()]
    },

]

print("accuracy:",
      embedding_instance.binary_relevance_classifiers(term_doc_train, train_labels, term_doc_test, test_labels))
