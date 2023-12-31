from __future__ import unicode_literals
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from hazm import Normalizer
from sklearn.ensemble import RandomForestClassifier


class Embedding():

    def __init__(self):
        self.polarity = ['مثبت', 'منفی', 'خنثی']
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    @staticmethod
    def tfidf_embedding(x_train, x_test, vocab=None):
        tfidf = TfidfVectorizer(vocabulary=vocab, min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
        term_doc_train = tfidf.fit_transform(raw_documents=x_train)
        term_doc_test = tfidf.transform(raw_documents=x_test)
        return term_doc_train, term_doc_test

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
    def random_forest(x_train, x_test, y_train, y_test):
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(x_train, y_train)
        predicted_array = clf.predict(x_test)
        score = clf.score(x_test, y_test)
        return score

    def cross_validation(self, X, y, fold_num, vocabs=None, shuffle=False):
        scores = []
        fold_numbers = fold_num
        kf = KFold(n_splits=fold_numbers, shuffle=shuffle)
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            term_doc_train, term_doc_test = self.tfidf_embedding(x_train, x_test, vocab=vocabs)
            final_train_labels = self.multi_label_to_one_label(y_train)
            final_test_labels = self.multi_label_to_one_label(y_test)
            # __________ classification part ___________
            score = self.random_forest(term_doc_train, term_doc_test, final_train_labels, final_test_labels)
            print(score)
            scores.append(score)
        print("average score", np.mean(scores))
        return scores

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
print(root_dir)
emotions_file = '{}/data/statistics/emotions_no_multi_label.csv'.format(root_dir)
polarity_file = '{}/data/statistics/polarity_no_multi_label.csv'.format(root_dir)
# Embedding('{}/data/manual_tag/statistics/clean_labeled_data.csv'.format(root_dir))
embedding_instance = Embedding()

####################################################
# #############   polarity data   ###################
####################################################
polarity_contents, polarity_labels = embedding_instance.seperate_content_lables(polarity_file, 'Content',
                                                                                embedding_instance.polarity)

# ____________ cross validation part ______________
embedding_instance.cross_validation(polarity_contents, polarity_labels, fold_num=10, shuffle=True)

####################################################
# #############   emotion data   ###################
####################################################
emotion_contents, emotion_labels = embedding_instance.seperate_content_lables(emotions_file, 'Content',
                                                                              embedding_instance.emotional_tags)
# ____________ cross validation part ______________
embedding_instance.cross_validation(emotion_contents, emotion_labels, fold_num=5, shuffle=True)

