# %%
from __future__ import unicode_literals
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from hazm import Normalizer


class Embedding():

    def __init__(self):
        self.polarity = {0: 'مثبت', 1: 'منفی', 2: 'خنثی'}
        self.emotional_tags = {0: 'شادی', 1: 'غم', 2: 'ترس', 3: 'تنفر', 4: 'خشم', 5: 'شگفتی', 6: 'اعتماد',
                               7: 'پیش‌بینی', 8: 'سایر هیجانات', 9: 'استرس'}

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
    def svm_model(x_train, x_test, y_train, y_test):
        clf = SVC()
        clf.fit(x_train, y_train)
        predicted_array = clf.predict(x_test)
        print(np.sum(predicted_array == y_test) / len(y_test))
        return predicted_array

    def seperate_content_lables(self, filename, content, label_fields):
        df = pd.read_csv(filename)
        content = df.loc[:, content]
        print([item for item in label_fields.values()])
        labels = df.loc[:, [item for item in label_fields.values()]].to_numpy()
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

    @staticmethod
    def compute_label_sets_occurances(train_labels):
        labels, occurs = np.unique(train_labels, return_counts=True)
        return labels, dict(zip(labels, occurs))

    def doc_term(self, sentences_list, labels, test_size, n_gram: int):
        vectorizer = CountVectorizer(binary=True, ngram_range=(n_gram, n_gram))
        x_train, x_test, y_train, y_test = train_test_split(sentences_list, labels, test_size=test_size)
        y_train_one_label = Embedding.multi_label_to_one_label(y_train)
        tf_idf_vector_train = vectorizer.fit_transform(x_train)
        features = dict(enumerate(vectorizer.get_feature_names()))
        info_gain_dt_matrix = np.zeros((tf_idf_vector_train.shape[0], tf_idf_vector_train.shape[1] + 1))
        info_gain_dt_matrix[:, 1:] = tf_idf_vector_train.toarray()
        info_gain_dt_matrix[:, 0] = y_train_one_label
        self.labels_set, self.label_occurances = Embedding.compute_label_sets_occurances(y_train_one_label)
        return info_gain_dt_matrix, features

    def top_100_words(self, features, info_gain_matrix, p_c_i, tags_type):
        extracted_features = []
        words_polarity_score = []
        words = []
        info_gain = []
        arg_max_p_c_i = np.argmax(p_c_i, axis=0)
        for arg in np.argsort(info_gain_matrix)[-100:][::-1]:
            words.append(features[arg])
            if tags_type == 'emotions':
                extracted_features.append([features[arg], self.emotional_tags[arg_max_p_c_i[arg]],
                    np.round(info_gain_matrix[arg], decimals=5)])
                words_polarity_score.append(
                    {'کلمه': features[arg], 'قطبیت': self.emotional_tags[arg_max_p_c_i[arg]],
                     'امتیاز': np.round(info_gain_matrix[arg], decimals=5)})
            elif tags_type == 'polarity':
                extracted_features.append([features[arg], self.polarity[arg_max_p_c_i[arg]],
                    np.round(info_gain_matrix[arg], decimals=5)])
                words_polarity_score.append(
                    {'کلمه': features[arg], 'قطبیت': self.polarity[arg_max_p_c_i[arg]],
                     'امتیاز': np.round(info_gain_matrix[arg], decimals=5)})
            info_gain.append(info_gain_matrix[arg])
        max_info_gain = np.max(info_gain)
        min_info_gain = np.min(info_gain)
        normalize_info_gain = []
        for info_gain in info_gain:
            normalize_info_gain.append(
                np.multiply(np.add(np.divide(np.subtract(info_gain, min_info_gain),
                                             np.subtract(max_info_gain, min_info_gain)), 1), 10))
        return words_polarity_score, words, normalize_info_gain, extracted_features

    def compute_label_entropies(self, train_labels):
        labels, label_occurances = Embedding.compute_label_sets_occurances(train_labels)
        sigma = 0
        for key, value in label_occurances.items():
            prob = float(value) / self.N
            entropy = np.multiply(prob, np.log2(prob))
            sigma += entropy
        sigma *= -1
        return sigma

    # compute N(W) and N(~W)
    def compute_n_w_n_w_not(self, d_t_matrix):
        self.N = d_t_matrix.shape[0]
        temp_matrix = d_t_matrix[:, 1:]
        # todo make matrix sparse for fewer time
        n_w_matrix = np.sum(temp_matrix, axis=0)
        n_w_not_matrix = self.N - n_w_matrix
        return n_w_matrix, n_w_not_matrix

    # compute P(W) and P(~W)
    def compute_p_w_p_w_not(self, d_t_matrix):
        n_w_matrix, n_w_not_matrix = self.compute_n_w_n_w_not(d_t_matrix)
        p_w_matrix = n_w_matrix / self.N
        p_w_not_matrix = n_w_not_matrix / self.N
        return n_w_matrix, p_w_matrix, n_w_not_matrix, p_w_not_matrix

    def information_gain(self, d_t_matrix, train_labels):
        n_w_matrix, p_w_matrix, n_w_not_matrix, p_w_not_matrix = self.compute_p_w_p_w_not(d_t_matrix)
        sigma = np.zeros(d_t_matrix.shape[1] - 1)
        # compute sigma P(Ci|w) * log(P(Ci|w) + P(Ci|~w) * log(P(Ci|~w)
        p_c_i_w_matrix_save = []
        temp_matrix = d_t_matrix[:, 1:]
        for label in self.labels_set:
            poet_matrix = np.zeros(d_t_matrix.shape[0])
            poet_rows = np.where(d_t_matrix[:, 0] == label)
            poet_matrix[poet_rows] = 1
            poet_matrix = poet_matrix.astype(bool)
            poet_matrix = poet_matrix.reshape(poet_matrix.shape[0], 1)

            # temp_matrix[:, 0] = 0
            # compute N(Wi)
            n_w_i_matrix = np.sum(temp_matrix, where=poet_matrix, axis=0)
            # compute P(Ci|w) = N(Wi) / N(W)
            p_c_i_w_matrix = n_w_i_matrix / n_w_matrix
            p_c_i_w_matrix_save.append(p_c_i_w_matrix)
            # compute P(Ci|w) * log(P(Ci|w)
            p_c_i_w_matrix_log = np.where(p_c_i_w_matrix != 0, np.log2(p_c_i_w_matrix), 0)
            p_c_i_w_matrix_entropy = np.multiply(p_c_i_w_matrix, p_c_i_w_matrix_log)

            # compute N(~Wi)
            n_w_i_not_matrix = self.label_occurances[label] - n_w_i_matrix
            # compute P(Ci|~w) = N(~Wi) / N(~W)
            p_c_i_w_not_matrix = n_w_i_not_matrix / n_w_not_matrix
            # compute P(Ci|~w) * log(P(Ci|~w)

            p_c_i_w_not_matrix_log = np.where(p_c_i_w_not_matrix != 0, np.log2(p_c_i_w_not_matrix), 0)
            p_c_i_w_not_matrix_entropy = np.multiply(p_c_i_w_not_matrix, p_c_i_w_not_matrix_log)

            sigma = sigma + np.multiply(p_w_matrix, p_c_i_w_matrix_entropy) + np.multiply(p_w_not_matrix,
                                                                                          p_c_i_w_not_matrix_entropy)

        sigma = sigma + self.compute_label_entropies(train_labels)
        return sigma, np.array(p_c_i_w_matrix_save)

    def create_features_dataframe(self, unigram, bigram, trigram, name):
        unigram = np.array(unigram)
        bigram = np.array(bigram)
        trigram = np.array(trigram)

        features = np.concatenate((unigram, bigram, trigram))
        features_df = pd.DataFrame(features, columns=['word', 'polarity', 'score'])
        features_df.to_csv('../../data/vectors/IG_features_'+name+'.csv')
        return features_df



sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.dirname(path)
root_dir = os.path.dirname(parent_dir)
emotions_file = '{}/data/statistics/emotions.csv'.format(root_dir)
polarity_file = '{}/data/statistics/polarity.csv'.format(root_dir)
embedding_instance = Embedding()


emotion_contents, emotion_labels = embedding_instance.seperate_content_lables(emotions_file, 'Content',
                                                                              embedding_instance.emotional_tags)
term_doc_train, term_doc_test, train_labels, test_labels = Embedding.tfidf_embedding(emotion_contents, emotion_labels,
                                                                                     0.1)
final_train_labels = Embedding.multi_label_to_one_label(train_labels)
final_test_labels = Embedding.multi_label_to_one_label(test_labels)

# _________ extract features using info gain __________
doc_term_matrix, features = embedding_instance.doc_term(emotion_contents, emotion_labels, 0.1, n_gram=1)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, unigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'emotions')
print(unigram_features)

doc_term_matrix, features = embedding_instance.doc_term(emotion_contents, emotion_labels, 0.1, n_gram=2)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, bigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'emotions')
print(bigram_features)

doc_term_matrix, features = embedding_instance.doc_term(emotion_contents, emotion_labels, 0.1, n_gram=3)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, trigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'emotions')
print(trigram_features)

features_df = embedding_instance.create_features_dataframe(unigram_features, bigram_features, trigram_features, name='emotion')
# print(features_df['polarity'])
# _______________ end of feature extraction part _________________________

# %%
Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels)



polarity_contents, polarity_labels = embedding_instance.seperate_content_lables(polarity_file, 'Content',
                                                                                embedding_instance.polarity)
# _________ extract features using info gain __________
doc_term_matrix, features = embedding_instance.doc_term(polarity_contents, polarity_labels, 0.1, n_gram=1)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, unigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'polarity')
print(unigram_features)

doc_term_matrix, features = embedding_instance.doc_term(polarity_contents, polarity_labels, 0.1, n_gram=2)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, bigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'polarity')
print(bigram_features)

doc_term_matrix, features = embedding_instance.doc_term(polarity_contents, polarity_labels, 0.1, n_gram=3)
info_gain_matrix, p_c_i = embedding_instance.information_gain(doc_term_matrix, doc_term_matrix[:, 0])
words_polarity_score, words, normalize_info_gain, trigram_features = embedding_instance.top_100_words(features, info_gain_matrix, p_c_i,
    'polarity')
print(trigram_features)

features_df = embedding_instance.create_features_dataframe(unigram_features, bigram_features, trigram_features, name='polarity')
# print(features_df['polarity'])
# _______________ end of feature extraction part _________________________


term_doc_train, term_doc_test, train_labels, test_labels = Embedding.tfidf_embedding(polarity_contents, polarity_labels,
                                                                                     0.1)
final_train_labels = Embedding.multi_label_to_one_label(train_labels)
final_test_labels = Embedding.multi_label_to_one_label(test_labels)
Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels)
