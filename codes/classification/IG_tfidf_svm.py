from __future__ import unicode_literals
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from hazm import stopwords_list, Normalizer, word_tokenize
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gensim.models import Word2Vec


class Embedding():

    def __init__(self):
        self.polarity = ['مثبت', 'منفی', 'خنثی', 'پست عینی']
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    def word_2vec(self, input_docs, size=300, window=5, min_count=1, workers=8, sg=1):
        model = Word2Vec(input_docs, size=size, window=window, min_count=min_count, workers=workers, sg=sg)
        return model

    def word2vec_mean_representation_sentences(self, model, input_doc):
        word_vector_size = model.wv.vector_size
        doc_size = len(input_doc)
        vocabs = model.wv.vocab
        doc_mean_matrix = np.zeros((doc_size, word_vector_size))

        for doc_index, doc in enumerate(input_doc):
            for word in doc:
                if word in vocabs:
                    doc_mean_matrix[doc_index, :] += model.wv[word]
            doc_mean_matrix[doc_index, :] = np.divide(doc_mean_matrix[doc_index, :], len(doc))

        return doc_mean_matrix

    def test_word2vec(self, polarity_contents, polarity_labels):
        polarity_contents = Embedding.hazm_sentences_tokenize(polarity_contents, joined=False,numpy_array=False)
        word2vec_model = self.word_2vec(polarity_contents)
        print('number of words : {}'.format(len(word2vec_model.wv.vocab)))
        term_doc_train = self.word2vec_mean_representation_sentences(word2vec_model, polarity_contents)
        final_train_labels = Embedding.multi_label_to_one_label(polarity_labels)
        C = [1]
        kernel = ['linear']
        predict_labels = Embedding.svm_model_train(term_doc_train, final_train_labels, C_list=C, kernels_list=kernel)
        accuracy = np.where(predict_labels == final_train_labels)[0].shape[0] / predict_labels.shape[0]
        print('model accurcay: {}'.format(accuracy))
        return

    @staticmethod
    def tfidf_embedding(x_train, x_test, test_size, vocab=None):
        # x_train, x_test, y_train, y_test = train_test_split(sentences_list, labels, test_size=test_size)
        tfidf = TfidfVectorizer(vocabulary=vocab)
        term_doc_train = tfidf.fit_transform(raw_documents=x_train)
        term_doc_test = tfidf.transform(raw_documents=x_test)
        return term_doc_train, term_doc_test

    @staticmethod
    def tfidf_embedding_train(x_train, vocab=None):
        tfidf = TfidfVectorizer(vocabulary=vocab, min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
        term_doc_train = tfidf.fit_transform(raw_documents=x_train)
        return term_doc_train

    # this function will calculate accuracy on common train test data with maryam(bert model)
    def svm_with_common_data(self, vocabs):
        print("you have been entered in svm classifier with common dataset between bert and svm")
        train = pd.read_csv('../../data/statistics/common_train_test/polarity_no_multi_label_train.csv')
        valid = pd.read_csv('../../data/statistics/common_train_test/polarity_no_multi_label_valid.csv')
        test = pd.read_csv('../../data/statistics/common_train_test/polarity_no_multi_label_test.csv')

        train_valid = pd.concat([train, valid], ignore_index=True)
        train_valid_test = pd.concat([train_valid, test], ignore_index=True)

        train_valid_size = train_valid.shape[0]
        test_size = test.shape[0]

        contents = train_valid_test['Content']
        contents = self.hazm_sentences_tokenize(contents)
        labels = train_valid_test[self.polarity].to_numpy()

        X_train, X_test = self.tfidf_embedding(contents[:train_valid_size],
                                                             contents[train_valid_size:],
                                                             test_size=0,
                                                             vocab=vocabs)
        one_labels = Embedding.multi_label_to_one_label(labels)

        Y_train = one_labels[:train_valid_size]
        Y_test = one_labels[train_valid_size:]
        post_ids = train_valid['Post Id']
        test_post_ids = test['Post Id']

        # ____________ cross validation part ______________
        fold_numbers = 2
        fold_index = 0
        kf = KFold(n_splits=fold_numbers, shuffle=False)
        for train_index, test_index in kf.split(X_train):
            fold_index += 1
            print(fold_index)
            x_train, x_test = X_train[train_index], X_train[test_index]  # split 45% as train data
            y_train, y_test = Y_train[train_index], Y_train[test_index]  # split 45% as test data
            # __________ classification part ___________
            c = 1
            kernel = 'linear'
            clf = SVC(C=c, kernel=kernel, probability=True)
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)  # these are the features for ensemble model
            y_prob = clf.predict_proba(x_test).max(axis=1)
            acc = accuracy_score(y_test, y_pred)
            print("accuracy on 45% test data", acc)

            main_y_pred = clf.predict(X_test)  # accuracy on 10% test data
            main_acc = accuracy_score(Y_test, main_y_pred)
            main_y_prob = clf.predict_proba(X_test).max(axis=1)
            print("accuracy on 10% test data", main_acc)

            x_train_pred = clf.predict(x_train)
            x_train_acc = accuracy_score(y_train, x_train_pred)
            print("accuracy on 45% train data", x_train_acc)

            # write results on new file
            self.write_results(post_ids[test_index], y_pred, y_test, y_prob, 'svm_common_dataset_result' + str(fold_index))
            self.write_results(test_post_ids, main_y_pred, Y_test, main_y_prob, '10%_test_result' + str(fold_index))


    @staticmethod
    def write_results(data_ids, data_prediction, data_real, probability, saved_file_name):
        result = np.concatenate((np.array(data_ids).reshape(-1, 1),
                                 np.array(data_prediction).reshape(-1, 1),
                                 np.array(data_real).reshape(-1, 1),
                                 np.array(probability).reshape(-1, 1)),
                                axis=1)

        result_df = pd.DataFrame(result, columns=['Post Id', 'predicted', 'real', 'probability'])

        os.makedirs('../../data/result/'.format(root_dir), exist_ok=True)
        with open('../../data/result/' + saved_file_name + '.csv'.format(root_dir), 'w') as result_file:
            result_df.to_csv(result_file)

    @staticmethod
    def svm_model_train(x_train, y_train, C_list, kernels_list):
        print("you have been entered in in svm train classifier")
        clf = SVC(C=C_list[0], kernel=kernels_list[0], probability=True)
        clf.fit(x_train, y_train)
        predicted_labels = clf.predict(x_train)
        return predicted_labels

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
        # param_grid = {'C': C_list, 'kernel': kernels_list}
        # clf = GridSearchCV(SVC(class_weight=cls_weight, probability=True), param_grid)
        # print("best estimator: ", clf.best_estimator_, clf.best_score_, clf.best_params_)
        clf = SVC(C=C_list[0], kernel=kernels_list[0], probability=True)
        clf.fit(x_train, y_train)
        predicted_labels = clf.predict(x_test)
        probability = clf.predict_proba(x_test)
        score = clf.score(x_test, y_test)
        print(score)
        return score

    def svm_all(self, polarity_ids, polarity_contents):
        polarity_ids = polarity_ids.to_numpy()
        polarity_contents = Embedding.hazm_sentences_tokenize(polarity_contents)
        term_doc_train = self.tfidf_embedding_train(polarity_contents, vocab=polarity_vocabs)
        final_train_labels = Embedding.multi_label_to_one_label(polarity_labels)

        C = [1]
        kernel = ['linear']
        predict_labels = Embedding.svm_model_train(term_doc_train, final_train_labels, C_list=C, kernels_list=kernel)
        accuracy = np.where(predict_labels == final_train_labels)[0].shape[0] / predict_labels.shape[0]
        print('model accurcay: {}'.format(accuracy))

        final_train_labels = final_train_labels.reshape((final_train_labels.shape[0], 1))
        predict_labels = predict_labels.reshape((predict_labels.shape[0], 1))
        polarity_ids_reshaped = polarity_ids.reshape((polarity_ids.shape[0], 1))
        result = np.concatenate((polarity_ids_reshaped, predict_labels, final_train_labels), axis=1)
        result_df = pd.DataFrame(result, columns=['Post Id', 'predicted', 'real'])
        os.makedirs('{}/data/result/'.format(root_dir), exist_ok=True)
        with open('{}/data/result/svm_result.csv'.format(root_dir), 'w') as result_file:
            result_df.to_csv(result_file)

    def seperate_content_lables(self, filename, post_id, content, label_fields):
        df = pd.read_csv(filename)
        post_id = df.loc[:, post_id]
        content = df.loc[:, content]
        labels = df.loc[:, label_fields].to_numpy()
        return post_id, content, labels

    @staticmethod
    def hazm_sentences_tokenize(sentences, joined=True, numpy_array=True):
        normalizer = Normalizer(persian_numbers=False)
        normalize_content = []
        hazm_stopwords = stopwords_list()
        for elem in sentences:
            normalize_sentence = normalizer.normalize(elem)
            sentence_words = word_tokenize(normalize_sentence)
            without_stop_words = [elem for elem in sentence_words if elem not in hazm_stopwords]
            if joined:
                normalize_content.append(' '.join(without_stop_words))
            else:
                normalize_content.append(without_stop_words)
        if numpy_array:
            return np.array(normalize_content)
        else:
            return normalize_content


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.dirname(path)
root_dir = os.path.dirname(parent_dir)

emotions_file = '{}/data/statistics/emotions_no_multi_label.csv'.format(root_dir)
polarity_file = '{}/data/statistics/polarity_no_multi_label_plus_eini_label.csv'.format(root_dir)

polarity_vocabs = pd.read_csv('../../data/vectors/IG_features_polarity.csv')['word']
emotions_vocabs = pd.read_csv('../../data/vectors/IG_features_emotion.csv')['word']
# Embedding('{}/data/manual_tag/statistics/clean_labeled_data.csv'.format(root_dir))

embedding_instance = Embedding()

# calculate accuracy on common train test data with maryam
embedding_instance.svm_with_common_data(polarity_vocabs)

############################################################
# ################### polarity_section #####################
############################################################

polarity_ids, \
polarity_contents, \
polarity_labels = embedding_instance.seperate_content_lables(polarity_file,
                                                             'Post Id',
                                                             'Content',
                                                             embedding_instance.polarity)
# embedding_instance.test_word2vec(polarity_contents, polarity_labels)
# embedding_instance.svm_all(polarity_ids, polarity_contents)

### stop_words
polarity_ids = polarity_ids.to_numpy()
polarity_contents = Embedding.hazm_sentences_tokenize(polarity_contents)
# term_doc_train = Embedding().tfidf_embedding_train(polarity_contents, vocab=polarity_vocabs)
# final_train_labels = Embedding.multi_label_to_one_label(polarity_labels)

# ____________ cross validation part ______________
average_scores = 0
fold_numbers = 10
kf = KFold(n_splits=fold_numbers, shuffle=False)
for train_index, test_index in kf.split(polarity_contents):
    x_train, x_test = polarity_contents[train_index], polarity_contents[test_index]
    y_train, y_test = polarity_labels[train_index], polarity_labels[test_index]

    term_doc_train, term_doc_test = Embedding().tfidf_embedding(x_train, x_test, 0.1, vocab=polarity_vocabs)
    final_train_labels = Embedding.multi_label_to_one_label(y_train)
    final_test_labels = Embedding.multi_label_to_one_label(y_test)
    # __________ classification part ___________
    C = [1]
    kernel = ['linear']
    score = Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels, C_list=C,
                                kernels_list=kernel,
                                cls_weight='balanced')
    average_scores += score
print("average score", average_scores / 10)

############################################################
# ################### emotion_section #####################
############################################################

# emotion_ids, emotions_content, emotions_labels = embedding_instance.seperate_content_lables(emotions_file,
#                     'Post Id','Content', embedding_instance.emotional_tags)
# # ____________ cross validation part ______________
# fold_numbers = 10
# kf = KFold(n_splits=fold_numbers, shuffle=False)
# average_scores_emotions = 0
# for train_index, test_index in kf.split(emotions_content):
#     x_train, x_test = emotions_content[train_index], emotions_content[test_index]
#     y_train, y_test = emotions_labels[train_index], emotions_labels[test_index]
#     term_doc_train, term_doc_test = Embedding().tfidf_embedding(x_train, x_test, 0.1, vocab=emotions_vocabs)
#     final_train_labels = Embedding.multi_label_to_one_label(y_train)
#     final_test_labels = Embedding.multi_label_to_one_label(y_test)
#     # __________ classification part ___________
#     C = [1]
#     kernel = ['linear']
#     score = Embedding.svm_model(term_doc_train, term_doc_test, final_train_labels, final_test_labels, C_list=C,
#                                 kernels_list=kernel,
#                                 cls_weight='balanced')
#     average_scores_emotions += score
# print(average_scores_emotions / 10)
