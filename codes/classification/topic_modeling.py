from __future__ import unicode_literals
import pandas as pd
import numpy as np
import time
import json
from hazm import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def display_current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def text_cleaner(docs):
    stop_words = open('../../resources/stopwords_list.txt', encoding="utf8").read().split('\n')
    normalizer = Normalizer(persian_numbers=False)
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    final_text = []

    for doc in docs:
        normal_text = normalizer.normalize(doc)
        doc_words = word_tokenize(normal_text)
        without_stop_words = [word for word in doc_words if word not in stop_words]
        # stem = [stemmer.stem(word) for word in without_stop_words]
        lemm = [lemmatizer.lemmatize(word).split('#')[0] for word in
                without_stop_words]  # get the past part of the lemm
        final_text.append(' '.join(lemm))
    return final_text


def hazm_sentences_tokenize(docs, joined=True, numpy_array=True):
    normalizer = Normalizer(persian_numbers=False)
    normalize_content = []
    # stop_words = stopwords_list()
    stop_words = open('../../resources/stopwords_list.txt', encoding="utf8").read().split('\n')

    for elem in tqdm(docs):
        normalize_sentence = normalizer.normalize(elem)
        sentence_words = word_tokenize(normalize_sentence)
        without_stop_words = [elem for elem in sentence_words if elem not in stop_words]
        for word in without_stop_words:
            if word == 'های':
                print(word)
            if word == 'اند':
                print(word)
            if word == 'ها':
                print(word)
            if word == 'می':
                print(word)
        if joined:
            normalize_content.append(' '.join(without_stop_words))
        else:
            normalize_content.append(without_stop_words)
    if numpy_array:
        return np.array(normalize_content)
    else:
        return normalize_content


def NMF_topic_modeling(docs, no_features, no_topics, no_top_words):
    # NMF is able to use tf-idf
    tfidf_vecs, tfidf_feature_names = build_tfidf(docs, no_features=no_features)
    print(tfidf_feature_names)
    for word in tfidf_feature_names:
        if word == 'های':
            print(word)
        if word == 'اند':
            print(word)
        if word == 'ها':
            print(word)
        if word == 'می':
            print(word)
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_vecs)
    topics_dict = return_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(topics_dict)

    # find each post is related to which topic?
    doc_topic = nmf.transform(tfidf_vecs)
    # find index of the best topic (topic with highest score) for each document
    doc_best_topic = np.argmax(doc_topic, axis=1)
    # print(len(doc_best_topic), doc_best_topic[0:20])
    return topics_dict, doc_best_topic


def LDA_topic_modeling(docs, no_features, no_topics, no_top_words):
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vecs, tf_feature_names = build_tf(docs, no_features=no_features)
    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0, n_jobs=-1).fit(tf_vecs)
    topics_dict = return_topics(lda, tf_feature_names, no_top_words)
    display_topics(topics_dict)

    # find each post is related to which topic?
    doc_topic = lda.transform(tf_vecs)
    # find index of the best topic (topic with highest score) for each document
    doc_best_topic = np.argmax(doc_topic, axis=1)
    # print(len(doc_best_topic), doc_best_topic)
    return topics_dict, doc_best_topic


def build_tfidf(docs, no_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=no_features)
    tfidf = tfidf_vectorizer.fit_transform(docs)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return tfidf, tfidf_feature_names


def build_tf(docs, no_features):
    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=5, max_features=no_features)
    tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names


def return_topics(model, feature_names, no_top_words):
    keys = []
    values = []
    for topic_idx, topic in enumerate(model.components_):
        topic_name = "topic" + str(topic_idx)
        topic_words = " ".join([feature_names[i]
                                for i in topic.argsort()[:-no_top_words - 1:-1]])
        keys.append(topic_name)
        values.append(topic_words)
    return dict(zip(keys, values))


def display_topics(topics_dict):
    for topic_name, topic_words in topics_dict.items():
        print(topic_name, ":")
        print(topic_words)


if __name__ == '__main__':
    # _______________ loading the data and preprocessing _______________
    dataset = pd.read_csv('../../data/politics.csv', na_values='')

    print("search and remove null values", display_current_time())
    nan_indices = np.where(dataset['textField_nlp_normal'].isnull())[0]  # get index of null values
    dataset = dataset.drop(nan_indices, axis=0)  # remove null values
    documents = dataset['textField_nlp_normal']

    print("preprocessing using hazm", display_current_time())
    documents = hazm_sentences_tokenize(documents)
    # documents = text_cleaner(documents)

    # _______________ topic modeling part _______________
    number_of_features = 1000
    number_of_topics = 10
    number_of_top_words = 10

    print("NMF topic modeling", display_current_time())
    nmf_topics_dictionary, nmf_document_best_topic = NMF_topic_modeling(documents, no_features=number_of_features,
                                                                        no_topics=number_of_topics,
                                                                        no_top_words=number_of_top_words)

    print("LDA topic modeling", display_current_time())
    lda_topics_dictionary, lda_document_best_topic = LDA_topic_modeling(documents, no_features=number_of_features,
                                                                        no_topics=number_of_topics,
                                                                        no_top_words=number_of_top_words)
    print(display_current_time())

    # _______________ write results _______________
    dataset["topic_index"] = nmf_document_best_topic
    dataset.to_csv('../../data/politics_with_topics.csv', index=False)

    with open('../../data/nmf_topics_dictionary.json', 'w', encoding='utf-8') as fp:
        json.dump(nmf_topics_dictionary, fp, ensure_ascii=False)
