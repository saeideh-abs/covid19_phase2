from __future__ import unicode_literals
import pandas as pd
import numpy as np
from hazm import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def hazm_sentences_tokenize(docs, joined=True, numpy_array=True):
    normalizer = Normalizer(persian_numbers=False)
    normalize_content = []
    hazm_stopwords = stopwords_list()
    for elem in docs:
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


def NMF_topic_modeling(docs, no_features, no_topics, no_top_words):
    print("NMF topic modeling")
    # NMF is able to use tf-idf
    tfidf_vecs, tfidf_feature_names = build_tfidf(docs, no_features=no_features)
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_vecs)
    display_topics(nmf, tfidf_feature_names, no_top_words)


def LDA_topic_modeling(docs, no_features, no_topics, no_top_words):
    print("LDA topic modeling")
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vecs, tf_feature_names = build_tf(docs, no_features=no_features)
    # Run LDA
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0).fit(tf_vecs)
    display_topics(lda, tf_feature_names, no_top_words)


def build_tfidf(docs, no_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features)
    tfidf = tfidf_vectorizer.fit_transform(docs)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return tfidf, tfidf_feature_names


def build_tf(docs, no_features):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features)
    tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == '__main__':
    # _______________ loading the data and preprocessing _______________
    dataset = pd.read_csv('../../data/politics.csv', na_values='')
    nan_indices = np.where(dataset['textField_nlp_normal'].isnull())[0]  # get index of null values
    documents = dataset.drop(nan_indices, axis=0)['textField_nlp_normal']  # remove null values
    print("preprocessing using hazm")
    documents = hazm_sentences_tokenize(documents, numpy_array=False)

    number_of_features = 1000
    number_of_topics = 10
    number_of_top_words = 10

    NMF_topic_modeling(documents, no_features=number_of_features,
                       no_topics=number_of_topics,
                       no_top_words=number_of_top_words)
    LDA_topic_modeling(documents, no_features=number_of_features,
                       no_topics=number_of_topics,
                       no_top_words=number_of_top_words)