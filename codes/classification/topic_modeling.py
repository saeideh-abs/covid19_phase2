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
import re
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def display_current_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time


def remove_null_values(data_set):
    nan_indices = np.where(data_set['textField_nlp_normal'].isnull())[0]  # get index of null values
    data_set = data_set.drop(nan_indices, axis=0)  # remove null values
    return data_set


def text_cleaner(docs):
    stop_words = open('../../resources/stopwords_list.txt', encoding="utf8").read().split('\n')
    normalizer = Normalizer(persian_numbers=False)
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    final_text = []

    for doc in docs:
        normal_text = normalizer.normalize(doc)
        doc_words = word_tokenize(normal_text)
        # stem = [stemmer.stem(word) for word in without_stop_words]
        lemm = [lemmatizer.lemmatize(word).split('#')[0] for word in
                doc_words]  # get the past part of the lemm
        without_stop_words = [word for word in lemm if word not in stop_words]
        final_text.append(' '.join(without_stop_words))
    return final_text


def hazm_sentences_tokenize(docs, joined=True, numpy_array=True):
    normalizer = Normalizer(persian_numbers=False)
    normalize_content = []
    # stop_words = stopwords_list()  #hazm stopwords
    stop_words = open('../../resources/stopwords_list.txt', encoding="utf8").read().split('\n')

    for elem in tqdm(docs):
        normalize_sentence = normalizer.normalize(elem)
        normalize_sentence = re.sub(r'\d+', '', normalize_sentence)
        sentence_words = word_tokenize(normalize_sentence)
        without_stop_words = [elem for elem in sentence_words if elem not in stop_words]
        # for word in without_stop_words:
        #     if word == 'های' or word == 'اند' or word == 'ها' or word == 'می':
        #         print(word, "in stop words")
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
    for word in tfidf_feature_names:
        if word == 'های' or word == 'اند' or word == 'ها' or word == 'می' or word == 'ای' or word == 'نمی' \
                or word == '.' or word == '!' or word == ',' or word == ':':
            print(word, "in NMF")

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_vecs)
    topics_dict = return_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(topics_dict)

    # find each post is related to which topic?
    doc_topic = nmf.transform(tfidf_vecs)
    # find index of the best topic (topic with highest score) for each document
    doc_best_topic = np.argmax(doc_topic, axis=1)
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
    return topics_dict, doc_best_topic


def build_tfidf(docs, no_features):
    # use hazm word_tokenize func for tokenization
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenize, max_df=0.7, min_df=10,
                                       max_features=no_features, ngram_range=(1, 1), stop_words=punctuation)
    tfidf = tfidf_vectorizer.fit_transform(docs)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return tfidf, tfidf_feature_names


def build_tf(docs, no_features):
    # use hazm word_tokenize func for tokenization
    tf_vectorizer = CountVectorizer(tokenizer=word_tokenize, max_df=0.7, min_df=10,
                                    max_features=no_features, ngram_range=(1, 1), stop_words=punctuation)
    tf = tf_vectorizer.fit_transform(docs)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names


def return_topics(model, feature_names, no_top_words):
    keys = []  # topics name
    values = []  # words related to each topic
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


def write_results(data__df, doc_best_topic, hashtags_filename, topics_dic, topics_filename):
    data_df = data__df.assign(topic_index=doc_best_topic)
    data_df.to_csv('../../data/searched_hashtags/' + hashtags_filename + '.csv', index=False)

    with open('../../data/topics/' + topics_filename + '.json', 'w', encoding='utf-8') as fp:
        json.dump(topics_dic, fp, ensure_ascii=False)


def topic_modeling(data_set, num_features, num_topics, num_top_words, hashtags_filename, topics_filename):
    documents = data_set['textField_nlp_normal']
    print("preprocessing using hazm", display_current_time())
    documents = hazm_sentences_tokenize(documents, numpy_array=False)
    # documents = text_cleaner(documents)

    # _______________ topic modeling part _______________
    print(" _____________NMF topic modeling______________", display_current_time())
    nmf_topics_dictionary, nmf_document_best_topic = NMF_topic_modeling(documents,
                                                                        no_features=num_features,
                                                                        no_topics=num_topics,
                                                                        no_top_words=num_top_words)

    print("_________________LDA topic modeling______________", display_current_time())
    lda_topics_dictionary, lda_document_best_topic = LDA_topic_modeling(documents,
                                                                        no_features=num_features,
                                                                        no_topics=num_topics,
                                                                        no_top_words=num_top_words)
    print(display_current_time())

    # _______________ write results _______________
    write_results(data_set, doc_best_topic=nmf_document_best_topic,
                  hashtags_filename='nmf_' + hashtags_filename,
                  topics_dic=nmf_topics_dictionary,
                  topics_filename='nmf_' + topics_filename)

    write_results(data_set, doc_best_topic=lda_document_best_topic,
                  hashtags_filename='lda_' + hashtags_filename,
                  topics_dic=lda_topics_dictionary,
                  topics_filename='lda_' + topics_filename)


# find topics of all data(news and social nets) together
def all_docs_main(data_set):
    topic_modeling(data_set, num_features=700, num_topics=10, num_top_words=10,
                   hashtags_filename='all_' + type + '_with_topics',
                   topics_filename='all_' + type + '_topics_dic')


# find topics of just news data
def news_main(data_set):
    print("################ news topic modeling #################")
    topic_modeling(data_set, num_features=500, num_topics=10, num_top_words=10,
                   hashtags_filename='news_' + type + '_with_topics',
                   topics_filename='news_' + type + '_topics_dic')


# find topics of social nets (telegram, twitter and instagram) data
def socialnets_main(data_set):
    print("################## social networks topic modeing ##################")
    topic_modeling(data_set, num_features=400, num_topics=10, num_top_words=10,
                   hashtags_filename='socialnets_' + type + '_with_topics',
                   topics_filename='socialnets_' + type + '_topics_dic')


if __name__ == '__main__':
    punctuation = ['.', ',', ':', ';', '!', '...', '?', '؟', '(', ')', ')', '(', '!!!', '!!', '…', '،', '\\', '/',
                   '<', '>', '»', '«', '*', '^', '&', '%', '$', '=', '+', '//t', '🆔', '#','✅','👈','🌐','http','//www']
    global type
    type = 'politics_100%'
    # _______________ loading the data and pre processing _______________
    dataset = pd.read_csv('../../data/' + type + '.csv', na_values='')

    print("search and remove null values", display_current_time())
    dataset = remove_null_values(dataset)

    mask = dataset['net_type'] == 'news'
    news_dataset = dataset[mask]
    socialnets_dataset = dataset[~mask]

    # all_docs_main(dataset)
    socialnets_main(socialnets_dataset)
    news_main(news_dataset)
