from __future__ import unicode_literals
import pandas as pd
import numpy as np
import csv
from hazm import *
from hazm import word_tokenize


def search_column_by_value(data, column_name, value):
    condition = data[column_name] == value
    indices = data[condition].index.values
    return indices


def text_cleaner(docs):
    normalizer = Normalizer(persian_numbers=False)
    stemmer = Stemmer()
    lemmatizer = Lemmatizer()
    final_text = []

    for doc in docs:
        normal_text = normalizer.normalize(doc)
        doc_words = word_tokenize(normal_text)
        without_stop_words = [word for word in doc_words if word not in stop_words]
        # stem = [stemmer.stem(word) for word in without_stop_words]
        lemm = [lemmatizer.lemmatize(word).split('#')[0] for word in without_stop_words] # get the past part of the lemm
        final_text.append(' '.join(lemm))
    return np.array(final_text)


if __name__ == '__main__':
    # load the data
    tagged_data = pd.read_csv('../data/manual_tag/Labeled-Data-v1.csv')

    # ______________ remove useless posts _______________
    empty_data_indices = search_column_by_value(tagged_data, 'پست خالی', 1)
    clean_data = tagged_data.drop(empty_data_indices)

    unrelated_data_indices = search_column_by_value(clean_data, 'پست بی‌ربط', 1)
    clean_data = clean_data.drop(unrelated_data_indices)

    valueless_data_indices = search_column_by_value(clean_data, 'پست ناقص یا  بی‌ارزش', 1)
    clean_data = clean_data.drop(valueless_data_indices)

    print("total number of posts after removing useless posts: ", len(clean_data))

    print("start pre processing operations")
    stop_words = open('../resources/stopwords_list.txt', encoding="utf8").read().split('\n')
    clean_texts = text_cleaner(clean_data['Content'])
    clean_data.insert(5, "preprocessed_content", clean_texts)
    clean_data.to_csv('../data/manual_tag/clean_labeled_data.csv', index=False)
