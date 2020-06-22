import pandas as pd
import numpy as np
import csv


def post_id_stats(post_id_series):
    print("__________ post ids stats ___________")
    print("total number of Post IDs:", post_id_series.size)
    print("total number of unique Post IDs:", post_id_series.unique().size)


def value_counts(series, data_title):
    print("____________ ", data_title, " ___________")
    print("total number of ", data_title, ":", series.size)
    print(data_title, " value counts:\n", series.value_counts())


# def unrelated_posts_stats(unrelated_series):
#     print("____________ unrelated posts stats ___________")
#     print("total number of unrelated posts:", unrelated_series.size)
#     print("unrelated posts value counts:\n", unrelated_series.value_counts())
#
#
# def valueless_posts_stats(valueless_series):
#     print("____________ valueless posts stats ___________")
#     print("total number of valueless posts:", valueless_series.size)
#     print("valueless posts value counts:\n", valueless_series.value_counts())


if __name__ == '__main__':
    tagged_data = pd.read_csv('../data/manual_tag/Labeled-Data-v1.csv')

    post_ids = tagged_data.loc[:, 'Post Id']
    empty_posts = tagged_data.loc[:, 'پست خالی']
    unrelated_posts = tagged_data.loc[:, 'پست بی‌ربط']
    valueless_posts = tagged_data.loc[:, 'پست ناقص یا  بی‌ارزش']

    post_id_stats(post_ids)
    value_counts(empty_posts, "empty posts")
    value_counts(unrelated_posts, "unrelated posts")
    value_counts(valueless_posts, "valueless posts")