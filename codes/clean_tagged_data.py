import pandas as pd
import numpy as np
import csv


def search_colmn_by_value(data, column_name, value):
    condition = data[column_name] == value
    indices = data[condition].index.values
    return indices


if __name__ == '__main__':
    # load the data
    tagged_data = pd.read_csv('../data/manual_tag/Labeled-Data-v1.csv')

    # ______________ remove useless posts _______________
    empty_data_indices = search_colmn_by_value(tagged_data, 'پست خالی', 1)
    clean_data = tagged_data.drop(empty_data_indices)

    unrelated_data_indices = search_colmn_by_value(clean_data, 'پست بی‌ربط', 1)
    clean_data = clean_data.drop(unrelated_data_indices)

    valueless_data_indices = search_colmn_by_value(clean_data, 'پست ناقص یا  بی‌ارزش', 1)
    clean_data = clean_data.drop(valueless_data_indices)

    print("total number of posts after removing useless posts: ", len(clean_data))
    print(clean_data.loc[48,:])