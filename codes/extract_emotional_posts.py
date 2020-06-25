# %%
import pandas as pd
import numpy as np
import os
import sys


class Extraction():

    def __init__(self, input_file):
        self.input_file = input_file
        self.polarity = ['مثبت', 'منفی', 'خنثی']
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    def extract_emotional_posts(self, tags):
        dataframe = pd.read_csv(self.input_file)
        numpy_emotional_array = dataframe.loc[:, tags].to_numpy()
        sum_array = np.sum(numpy_emotional_array, axis=1)
        row_arg_list = np.where(sum_array > 0)[0].tolist()
        return row_arg_list, dataframe

    def write_emotional_posts_to_csv(self, row_args_list, dataframe, folder_name, file_name):
        target_df = dataframe.loc[row_args_list, :]
        folder = '{}/data/{}'.format(path, folder_name)
        os.makedirs(name=folder, exist_ok=True)
        target_df.to_csv('{}/{}.csv'.format(folder, file_name))
        return target_df


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(path, os.pardir))

ex_instance = Extraction('{}/data/manual_tag/clean_labeled_data.csv'.format(path))
row_args_list, dataframe = ex_instance.extract_emotional_posts(ex_instance.emotional_tags)
final_df_emotions = ex_instance.write_emotional_posts_to_csv(row_args_list, dataframe, 'statistics', 'emotions')

row_args_list, dataframe = ex_instance.extract_emotional_posts(ex_instance.polarity)
final_df_polarity = ex_instance.write_emotional_posts_to_csv(row_args_list, dataframe, 'statistics', 'polarity')
