# %%
import pandas as pd
import numpy as np
import os
import sys


class Extraction():

    def __init__(self, input_file):
        self.input_file = input_file
        self.emotional_tags = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات',
                               'استرس']

    def extract_emotional_posts(self):
        dataframe = pd.read_csv(self.input_file)
        numpy_emotional_array = dataframe.loc[:, self.emotional_tags].to_numpy()
        sum_array = np.sum(numpy_emotional_array, axis=1)
        row_arg_list = np.where(sum_array > 0)[0].tolist()
        return row_arg_list, dataframe

    def write_emotional_posts_to_csv(self, row_args_list, dataframe):
        target_df = dataframe.loc[row_args_list, :]
        folder = '{}/data/emotional_posts'.format(path)
        os.makedirs(name=folder, exist_ok=True)
        target_df.to_csv('{}/emotional_posts.csv'.format(folder))
        return target_df


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(path, os.pardir))

ex_instance = Extraction('{}/data/manual_tag/Labeled-Data-v1.csv'.format(path))
row_args_list, dataframe = ex_instance.extract_emotional_posts()
final_df = ex_instance.write_emotional_posts_to_csv(row_args_list, dataframe)
