import pandas as pd
import numpy as np
import gc


# this function returns the path of all datasets in ./data/preprocess_1 folder
def get_preprocessed_data_paths():
    files_path = []
    # because of the lake of memory I used this way: in each run changed the social type to news, telegram , ...
    social_type = ['twitter']

    for type in social_type:
        for i in range(1, 10):
            files_path.append("../data/pre_process_1/" + type + "/" + type + "_corona98" + str(i) + "_normalized_tokenized_20.csv")
    return files_path, social_type[0]


def search_for_fields(path, tagged_data, all_fields_df):
    post_ids = tagged_data['Post Id']
    df = pd.read_csv(path)
    columns_name = df.iloc[0, :].tolist()
    df.columns = columns_name

    # search for post ids
    condition = df['postId'].isin(post_ids)  # returns list of true/false
    common_post_ids = df.loc[condition, 'postId']
    # print(len(common_post_ids))
    if len(common_post_ids) == 0:
        return all_fields_df

    for id in common_post_ids:
        tagged_row = tagged_data.loc[tagged_data['Post Id'] == id]
        df_row = df.loc[df['postId'] == id]

        tagged_row.reset_index(drop=True, inplace=True)
        df_row.reset_index(drop=True, inplace=True)

        new_row = pd.concat([tagged_row, df_row['textField_nlp_process']], axis=1, ignore_index=True)
        all_fields_df = pd.concat([all_fields_df, new_row], axis=0)
    del df
    gc.collect()
    return all_fields_df


if __name__ == '__main__':
    # load the data
    clean_tagged_data = pd.read_csv('../data/manual_tag/clean_labeled_data.csv')
    paths, social_topic = get_preprocessed_data_paths()

    # create new data frame to save new data in it
    all_fields_columns = clean_tagged_data.columns.tolist()
    all_fields_columns.append('textField_nlp_process')
    all_fields_df = pd.DataFrame()

    for path in paths:
        all_fields_df = search_for_fields(path, clean_tagged_data, all_fields_df)

    all_fields_df.columns = all_fields_columns
    all_fields_df.to_csv("../data/manual_tag/all_fields_" + social_topic + ".csv", index=False)
