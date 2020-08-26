from datetime import datetime
import pandas as pd
from tqdm import tqdm


def clean_data(data):
    # delete empty and unrelated data
    data = data[(data.post_type != "U") | (data.post_type != "E")]
    # find unique data
    unique_data = data[data['cluster_idx'].isnull()]
    # get one sample from each duplicate data
    duplicate_data = data.drop_duplicates(subset='cluster_idx', keep="first")
    # concatenate unique data and one samples of each duplicate data
    cleaned_data = pd.concat([unique_data, duplicate_data])

    return cleaned_data


# retrieved post based on hashtag file
def ret_posts_based_on_hashtags(data_folder_path, hashtags_file_path, labeled_data_path):
    current_time = datetime.now()
    retrieved_data_ids = []
    file_types = ["instagram", "telegram", "twitter"]

    labeled_data = pd.read_csv(labeled_data_path)

    with open(hashtags_file_path, encoding="utf-8") as h_file:
        hashtags = h_file.readlines()

    # read social network data
    for file_typ in file_types:
        for i in tqdm(range(1, 10)):
            file_name = data_folder_path + "/" + file_typ + "/" + file_typ + "_corona98" + str(
                i) + "_normalized_tokenized_20.csv"
            data = pd.read_csv(file_name, header=1)
            cleaned_data = clean_data(data)

            tokens = cleaned_data['tokens']

            for index, row in enumerate(tokens):
                if any(hashtag.rstrip() in row.split("|") for hashtag in hashtags):
                    if str(cleaned_data.iloc[index,1]) not in labeled_data["Post Id"].values.tolist():
                        retrieved_data_ids.append(
                            [cleaned_data.iloc[index,1], file_typ, cleaned_data.iloc[index, 3]])

    # read news data
    file_typ = "news"
    for i in tqdm(range(1, 10)):
        file_name = data_folder_path + "/" + file_typ + "/" + file_typ + "_corona98" + str(
            i) + "_normalized_tokenized_20.csv"
        data = pd.read_csv(file_name, header=1)
        cleaned_data = clean_data(data)
        tokens = cleaned_data['textField_nlp_normal']

        for index, row in enumerate(tokens):
            # check for empty abstract news
            if isinstance(row, str):
                if any(hashtag.rstrip().replace("_", " ").replace("#", "") in row for hashtag in hashtags):
                    if str(cleaned_data.iloc[index,1]) not in labeled_data["Post Id"].values.tolist():
                        retrieved_data_ids.append([cleaned_data.iloc[index,1], file_typ, row])


    print("run took time ", datetime.now() - current_time)
    print("number of posts:", len(retrieved_data_ids))
    return retrieved_data_ids


if __name__ == '__main__':
    # data files path
    hashtags_file_path = "../resources/politics_hashtag.txt"
    data_folder_path = "../data/pre_process_1"
    labeled_data_path = "../data/manual_tag/Labeled-Data-v1.csv"

    retrieved_data_ids = ret_posts_based_on_hashtags(data_folder_path, hashtags_file_path, labeled_data_path)
    ret = pd.DataFrame(retrieved_data_ids,
                       columns=['postId', 'net_type', 'textField_nlp_normal'])
    ret.to_csv("../data/politics.csv", index=False)
