from datetime import datetime

import pandas as pd


# retrieved post based on hashtag file
def ret_posts_based_on_hashtags(data_folder_path, hashtags_file_path, labeled_data_path):
    current_time = datetime.now()
    retrieved_data_ids = []
    file_types = ["instagram", "telegram", "twitter"]

    labeled_data = pd.read_csv(labeled_data_path)

    with open(hashtags_file_path, encoding="utf-8") as h_file:
        hashtags = h_file.readlines()

    for hashtag in hashtags:
        # read social network data
        for file_typ in file_types:
            for i in range(1, 10):
                file_name = data_folder_path + "/" + file_typ + "/" + file_typ + "_corona98" + str(
                    i) + "_normalized_tokenized_20.csv"
                data = pd.read_csv(file_name, header=1)
                # delete empty and unrelated data
                data = data[(data.post_type != "U") | (data.post_type != "E")]
                # find unique data
                unique_data = data[data['cluster_idx'].isnull()]
                # get one sample from each duplicate data
                duplicate_data = data.drop_duplicates(subset='cluster_idx', keep="first")
                # concatenate unique data and one samples of each duplicate data
                data = pd.concat([unique_data, duplicate_data])
                for index, row in data.iterrows():
                    if hashtag.rstrip() in row["tokens"].split("|"):
                        if str(row["postId"]) not in labeled_data["Post Id"].values.tolist():
                            retrieved_data_ids.append([row["postId"], file_typ])

        # read news data
        file_typ = "news"
        for i in range(1, 10):
            file_name = data_folder_path + "/" + file_typ + "/" + file_typ + "_corona98" + str(
                i) + "_normalized_tokenized_20.csv"
            data = pd.read_csv(file_name, header=1)
            # delete empty and unrelated data
            data = data[(data.post_type != "U") | (data.post_type != "E")]
            # find unique data
            unique_data = data[data['cluster_idx'].isnull()]
            # get one sample from each duplicate data
            duplicate_data = data.drop_duplicates(subset='cluster_idx', keep="first")
            # concatenate unique data and one samples of each duplicate data
            data = pd.concat([unique_data, duplicate_data])
            for index, row in data.iterrows():
                # check for empty abstract news
                if isinstance(row["newsAbstract_nlp_normal"], str):
                    if hashtag.rstrip().replace("_", " ").replace("#", "") in row["newsAbstract_nlp_normal"]:
                        if str(row["postId"]) not in labeled_data["Post Id"].values.tolist():
                            retrieved_data_ids.append([row["postId"], file_typ])
                            break

                # check for empty news
                if isinstance(row["textField_nlp_normal"], str):
                    if hashtag.rstrip().replace("_", " ").replace("#", "") in row["textField_nlp_normal"]:
                        if str(row["postId"]) not in labeled_data["Post Id"].values.tolist():
                            retrieved_data_ids.append([row["postId"], file_typ])
                            break

    print("run took time ", datetime.now() - current_time)
    print("number of posts:", len(retrieved_data_ids))
    return retrieved_data_ids


if __name__ == '__main__':
    # data files path
    hashtags_file_path = "../resources/hashtags.txt"
    data_folder_path = "../data/pre_process_1"
    labeled_data_path = "../data/manual_tag/Labeled-Data-v1.csv"

    retrieved_data_ids = ret_posts_based_on_hashtags(data_folder_path, hashtags_file_path, labeled_data_path)
    ret = pd.DataFrame(retrieved_data_ids, columns=['postId', 'net_type'])
    ret.to_csv("../data/gharantineh1.csv", index=False)
