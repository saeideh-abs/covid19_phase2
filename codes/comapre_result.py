import pandas as pd
from sklearn.metrics import accuracy_score


def find_predictions_in_competition_data(true_labels_files_path, predictions):
    true_labels = []
    match_pred = []
    sentimental_labels = ["مثبت", "منفی", "خنثی", "پست عینی"]
    file_types = ["instagram", "telegram", "twitter"]
    for file_typ in file_types:
        for i in range(1, 14):
            file_name = true_labels_files_path + "/" + file_typ + "_corona98" + str(i) + "_20.csv"
            print(file_name)
            data = pd.read_csv(file_name)
            common_rows = data.loc[data['postId'].isin(predictions["postId"])]
            data_prob_columns = ["مثبت_pred", "منفی_pred", "خنثی_pred", "پست عینی_pred"]
            for i, d in common_rows.iterrows():
                true_labels.append([d["postId"], sentimental_labels[
                    d[data_prob_columns].values.tolist().index(max(d[data_prob_columns].values.tolist()))]])
                match_pred.append(predictions[predictions["postId"] == d["postId"]])

    print("number of true labels:", len(true_labels))
    print("number of match predictions:", len(match_pred))
    return true_labels, match_pred


if __name__ == '__main__':
    # data files path
    predictions = pd.read_csv("../data/final_labeled_social_distance_data.csv")
    true_labels_files_path = "../data/BERT"

    true_labels, match_pred = find_predictions_in_competition_data(true_labels_files_path, predictions)
    true_labels = pd.DataFrame(true_labels, columns=["postId", "label"])
    true_labels.to_csv("../data/true_labels_social_distance_data.csv")
    match_pred = pd.DataFrame(true_labels, columns=["postId", "label"])
    match_pred.to_csv("../data/match_pred_social_distance_data.csv")
    print("accuracy score:", accuracy_score(true_labels["label"], match_pred["label"]))
