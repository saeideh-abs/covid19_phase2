import pandas as pd


def extract_one_label_data(data, labels):
    data['number_of_ones'] = [row.tolist().count(1) for row in data[labels].values]
    data = data[data['number_of_ones'] == 1]
    data = data.drop('number_of_ones', 1)
    return data


def split_data(data):
    data = data.sample(frac=1).reset_index(drop=True)
    train = pd.DataFrame(data.values[:int(len(data) * 0.9)], columns=data.columns)
    test = pd.DataFrame(data.values[int(len(data) * 0.9):], columns=data.columns)
    valid = pd.DataFrame(train.values[int(len(train) * 0.9):], columns=data.columns)
    train = train.loc[~train.index.isin(valid.index)]
    return train, test, valid


if __name__ == '__main__':
    # read polarity data
    sentiment_labels = ['خنثی', 'منفی', 'مثبت']
    polarity = pd.read_csv("../data/statistics/polarity.csv")
    print("original polarity:", len(polarity))

    # extract polarity data that have one sentimental label
    polarity_no_multi_label = extract_one_label_data(polarity, sentiment_labels)
    polarity_no_multi_label.to_csv("../data/statistics/polarity_no_multi_label.csv", index=False)
    print("polarity no multi label:", len(polarity_no_multi_label))

    # split polarity data to train, test and valid
    polarity_train, polarity_test, polarity_valid = split_data(polarity_no_multi_label)
    polarity_train.to_csv("../data/statistics/polarity_no_multi_label_train.csv", index=False)
    polarity_test.to_csv("../data/statistics/polarity_no_multi_label_test.csv", index=False)
    polarity_valid.to_csv("../data/statistics/polarity_no_multi_label_valid.csv", index=False)
    print("polarity_train length:", len(polarity_train))
    print("polarity_test length:", len(polarity_test))
    print("polarity_valid length:", len(polarity_valid))

    # read emotion data
    emotional_labels = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'سایر هیجانات', 'استرس']
    emotion = pd.read_csv("../data/statistics/emotions.csv")
    print("original emotion:", len(emotion))

    # extract emotion data that have one emotional label
    emotions_no_multi_label = extract_one_label_data(emotion, emotional_labels)
    emotions_no_multi_label.to_csv("../data/statistics/emotions_no_multi_label.csv", index=False)
    print("emotions no multi label:", len(emotions_no_multi_label))

    # split emotion data to train, test and valid
    emotion_train, emotion_test, emotion_valid = split_data(emotions_no_multi_label)
    emotion_train.to_csv("../data/statistics/emotions_no_multi_label_train.csv", index=False)
    emotion_test.to_csv("../data/statistics/emotions_no_multi_label_test.csv", index=False)
    emotion_valid.to_csv("../data/statistics/emotions_no_multi_label_valid.csv", index=False)
    print("emotion_train length:", len(emotion_train))
    print("emotion_test length:", len(emotion_test))
    print("emotion_valid length:", len(emotion_valid))

    # read polarity data (include post eini label)
    print("polarity data plus post eini label:")
    sentiment_labels = ['خنثی', 'منفی', 'مثبت', 'پست عینی']
    polarity = pd.read_csv("../data/manual_tag/clean_labeled_data.csv")
    print("original polarity:", len(polarity))

    # extract polarity data that have one sentimental label
    polarity_no_multi_label = extract_one_label_data(polarity, sentiment_labels)
    polarity_no_multi_label.to_csv("../data/statistics/polarity_no_multi_label_plus_eini_label.csv", index=False)
    print("polarity no multi label:", len(polarity_no_multi_label))

    # split polarity data to train, test and valid
    polarity_train, polarity_test, polarity_valid = split_data(polarity_no_multi_label)
    polarity_train.to_csv("../data/statistics/polarity_no_multi_label_plus_eini_label_train.csv", index=False)
    polarity_test.to_csv("../data/statistics/polarity_no_multi_label_plus_eini_label_test.csv", index=False)
    polarity_valid.to_csv("../data/statistics/polarity_no_multi_label_plus_eini_label_valid.csv", index=False)
    print("polarity_train length:", len(polarity_train))
    print("polarity_test length:", len(polarity_test))
    print("polarity_valid length:", len(polarity_valid))
