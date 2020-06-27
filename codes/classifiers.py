import pickle

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def classifier(data, model, labels, classifier_typ):
    doc_vectors = []
    doc_labels = []

    # get data vectors
    for index, row in data.iterrows():
        # doc_words = word_tokenize(row['Content'])
        # doc_vectors.append(model.infer_vector(doc_words=doc_words))
        doc_vectors.append(model["labeled" + str(index)])
        doc_labels.append(row[labels].values.tolist())

    # create classifier based on its type
    if classifier_typ == 'rand_forest':
        clf = RandomForestClassifier()
    elif classifier_typ == 'SVM':
        # clf = svm.SVC(kernel='linear', C=1)
        print("******")
    else:
        print("!!!!!")

    # cross validation
    scores = cross_val_score(clf, doc_vectors, doc_labels, cv=10)

    return scores.mean()


if __name__ == '__main__':
    with open("../data/vectors/emotions_doc2vec.model", "rb") as f:
        emotions_doc2vec = pickle.load(f)
    with open("../data/vectors/polarity_doc2vec.model", "rb") as f:
        polarity_doc2vec = pickle.load(f)

    emotions_data = pd.read_csv("../data/emotions.csv")
    polarity_data = pd.read_csv("../data/polarity.csv")

    emotional_labels = ["شادی", "غم", "ترس", "تنفر", "خشم", "شگفتی", "استرس"]
    polarity_labels = ["مثبت", "منفی", "خنثی"]

    classifier_types = ['rand_forest', 'SVM', 'max_ent']

    print("test accuracy for emotional data:",
          classifier(emotions_data, emotions_doc2vec, emotional_labels, classifier_types[0]))
    print("test accuracy for polarity data:",
          classifier(polarity_data, polarity_doc2vec, polarity_labels, classifier_types[0]))
