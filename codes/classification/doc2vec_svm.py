from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def load_data(path, columns):
    data = pd.read_csv(path)
    content = data['Content']
    labels = data[columns]
    return content, labels


def create_doc2vec_embeddings(docs, save_path, vec_size, win, min_cnt, worker_threads):
    tokens = np.char.split(docs)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
    model = Doc2Vec(documents, vector_size=vec_size, window=win, min_count=min_cnt, workers=worker_threads)
    model.save(save_path)
    return model


def convert_embedding2array(model, seperate_index):
    part1_array = np.array([model.docvecs[i] for i in range(seperate_index)])
    part2_array = np.array([model.docvecs[i] for i in range(seperate_index, len(model.docvecs))])
    return part1_array, part2_array


def multi_label_to_one_label(labels):
    one_labels = np.zeros(labels.shape, dtype=int)
    for i in range(labels.shape[0]):
        label_list = np.where(labels[i, :] > 0)[0]
        if len(label_list) > 1:
            random_one_label_tag = np.random.choice(label_list)
            one_labels[i, :] = 0
            one_labels[i, random_one_label_tag] = 1
        else:
            one_labels[i, :] = labels[i, :]
    final_labels = np.zeros(labels.shape[0])
    final_labels = np.argwhere(one_labels > 0)[:, 1]
    return final_labels


def svm_classification(data, labels, test_size, C_list, kernels_list, cls_weight):
    print("you have been enterd in svm classifier")
    labels_1d_array = multi_label_to_one_label(labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels_1d_array, test_size=test_size)
    # param_grid = {'C': C_list, 'kernel': kernels_list}
    # clf = GridSearchCV(svm.SVC(class_weight=cls_weight, probability=True), param_grid)
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    # print("best estimator: ", clf.best_estimator_, clf.best_score_, clf.best_params_)
    predicted_labels = clf.predict(x_test)
    probability = clf.predict_proba(x_test)
    score = clf.score(x_test, y_test)
    print(score)
    return clf


if __name__ == '__main__':
    sentiment_labels = ['خنثی', 'منفی', 'مثبت']
    excitement_labels = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'استرس', 'سایر هیجانات']
    cores = multiprocessing.cpu_count()

    untagged_data = pd.read_csv('../../data/social_distance.csv')['textField_nlp_normal']
    polarity_content, polarity_labels = load_data('../../data/statistics/polarity.csv', sentiment_labels)
    emotions_content, emotions_labels = load_data('../../data/statistics/emotions.csv', excitement_labels)
    polarity_count = len(polarity_content)  # 3692
    emotions_count = len(emotions_content)  # 2147
    polarity_and_untagged = pd.concat([polarity_content, untagged_data], axis=0)
    emotions_and_untagged = pd.concat([emotions_content, untagged_data], axis=0)

    # create doc2vec models for all datasets
    polarity_content = np.array(polarity_content.tolist())
    emotions_content = np.array(emotions_content.tolist())

    # polarity_model = create_doc2vec_embeddings(np.array(polarity_and_untagged.tolist()), '../../data/vectors/polarity_doc2vec.bin',
    #                                            vec_size=300, win=4, min_cnt=3, worker_threads=cores)
    # emotions_model = create_doc2vec_embeddings(np.array(emotions_and_untagged.tolist()), '../../data/vectors/emotions_doc2vec.bin',
    #                                            vec_size=300, win=4, min_cnt=3, worker_threads=cores)

    polarity_model = Doc2Vec.load("../../data/vectors/polarity_doc2vec.bin")
    emotions_model = Doc2Vec.load("../../data/vectors/emotions_doc2vec.bin")

    polarity_vecs, polarity_untagged_vecs = convert_embedding2array(polarity_model, polarity_count)
    emotions_vecs, emotions_untagged_vecs = convert_embedding2array(emotions_model, emotions_count)
    # print("len of polarity_and_untagged: ", len(polarity_untagged_vecs), len(polarity_vecs), len(polarity_and_untagged))
    # print(len(polarity_model.wv.vocab.items()))
    # print(polarity_model.wv.most_similar('کرونا'))

    # classify data using SVM
    C = [30, 50, 100]
    kernel = ['rbf', 'linear']
    # classify polarity dataset
    polarity_clf = svm_classification(polarity_vecs, np.array(polarity_labels), test_size=0.1, C_list=C, kernels_list=kernel,
                                 cls_weight='balanced')
    # classify emotions dataset
    # emotions_clf = svm_classification(emotions_vecs, np.array(emotions_labels), test_size=0.1, C_list=C, kernels_list=kernel,
    #                              cls_weight='balanced')
