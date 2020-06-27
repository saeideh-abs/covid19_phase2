from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm


def load_data(path, columns):
    data = pd.read_csv(path)
    content = np.array(data['Content'].tolist())
    labels = np.array(data[columns])
    return content, labels


def create_doc2vec_embeddings(docs, save_path, vec_size, win, min_cnt, worker_threads):
    tokens = np.char.split(docs)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
    model = Doc2Vec(documents, vector_size=vec_size, window=win, min_count=min_cnt, workers=worker_threads)
    model.save(save_path)
    return model


def svm_classification(data, labels, test_size, C_list, kernels_list, cls_weight):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    param_grid = {'C': C_list, 'kernel': kernels_list}
    clf = GridSearchCV(svm.SVC(class_weight=cls_weight, probability=True), param_grid)
    clf.fit(x_train, y_train)
    # predicted_labels = clf.predict(x_test)
    # probability = clf.predict_proba(x_test)
    # score = clf.score(x_test, y_test)
    # print(score)
    # return clf


def convert_embedding2array(model):
    data_array = np.array([model.docvecs[i] for i in range(len(model.docvecs))])
    return data_array


if __name__ == '__main__':
    sentiment_labels = ['خنثی', 'منفی', 'مثبت']
    excitement_labels = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'استرس']
    cores = multiprocessing.cpu_count()

    polarity_content, polarity_labels = load_data('../../data/statistics/polarity.csv', sentiment_labels)
    emotions_content, emotions_labels = load_data('../../data/statistics/emotions.csv', excitement_labels)
    polarity_count = len(polarity_content)  # 3692
    emotions_count = len(emotions_content)  # 2147

    # create doc2vec models for all datasets
    polarity_model = create_doc2vec_embeddings(polarity_content, '../../data/vectors/polarity_doc2vec.bin',
                                               vec_size=300, win=4, min_cnt=3, worker_threads=cores)
    emotions_model = create_doc2vec_embeddings(emotions_content, '../../data/vectors/emotions_doc2vec.bin',
                                               vec_size=300, win=4, min_cnt=3, worker_threads=cores)
    polarity_vecs = convert_embedding2array(polarity_model)
    # print(len(emotions_model.docvecs))
    # print(len(polarity_model.wv.vocab.items()))
    # print(polarity_model.wv.most_similar('عالی'))

    # classify data using SVM
    C = [0.01, 0.1, 0.5, 1, 10, 50, 100]
    kernel = ['rbf', 'linear']
    # svm_clf = svm_classification(polarity_vecs, polarity_labels, test_size=0.1, C_list=C, kernels_list=kernel, cls_weight='balanced')
