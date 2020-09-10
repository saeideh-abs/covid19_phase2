from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from typing import Callable, List, Optional, Tuple

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import torch
import numpy as np
from sklearn.model_selection import KFold
import sys
import os


class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            bert_tokenizer,
            bert_model,
            max_length: int = 512,
            embedding_func: Optional[Callable[[torch.tensor], torch.tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

    def _tokenize(self, text: str) -> Tuple[torch.tensor, torch.tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=True,
                                                    truncation=True,
                                                    max_length=self.max_length
                                                    )["input_ids"]

        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self

    @staticmethod
    def seperate_content_lables(filename, post_id, content, label_fields):
        df = pd.read_csv(filename)
        post_id = df.loc[:, post_id]
        content = df.loc[:, content]
        labels = df.loc[:, label_fields].to_numpy()
        return post_id, content, labels

    @staticmethod
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


polarity = ['مثبت', 'منفی', 'خنثی', 'پست عینی']


sys.path.extend([os.getcwd()])
path = os.getcwd()
parent_dir = os.path.dirname(path)
root_dir = os.path.dirname(parent_dir)

tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
bert_model = BertModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
polarity_file = '{}/data/statistics/polarity_no_multi_label_plus_eini_label.csv'.format(root_dir)
bert_transformer = BertTransformer(tokenizer, bert_model)
C = [1]
polarity_ids, \
polarity_contents, \
polarity_labels = BertTransformer.seperate_content_lables(polarity_file,
                                                          'Post Id',
                                                          'Content',
                                                          polarity)
kernel = ['linear']
classifier = SVC(C=C[0], kernel=kernel[0], probability=False)

model = Pipeline(
    [
        ("vectorizer", bert_transformer),
        ("classifier", classifier),
    ]
)

final_train_labels = BertTransformer.multi_label_to_one_label(polarity_labels)
kf = KFold(n_splits=10, shuffle=False)
accuracy_score_sum = 0
for train_index, test_index in kf.split(polarity_contents):
    x_train, x_test = polarity_contents[train_index], polarity_contents[test_index]

    y_train, y_test = final_train_labels[train_index], final_train_labels[test_index]
    model.fit(x_train, y_train)
    print('fit done')
    predictions = model.predict(x_test)
    print(predictions)
    accuracy_score_sum += accuracy_score(y_test, predictions)
    print('Accuracy score: ', accuracy_score(y_test, predictions))

accuracy_score_sum /= 10
print(accuracy_score_sum)