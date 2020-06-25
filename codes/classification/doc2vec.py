import gensim
import pandas as pd
import numpy as np


def load_data(path, columns):
    data = pd.read_csv(path)
    content = np.array(data['Content'])
    labels = np.array(data[columns])
    return content, labels


if __name__ == '__main__':
    sentiment_labels = ['خنثی', 'منفی', 'مثبت']
    excitement_labels = ['شادی', 'غم', 'ترس', 'تنفر', 'خشم', 'شگفتی', 'اعتماد', 'پیش‌بینی', 'استرس']

    polarity_content, polarity_labels = load_data('../../data/statistics/polarity.csv', sentiment_labels)
    emotions_content, emotions_labels = load_data('../../data/statistics/emotions.csv', excitement_labels)

    polarity_count = len(polarity_content)  # 3692
    emotions_count = len(emotions_content)  # 2147

    print(type(polarity_content[0]))
    # polarity_tokens = np.char.split(polarity_content)
