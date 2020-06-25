import gensim
import pandas as pd

if __name__ == '__main__':
    tagged_data = pd.read_csv('../../data/manual_tag/clean_labeled_data.csv')
    print(tagged_data)