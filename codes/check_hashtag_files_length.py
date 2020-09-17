import pandas as pd
import numpy as np

file_types = ['nmf_news_politics_100%', 'nmf_news_economy_100%', 'nmf_news_gharantineh_100%', 'nmf_news_health_100%',
              'nmf_socialnets_politics_100%', 'nmf_socialnets_economy_100%', 'nmf_socialnets_gharantineh_100%',
              'nmf_socialnets_health_100%', ]


file_types_topic = {'nmf_news_politics_100%': [3, 4, 5, 7], 'nmf_news_economy_100%': [0, 1, 2, 4, 5, 8],
                    'nmf_news_gharantineh_100%': [2, 4, 5, 8], 'nmf_news_health_100%': [1, 3, 5, 8, 9],
                    'nmf_socialnets_politics_100%': [], 'nmf_socialnets_economy_100%': [],
                    'nmf_socialnets_gharantineh_100%': [],
                    'nmf_socialnets_health_100%': []}


def extract_file_length():
    for file in file_types:
        dataset = pd.read_csv('../data/searched_hashtags/' + file + '_with_topics.csv')
        print(file, dataset.shape[0])


def extract_post_related_topics():

    list_data = []
    for file, topics in file_types_topic.items():
        if topics:
            dataset = pd.read_csv('../data/searched_hashtags/' + file + '_with_topics.csv')
            data = dataset.loc[:, ['topic_index']].to_numpy()
            for topic_index in topics:
                print('topic_index:{}|||||file_name:{}|||||number_of_topic:{}'.format(topic_index, file,
                                                                                 np.where(data == topic_index)[
                                                                                     0].shape[0]))
                list_data.append([topic_index, file,np.where(data == topic_index)[0].shape[0]])

    df = pd.DataFrame(list_data, columns=['topic_index', 'file_name', 'number_of_topics'])
    df.to_csv('numbers.csv', index=False)


extract_post_related_topics()
