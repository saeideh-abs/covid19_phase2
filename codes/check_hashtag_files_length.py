import pandas as pd

file_types = ['nmf_news_politics_100%', 'nmf_news_economy_100%', 'nmf_news_gharantineh_100%', 'nmf_news_health_100%',
              'nmf_socialnets_politics_100%', 'nmf_socialnets_economy_100%', 'nmf_socialnets_gharantineh_100%',
              'nmf_socialnets_health_100%', ]
# _______________ loading the data and pre processing _______________


for file in file_types:
    dataset = pd.read_csv('../data/searched_hashtags/' + file + '_with_topics.csv')
    print(file, dataset.shape[0])


