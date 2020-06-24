import pandas as pd


if __name__ == '__main__':
    instagram = pd.read_csv('../data/manual_tag/all_fields_instagram.csv')
    twitter = pd.read_csv('../data/manual_tag/all_fields_twitter.csv')
    news = pd.read_csv('../data/manual_tag/all_fields_news.csv')
    telegram = pd.read_csv('../data/manual_tag/all_fields_telegram.csv')
    all_fields_df = pd.concat([instagram, twitter, news, telegram], axis=0, ignore_index=True)
    print(len(instagram))
    print(len(twitter))
    print(len(news))
    print(len(telegram))

    all_fields_df.to_csv("../data/manual_tag/all_fields.csv", index=False)
