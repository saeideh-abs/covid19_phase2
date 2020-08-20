import matplotlib.pyplot as plt
import pandas as pd


# create pie chart
def pie_chart(labels, sizes, colors):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    fig1, ax1 = plt.subplots()
    if colors:
        ax1.pie(sizes, colors=colors, shadow=True, startangle=90)
    else:
        ax1.pie(sizes, shadow=True, startangle=90)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, sizes)]
    ax1.legend(labels, loc="best", prop={'size': 6})
    ax1.axis('equal')
    plt.show()


if __name__ == '__main__':
    emotion = pd.read_csv("../data/statistics/emotions.csv")
    polarity = pd.read_csv("../data/statistics/polarity_no_multi_label_plus_eini_label.csv")

    emotional_labels = ["شادی", "غم", "ترس", "تنفر", "خشم", "شگفتی", "استرس", "اعتماد", "پیش‌بینی", "سایر هیجانات"]
    polarity_labels = ["مثبت", "منفی", "خنثی", "پست عینی"]

    emotional_chart_labels = ["happy", "sad", "fear", "hate", "anger", "surprise", "stress", "trust", "forecast",
                              "other excitements"]
    polarity_chart_labels = ["positive", "negative", "neutral", "fact"]

    emotion_sizes = []
    polarity_sizes = []
    colors = []

    for label in emotional_labels:
        emotion_sizes.append(len(emotion[emotion[label] == 1]) / len(emotion))

    # pie chart for emotion data
    pie_chart(emotional_chart_labels, emotion_sizes, colors)

    for label in polarity_labels:
        polarity_sizes.append(len(polarity[polarity[label] == 1]) / len(polarity))

    # pie chart for polarity data
    pie_chart(polarity_chart_labels, polarity_sizes, colors)

    # social distance data based on post sources
    social_distance = pd.read_csv("../data/social_distance/social_distance.csv")
    net_type_labels = ["instagram", "telegram", "twitter", "news"]
    net_type_sizes = []
    colors = ["green", "red", "gray", "orange"]

    for label in net_type_labels:
        net_type_sizes.append(len(social_distance[social_distance["net_type"] == label]) / len(social_distance))

    # pie chart for post sources data
    pie_chart(net_type_labels, net_type_sizes, colors)
