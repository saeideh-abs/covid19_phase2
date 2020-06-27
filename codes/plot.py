import matplotlib.pyplot as plt
import pandas as pd


# create pie chart
def pie_chart(labels, sizes):
    fig = plt.gcf()
    fig.set_size_inches(1, 1)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.legend(labels, loc="best", prop={'size': 6})
    ax1.axis('equal')
    plt.show()


if __name__ == '__main__':
    emotion = pd.read_csv("../data/emotions.csv")
    polarity = pd.read_csv("../data/polarity.csv")

    emotional_labels = ["شادی", "غم", "ترس", "تنفر", "خشم", "شگفتی", "استرس", "اعتماد", "پیش‌بینی", "سایر هیجانات"]
    polarity_labels = ["مثبت", "منفی", "خنثی"]

    emotional_chart_labels = ["happy", "sad", "fear", "hate", "anger", "surprise", "stress", "trust", "forecast",
                              "other excitements"]
    polarity_chart_labels = ["positive", "negative", "neutral"]

    emotion_sizes = []
    polarity_sizes = []

    for label in emotional_labels:
        emotion_sizes.append(len(emotion[emotion[label] == 1]) / len(emotion))

    # pie chart for emotion data
    pie_chart(emotional_chart_labels, emotion_sizes)

    for label in polarity_labels:
        polarity_sizes.append(len(polarity[polarity[label] == 1]) / len(polarity))

    # pie chart for polarity data
    pie_chart(polarity_chart_labels, polarity_sizes)
