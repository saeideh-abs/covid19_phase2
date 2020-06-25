import matplotlib.pyplot as plt
import pandas as pd


# create pie chart
def pie_chart(labels, sizes):
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.legend(labels, loc="best")
    ax1.axis('equal')
    plt.show()


if __name__ == '__main__':
    emotion = pd.read_csv("../data/emotions.csv")
    polarity = pd.read_csv("../data/polarity.csv")

    emotion_labels = ["شادی", "غم", "ترس", "تنفر", "خشم", "شگفتی", "استرس"]
    polarity_labels = ["مثبت", "منفی", "خنثی"]

    emotion_chart_labels = ["happy", "sad", "fear", "hate", "anger", "surprise", "stress"]
    polarity_chart_labels = ["positive", "negative", "neutral"]

    emotion_sizes = []
    polarity_sizes = []

    for label in emotion_labels:
        emotion_sizes.append(len(emotion[emotion[label] == 1]) / len(emotion))

    # pie chart for emotion data
    pie_chart(emotion_chart_labels, emotion_sizes)

    for label in polarity_labels:
        polarity_sizes.append(len(polarity[polarity[label] == 1]) / len(polarity))

    # pie chart for polarity data
    pie_chart(polarity_chart_labels, polarity_sizes)
