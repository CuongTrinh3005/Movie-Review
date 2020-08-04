import pandas as pd
import re

def load_data_review(file):
    data = pd.read_csv(file)
    # print(data.head())
    # # Check for any null values in the dataframe.
    # print("Check whether there are any null values: ", data.isnull().values.any())
    # print("The data frame's shape is: ", data.shape)
    # # Describe the data frame of movie review
    # data.describe(include='all')
    # # Address the statistic with this dataset
    # print(data['sentiment'].value_counts())
    return data

#movie_review = load_data_review('IMDB Dataset.csv')

def remove_tags(raw_html):
    pat = re.compile(r'<.*?>')
    # pat = re.compile(r'<[^>]+>')
    return re.sub(pat, '', raw_html)

def clean_data(text):
    # Remove html tags
    sent = remove_tags(text)
    # Remove punctuation and numbers (replace any non-characters with empty character)
    sent = re.sub('[^a-zA-Z]', ' ', sent)
    # Remove one character which is not useful
    sent = re.sub('\s+[a-zA-Z]\s+', ' ', sent)
    # Remove multiple spaces
    sent = re.sub('\s+', ' ', sent)
    return sent

def creat_labeled_data(data):
    reviews = list(data['review'])
    sentiments = list(data['sentiment'])
    comments = []
    labels = [0 for _ in range(len(sentiments))]

    if len(reviews) == len(sentiments):
        for index in range(len(reviews)):
            reviews[index] = clean_data(reviews[index])
            comments.append(reviews[index])

            if sentiments[index] == 'positive':
                labels[index] = 1

    return comments, labels