import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


def get_training_and_testing_data(csv_file):
    data = pd.read_csv(csv_file)
    # data = data.iloc[:100, :]
    lemmatizer = WordNetLemmatizer()

    data['text'] = [data['text'][i].translate(str.maketrans('', '', string.punctuation)) for i in range(len(data))]
    data['text'] = [data['text'][i].lower() for i in range(len(data))]
    data['text'] = [data['text'][i].strip() for i in range(len(data))]
    data['text'] = [re.sub(r'\d+', '', data['text'][i]) for i in range(len(data))]
    data['text'] = [word_tokenize(data['text'][i]) for i in range(len(data))]
    data['text'] = [[word for word in data['text'][i] if not word in stop_words] for i in range(len(data))]
    data['text'] = [[lemmatizer.lemmatize(word) for word in data['text'][i]] for i in range(len(data))]
    data = data.sample(len(data))
    train = data.iloc[:5500, :]
    test = data.iloc[5500:, :]
    return train, test


train_data, test_data = get_training_and_testing_data('train_data.csv')
print(train_data)
