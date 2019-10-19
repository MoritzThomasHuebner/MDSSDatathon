#!/usr/bin/env python3

'''

0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

'''

import pandas as pd

c = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv', sep='\t')


def label_race(row):
    if row['Sentiment'] == 0:
        return 0
    if row['Sentiment'] == 4:
        return 2
    else:
        return 0

c.apply(lambda row: label_race(row), axis=1)

c['label'] = c['Sentiment']
c['text'] = c['Phrase']

c = c[['text', 'label']]
c.to_csv('new_data.csv')
c.head()

