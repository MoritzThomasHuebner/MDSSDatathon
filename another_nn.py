#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

 # pip install livelossplot

import numpy as np
# from livelossplot import PlotLossesKeras

import tensorflow as tf
import pandas as pd

import tensorflow_hub as hub
from matplotlib import pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


# preprocess (clean) data here
def load_dataset(path):
    df = pd.read_csv(path)
    df['label'] += 1
    return df[['text', 'label']]


# split the "train" into a "train" and "test" here
def load_datasets():
    df = load_dataset('train_data.csv')

    train_df = df.sample(frac=0.8, random_state=200)  # random state is a seed value
    test_df = df.drop(train_df.index)

    return train_df, test_df


def dataframe_to_tensor(df):
    return tf.data.Dataset.from_tensor_slices((train_df['text'], train_df['label']))


def print_my_tensor(tensor):
    for slice in tensor:
        print(slice)


def get_compiled_model():
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    embedding_nnlm = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    hub_layer = hub.KerasLayer(embedding_nnlm, input_shape=[], dtype=tf.string,
                               trainable=False)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model






def train_model(model, train_data, validation_data, shuffle_amount=100, batch_size=16):
    history = model.fit(
        train_data.batch(batch_size),
        epochs=20,
        validation_data=validation_data.batch(batch_size),
        verbose=1,
    )
    plot_progress(history)

def plot_progress(history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


train_df, test_df = load_datasets()
train_dataset = dataframe_to_tensor(train_df)
test_dataset = dataframe_to_tensor(test_df)
model = get_compiled_model()
train_model(model, train_dataset, test_dataset)

