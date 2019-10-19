# # Install TF-Hub.
# !pip install -q tensorflow-hub
# !pip install -q seaborn

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from cleaning import get_training_and_testing_data


# preprocess (clean) data here
def load_dataset(path):
    df = pd.read_csv(path)
    df['label'] += 1  # labels should be >= 0
    return df[['text', 'label']]


# split the "train" into a "train" and "test" here
def load_datasets():
    df = load_dataset('train_data.csv')
    train_df = df.sample(frac=0.8, random_state=200)  # random state is a seed value
    test_df = df.drop(train_df.index)
    return train_df, test_df


def load_and_train_data():
    # # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)
    train_df, test_df = get_training_and_testing_data('train_data.csv')

    # Format the data
    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['label'], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['label'], shuffle=False)

    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df['label'], shuffle=False)

    # Setup the 'feature'
    embedded_text_feature_column = hub.text_embedding_column(
        key='text',
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    # Setup the DNN classifier
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=3,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # Training for 1,000 steps means 128,000 training eg with the default batch size.
    # number epochs = 128,000/len(train)
    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

load_and_train_data()
