# # Install TF-Hub.
# !pip install -q tensorflow-hub
# !pip install -q seaborn

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from cleaning import get_training_and_testing_data

import tf_mods


# # Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)

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
    # train_df, test_df = load_datasets()
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
        module_spec=tf_mods.NNLM,
        trainable=True)

    # Setup the DNN classifier
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[512, 256],
        feature_columns=[embedded_text_feature_column],
        n_classes=3,
        dropout=0.2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
        # ,
        #                                             initial_accumulator_value=0.1,
        #                                             l1_regularization_strength=0.0,
        #                                             l2_regularization_strength=0.0,
        #                                             use_locking=False))

    # Training for 1,000 steps means 128,000 training eg with the default batch size.
    # number epochs = 128,000/len(train)
    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    save_predicitions(estimator, "submission.csv")

    return estimator, train_df, test_df, predict_train_input_fn, predict_test_input_fn


def save_predicitions(estimator, sub_fn):
    submission_df = pd.read_csv("test_data.csv")
    submission_df = submission_df[['test_id', 'text']]
    submission_input_fn = tf.estimator.inputs.pandas_input_fn(
        submission_df, shuffle=False)
    pred = get_predictions(estimator, submission_input_fn)
    submission_df['label'] = [p - 1 for p in pred]
    submission_df = submission_df[['test_id', 'label']]
    submission_df.to_csv(sub_fn, index=False)


def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


def make_confusion_matrix_plot(estimator, train_df, predict_train_input_fn, filename):
    LABELS = [
        "negative", "neutral", "positive"
    ]

    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(train_df['label'],
                                 get_predictions(estimator, predict_train_input_fn))
        with tf.Session() as session:
            cm_out = session.run(cm)

    # Normalize the confusion matrix so that each row sums to 1.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.show()
    plt.clf()


# GET PREDICTION
estimator, train_df, test_df, predict_train_input_fn, predict_test_input_fn = load_and_train_data()

make_confusion_matrix_plot(estimator, train_df, predict_train_input_fn, 'train_confusion.png')
make_confusion_matrix_plot(estimator, test_df, predict_test_input_fn, 'test_confusion.png')


