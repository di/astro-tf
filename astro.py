from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

COLUMNS = ["hue_rat", "col_min", "dis_min", "astro"]
LABEL_COLUMN = "label"
CONTINUOUS_COLUMNS = ["hue_rat", "col_min", "dis_min"]


def build_estimator(model_dir):
    """Build an estimator."""
    # Continuous base columns.
    hue_rat = tf.contrib.layers.real_valued_column("hue_rat")
    col_min = tf.contrib.layers.real_valued_column("col_min")
    dis_min = tf.contrib.layers.real_valued_column("dis_min")

    return tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        feature_columns=[hue_rat, col_min, dis_min],
        hidden_units=[100, 50]
    )


def input_fn(df):
    feature_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model."""
    df_train = pd.read_csv(
        tf.gfile.Open('./astro.train'),
        names=COLUMNS,
        skipinitialspace=True)
    df_test_1 = pd.read_csv(
        tf.gfile.Open('./astro.test.1'),
        names=COLUMNS,
        skipinitialspace=True)
    df_test_2 = pd.read_csv(
        tf.gfile.Open('./astro.test.2'),
        names=COLUMNS,
        skipinitialspace=True)
    df_test_3 = pd.read_csv(
        tf.gfile.Open('./astro.test.3'),
        names=COLUMNS,
        skipinitialspace=True)

    df_train[LABEL_COLUMN] = (df_train["astro"].apply(lambda x: x == 1.0)).astype(int)
    df_test_1[LABEL_COLUMN] = (df_test_1["astro"].apply(lambda x: x == 1.0)).astype(int)
    df_test_2[LABEL_COLUMN] = (df_test_2["astro"].apply(lambda x: x == 1.0)).astype(int)
    df_test_3[LABEL_COLUMN] = (df_test_3["astro"].apply(lambda x: x == 1.0)).astype(int)

    m = build_estimator('./model')
    m.fit(input_fn=lambda: input_fn(df_train), steps=200)

    print("Results of Trial #1")
    results = m.evaluate(input_fn=lambda: input_fn(df_test_1), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    print("Results of Trial #2")
    results = m.evaluate(input_fn=lambda: input_fn(df_test_2), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    print("Results of Trial #3")
    results = m.evaluate(input_fn=lambda: input_fn(df_test_3), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def main(_):
    train_and_eval()


if __name__ == "__main__":
    tf.app.run()
