#DEEP LEARNING APPROACH TO CLASSIFICATION OF BREAST CANCER TUMOURS
# WRITTEN BY OMKAR VIVEK SABNIS -> 16-06-2018

#IMPORTING ALL MODULES
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
import warnings
warnings.filterwarnings("ignore")


# FUNCTION TO CLEAN PREVIOUS DIRECTORIES FOR FRESH RUN - IMPROVES PERFORMANCE
def clean_run(model_dir='', source_data=''):
    if model_dir:
        if os.path.exists(model_dir):
            print("Deleting resource: Model directory [%s]." % model_dir)
            shutil.rmtree(model_dir)
            print("Removed resource: Model directory [%s]." % model_dir)
    for resource in [source_data, 'training_set.csv', 'test_set.csv']:
        if resource:
            if os.path.exists(resource):
                print("Deleting resource: Data [%s]." % resource)
                os.remove(resource)
                print("Removed resource: Data [%s]." % resource)


# FUNCTION TO DOWNLOAD THE DATASET FROM THE WEBSITE WHERE IT IS HOSTED
def download_data(data_file, url):
    download_url = url + data_file
    if not os.path.exists(data_file):
        print(
            "%s not found on local filesystem. File will be downloaded from %s."
            % (data_file, download_url))
        raw = urlopen(download_url).read()
        with open(data_file, 'wb') as f:
            f.write(raw)
            print("%s written to local filesystem." % data_file)


# FUNCTION TO CLEAN THE DATASET OF ALL IRRELEVANT DATA AND DROP COLUMNS
def process_source(local_data, col_names, missing_vals='', drop_cols=[]):
    dataframe = pd.read_csv(local_data, names=col_names)
    dataframe.replace(missing_vals, np.nan, inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.drop(drop_cols, axis=1, inplace=True)
    return dataframe


# FUNCTION TO REPLACE CLASSIFICATION COLUMNS WITH NUMERIC DATA
def replace_classification_labels(dataframe, result_col='', values_to_replace=[]):
    target_labels = [x for x in range(0, len(values_to_replace))]
    dataframe[result_col].replace(values_to_replace, target_labels, inplace=True)
    return dataframe


# FUNCTION TO SPLIT THE DATASET IN A RATIO OF 80:20 as training and testing datasets
def split_sets(dataframe_all):
    train_set, test_set = train_test_split(dataframe_all, test_size=0.2, random_state=0)
    train_set.to_csv("training_set.csv", index=False, header=None)
    test_set.to_csv("test_set.csv", index=False, header=None)
    return load_tensor_data("training_set.csv"), load_tensor_data("test_set.csv")


# FUNCTION TO LOAD THE DATASET INTO TENSORFLOW
def load_tensor_data(dataset):
    return tf.contrib.learn.datasets.base.load_csv_without_header(
                filename=dataset,
                target_dtype=np.int,
                features_dtype=np.float32,
                target_column=-1)


# FUNCTION TO DEFINE THE INPUTS FOR TENSORFLOW INPUT FUNCTION
def get_inputs(data_set):
    data = tf.constant(data_set.data)
    target = tf.constant(data_set.target)
    return data, target


# FUNCTION TO CREATE A NEURAL NETWORK OF 3 LAYERS
def construct_net(num_features, model_dir):
    feature_cols = [tf.contrib.layers.real_valued_column("", dimension=num_features)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2,
                                                model_dir=model_dir)
    return classifier


# FUNCTION TO FIT THE MODEL WITH CUSTOM INPUTS
def fit_model(model, train_data, steps):
    model.fit(input_fn=lambda: get_inputs(train_data), steps=steps)
    print("\nModel trained after %s steps." % steps)


# FUNCTION TO SCORE THE MODEL WITH THE CUSTOM DATA
def evaluate_model(model, test_data, steps):
    accuracy_score = model.evaluate(input_fn=lambda: get_inputs(test_data), steps=steps)["accuracy"]
    print("\nModel Accuracy: {0:f}\n".format(accuracy_score))


# FUNCTION TO REQUEST FOR NEW SAMPLE FOR CLASSIFICATION
def new_samples(feature_names):
    request_input = 0
    while int(request_input) not in [1, 2]:
        request_input = input(
            "Predict classification: Enter own data (1) or simulate fake data (2)?\n Enter 1 or 2: ")
    if int(request_input) == 1:
        sample = np.array([[int(input("Enter value 0-10 for %s: " % x)) for x in feature_names]], dtype=np.float32)
    else:
        sample = np.array([np.random.randint(11, size=len(feature_names))], dtype=np.float32)
        print("Data generated:")
        for i, x in enumerate(feature_names):
            print("%s: %s" % (x, i))
    return sample


# FUNCTION FOR PREDICTIONS
def predict_class(model, binary_mappings):
    predict_loop = 'Y'
    while predict_loop.upper() == 'Y':
        binary_prediction = list(model.predict(input_fn=lambda: new_samples(feature_names)))
        print("\nClass Prediction: %s\n" % binary_mappings[binary_prediction[0]])
        predict_loop = input("Would you like to try another prediction? Enter Y/N: ")


# MAIN FUNCTION
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    model_dir = 'nn_classifier'
    cancer_data = 'breast-cancer-wisconsin.data'
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'
    clean_run(model_dir=model_dir)
    feature_names = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
                     'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses']
    column_names = ['id'] + feature_names + ['class']
    download_data(cancer_data, data_url)
    cancer_df = process_source(cancer_data, column_names, missing_vals='?', drop_cols=['id'])
    replace_classification_labels(cancer_df, result_col='class', values_to_replace=[2, 4])
    train_set, test_set = split_sets(cancer_df)
    dnn_model = construct_net(num_features=9, model_dir=model_dir)
    fit_model(dnn_model, train_set, steps=2000)
    evaluate_model(dnn_model, test_set, steps=1)
    predict_class(dnn_model, {0: 'benign', 1: 'malignant'})
