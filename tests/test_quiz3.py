from joblib import load
import sys
import numpy as np
from sklearn import datasets

sys.path.append(".")

from utils import preprocess_digits

def helper_create_sample_digit_data(n=100):
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    x = data[np.random.choice(a=data.shape[0], size=n, replace=False)]
    return x

def helper_model_predict_get_unique_classes():
    model = load('svm_gamma=0.0001_C=4.joblib')
    x = helper_create_sample_digit_data(n=100)
    y_hyp = model.predict(x)
    unique_predicted_classes = np.unique(y_hyp)
    return unique_predicted_classes.shape[0]

def test_classifier_not_biased_one_class():
    unique_classes_count = helper_model_predict_get_unique_classes()
    assert unique_classes_count != 1

def test_classifier_predicts_all_classes():
    TOTAL_CLASSES_TRAIN = 10
    unique_classes_count = helper_model_predict_get_unique_classes()
    assert unique_classes_count == TOTAL_CLASSES_TRAIN