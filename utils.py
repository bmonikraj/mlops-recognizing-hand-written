import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import copy
from joblib import dump

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def data_viz(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def get_hyperparameters(params):
    params_combinations = [{"gamma":g, "C":c} for g in params['gamma'] for c in params['C']]
    return params_combinations

def hyperparam_search(hyper_params, clf, X_train, y_train, X_test, y_test, X_dev, y_dev, metric):
    best_metric, best_model, best_h_params = -0.1, None, None
    
    for param in hyper_params:
        clf.set_params(**param)
        clf.fit(X_train, y_train)
        pre_dev = clf.predict(X_dev)
        cur_metric = metric(y_pred=pre_dev, y_true=y_dev)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = copy.copy(clf)
            best_h_params = param
            print("Found new best with:" + str(param))
            print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params

def tune_and_save(hyper_params, clf, X_train, y_train, X_test, y_test, X_dev, y_dev, metric, model_path):
    best_model, best_metric, best_h_params = hyperparam_search(
        hyper_params,
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        X_dev,
        y_dev,
        metric
    )
    best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )
    if type(clf) == svm.SVC:
        model_type = 'svm'
    else:
        model_type = 'none'

    best_model_file = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_file
    print(best_model)
    dump(best_model, model_path)

    print(f"Best hyperparameters: {best_h_params}")
    print(f"Best metric on Dev data: {best_metric}")
    return model_path
