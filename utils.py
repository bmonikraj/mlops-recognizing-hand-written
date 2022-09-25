import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import pandas as pd

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

##############################################################################
# Hyperparameter search
def hyperparam_search(gamma_list, c_list, X_train, y_train, X_test, y_test, X_dev, y_dev):
    hyperparam_search = []
    acc_list = []
    for g in gamma_list:
        for c in c_list:
            h_params = {
                'gamma': g,
                'C': c
            }
            clf_ = svm.SVC()
            clf_.set_params(**h_params)
            clf_.fit(X_train, y_train)
            predicted = clf_.predict(X_test)
            result = {
                    'accuracy': metrics.classification_report(y_test, predicted, output_dict=True)['accuracy'],
                    'gamma': g,
                    'c': c
                }
            hyperparam_search.append(
                {
                    "params": h_params,
                    "train_acc": metrics.classification_report(y_train, clf_.predict(X_train), output_dict=True)['accuracy'],
                    "test_acc": metrics.classification_report(y_test, clf_.predict(X_test), output_dict=True)['accuracy'],
                    "dev_acc": metrics.classification_report(y_dev, clf_.predict(X_dev), output_dict=True)['accuracy']
                }
            )
            acc_list.append(
                result
            )
    best_hyper_param = max(acc_list, key=lambda x: x['accuracy'])
    print(pd.DataFrame(hyperparam_search))
    return best_hyper_param
