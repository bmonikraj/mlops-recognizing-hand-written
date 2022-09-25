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
    acc_list_train = []
    acc_list_test = []
    acc_list_dev = []
    for g in gamma_list:
        for c in c_list:
            h_params = {
                'gamma': g,
                'C': c
            }
            clf_ = svm.SVC()
            clf_.set_params(**h_params)
            clf_.fit(X_train, y_train)
            result = {
                    'accuracy': metrics.classification_report(y_train, clf_.predict(X_train), output_dict=True)['accuracy'],
                    'gamma': g,
                    'c': c
                }
            acc_list_train.append(
                result
            )
            result = {
                    'accuracy': metrics.classification_report(y_test, clf_.predict(X_test), output_dict=True)['accuracy'],
                    'gamma': g,
                    'c': c
                }
            acc_list_test.append(
                result
            )
            result = {
                    'accuracy': metrics.classification_report(y_dev, clf_.predict(X_dev), output_dict=True)['accuracy'],
                    'gamma': g,
                    'c': c
                }
            acc_list_dev.append(
                result
            )
            hyperparam_search.append(
                {
                    "params": h_params,
                    "train_acc": metrics.classification_report(y_train, clf_.predict(X_train), output_dict=True)['accuracy'],
                    "test_acc": metrics.classification_report(y_test, clf_.predict(X_test), output_dict=True)['accuracy'],
                    "dev_acc": metrics.classification_report(y_dev, clf_.predict(X_dev), output_dict=True)['accuracy']
                }
            )
    best_hyper_param = {
        "train_acc": max(acc_list_train, key=lambda x: x['accuracy']),
        "test_acc": max(acc_list_test, key=lambda x: x['accuracy']),
        "dev_acc": max(acc_list_dev, key=lambda x: x['accuracy']),
    }
    df = pd.DataFrame(hyperparam_search)
    print(df)
    print("Train Acc stats")
    print(f"Min {df['train_acc'].min()}")
    print(f"Max {df['train_acc'].max()}")
    print(f"Mean {df['train_acc'].mean()}")
    print(f"Median {df['train_acc'].median()}")
    print("\n")
    print("Test Acc stats")
    print(f"Min {df['test_acc'].min()}")
    print(f"Max {df['test_acc'].max()}")
    print(f"Mean {df['test_acc'].mean()}")
    print(f"Median {df['test_acc'].median()}")
    print("\n")
    print("Dev Acc stats")
    print(f"Min {df['dev_acc'].min()}")
    print(f"Max {df['dev_acc'].max()}")
    print(f"Mean {df['dev_acc'].mean()}")
    print(f"Median {df['dev_acc'].median()}")
    print("\n")
    return best_hyper_param
