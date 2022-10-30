# Author: Monik R Behera (M20AIE258)

from sklearn import datasets, svm, metrics, tree
import numpy as np
from joblib import dump, load
import pandas as pd

from utils import data_viz, preprocess_digits, train_dev_test_split, hyperparam_search, get_hyperparameters, tune_and_save, pred_image_viz, confusion_matrix_viz

import warnings
warnings.filterwarnings('ignore')

gamma_list = [0.0001, 0.001, 0.01, 0.1]
c_list = [1, 2, 3, 4, 5]
ccp_alpha_list = [0.0, 0.2, 0.4, 0.6, 0.8]
min_impurity_decrease_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8]

train_test_dev_split = [
    (0.5, 0.25, 0.25),
    (0.6, 0.2, 0.2),
    (0.7, 0.15, 0.15),
    (0.8, 0.1, 0.1),
    (0.9, 0.05, 0.05)
]

params = {}
params['svc'] = {}
params['dtc'] = {}

params['svc']['gamma'] = gamma_list
params['svc']['C'] = c_list

params['dtc']['ccp_alpha'] = ccp_alpha_list
params['dtc']['min_impurity_decrease'] = min_impurity_decrease_list

digits = datasets.load_digits()

# data_viz(digits)

data, label = preprocess_digits(digits)

del digits

#
# Training and evaluation
#

clf_svc = svm.SVC()
hyper_params_set_svc = get_hyperparameters("svc", params['svc'])

clf_dtree = tree.DecisionTreeClassifier()
hyper_params_set_dtc = get_hyperparameters("dtc", params['dtc'])

metric = metrics.accuracy_score

model_eval_runs_results = {
    "run": [],
    "svm": [],
    "decision_tree": []
}
model_eval_runs_counter = 1

for data_split in train_test_dev_split:
    assert data_split[0] + data_split[1] + data_split[2] == 1.0
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, data_split[0], data_split[2]
    )

    model_file_svc = tune_and_save(
        hyper_params_set_svc,
        clf_svc,
        X_train,
        y_train,
        X_test,
        y_test,
        X_dev,
        y_dev,
        metric,
        model_path=str(model_eval_runs_counter)+"-svc.joblib"
    )
    best_model_svc = load(model_file_svc)
    predicted_svc = best_model_svc.predict(X_test)

    model_file_dtree = tune_and_save(
        hyper_params_set_dtc,
        clf_dtree,
        X_train,
        y_train,
        X_test,
        y_test,
        X_dev,
        y_dev,
        metric,
        model_path=str(model_eval_runs_counter)+"-dtree.joblib"
    )
    best_model_dtree = load(model_file_dtree)
    predicted_dtree = best_model_dtree.predict(X_test)

    model_eval_runs_results["run"].append(
        model_eval_runs_counter
    )
    model_eval_runs_counter += 1
    model_eval_runs_results["svm"].append(
        metric(y_pred=predicted_svc, y_true=y_test)
    )
    model_eval_runs_results["decision_tree"].append(
        metric(y_pred=predicted_dtree, y_true=y_test)
    )

model_svm_runs = model_eval_runs_results["svm"]
model_dtree_runs = model_eval_runs_results["decision_tree"]

model_eval_runs_results["run"].append(
    'mean'
)
model_eval_runs_results["run"].append(
    'std'
)

model_eval_runs_results["svm"].append(
    np.mean(np.array(model_svm_runs))
)
model_eval_runs_results["svm"].append(
    np.std(np.array(model_svm_runs))
)

model_eval_runs_results["decision_tree"].append(
    np.mean(np.array(model_dtree_runs))
)
model_eval_runs_results["decision_tree"].append(
    np.std(np.array(model_dtree_runs))
)

df_model_runs_eval = pd.DataFrame(model_eval_runs_results)
print(df_model_runs_eval.to_markdown(index=False))


# 
# Visualizations
#

X_train_best_svc, y_train_best_svc, X_dev_best_svc, y_dev_best_svc, X_test_best_svc, y_test_best_svc = train_dev_test_split(
    data, label, 0.9, 0.05
)
X_train_best_dtree, y_train_best_dtree, X_dev_best_dtree, y_dev_best_dtree, X_test_best_dtree, y_test_best_dtree = train_dev_test_split(
    data, label, 0.7, 0.15
)

best_model_svc = load('5-svc.joblib')
best_predicted_svc = best_model_svc.predict(X_test_best_svc)

best_model_dtree = load('3-dtree.joblib')
best_predicted_dtree = best_model_dtree.predict(X_test_best_dtree)

# pred_image_viz(X_test_best_svc, best_predicted_svc)
# pred_image_viz(X_test_best_dtree, best_predicted_dtree)

# confusion_matrix_viz(y_test_best_svc, best_predicted_svc, best_model_svc)
# confusion_matrix_viz(y_test_best_dtree, best_predicted_dtree, best_model_dtree)

print("")
print(f"Classification report for classifier {best_model_svc}:\n")
print(f"{metrics.classification_report(y_test_best_svc, best_predicted_svc)}\n")
