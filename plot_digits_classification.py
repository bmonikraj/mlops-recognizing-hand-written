# Author: Monik R Behera (M20AIE258)

from sklearn import datasets, svm, metrics
import numpy as np
from joblib import dump, load

from utils import data_viz, preprocess_digits, train_dev_test_split, hyperparam_search, get_hyperparameters, tune_and_save, pred_image_viz

import warnings
warnings.filterwarnings('ignore')

gamma_list = [0.0001, 0.001, 0.01, 0.1]
c_list = [1, 2, 3, 4, 5]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

params = {}
params['gamma'] = gamma_list
params['C'] = c_list

hyper_params_set = get_hyperparameters(params)

digits = datasets.load_digits()

data_viz(digits)

'''
# Resize of images 

r_images = []
for i in range(len(digits.images)):
    r_images.append(resize(digits.images[i], (32,32), anti_aliasing=True))
digits.images = np.array(r_images)

# flatten the images
n_samples = len(digits.images)
print(f"Image shape resized : {digits.images[0].shape}")
data = digits.images.reshape((n_samples, -1))
'''

data, label = preprocess_digits(digits)

del digits

assert train_frac + dev_frac + test_frac == 1.0

X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

clf = svm.SVC()
metric = metrics.accuracy_score

model_file = tune_and_save(
    hyper_params_set,
    clf,
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    y_test,
    metric,
    model_path=None
)

best_model = load(model_file)

print(model_file)
print(clf)
print(best_model)

predicted = best_model.predict(X_test)

pred_image_viz(X_test, predicted)

print(f"Classification report for classifier {best_model}:\n")
print(f"{metrics.classification_report(y_test, predicted)}\n")
