# Author: Monik R Behera (M20AIE258)

from sklearn import datasets, svm, metrics
import numpy as np

from utils import data_viz, preprocess_digits, train_dev_test_split, hyperparam_search

import warnings
warnings.filterwarnings('ignore')

gamma_list = [0.0001, 0.001, 0.01, 0.1]
c_list = [1, 2, 3, 4, 5]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

data_viz(digits)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

#############################################################################
# Resize of image
'''
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

# Split data into 50% train and 50% test subsets
X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)


best_params = hyperparam_search(gamma_list, c_list, X_train, y_train, X_dev, y_dev, X_test, y_test)

print("Best Params:\n",best_params)

GAMMA = best_params['dev_acc']['gamma']
C = best_params['dev_acc']['c']

# Create a classifier: a support vector classifier
clf = svm.SVC()

hyper_params = {'gamma':GAMMA, 'C':C}
clf.set_params(**hyper_params)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()
