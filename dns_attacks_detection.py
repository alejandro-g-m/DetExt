from feature_vector_creation import *
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib


"""
Plotting functions
"""


def plot_dataset(X, y, xlabel, ylabel, alpha=0.1, legend_position='lower right'):
    """
    Plots the dataset.
    If there are more than two features, only the first two will be displayed.
    """
    # Plot the no attacks (y==0) with the two features
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs', alpha=alpha, label="No attack")
    # Plot the attacks (y==1) with the two features
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r^', alpha=alpha, label="Attack")
    plt.grid(True, which='both')
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    leg = plt.legend(loc=legend_position, fontsize=16)
    # In order to show the legend with an alpha of 1
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plots the precision and recall curves vs threshold.
    Used to analyse the SGD classifiers.
    It expects the "precisions" and "recalls" that are returned from the function
    "precision_recall_curve", so it removes the last value, as there is no
    corresponding threshold.
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0, 1])
    plt.grid(True, which='both')

def plot_predictions_for_logistic_regression(clf, axes):
    """
    Plots the predictions for a logistic regression model in a 2-D graph.
    """
    # Generate 1000 numbers for predictions, from the lowest axes to the highest
    x0s = np.linspace(axes[0], axes[1], 1000)
    x1s = np.linspace(axes[2], axes[3], 1000)
    # Create two matrices with all the numbers generated before
    # x0 represents the first feature, x1 the second feature
    x0, x1 = np.meshgrid(x0s, x1s)
    # Combine the two matrices to generate a combined feature vector
    X = np.c_[x0.ravel(), x1.ravel()]
    # Get the predicitons of the clasifier for our generated feature vector
    # y_proba holds two predictions (one per class) for each generated instance in X
    y_proba = clf.predict_proba(X)
    # Generate the different decision probability boundaries
    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, 4) # Countour lines, 4 data intervals
    # Generate the main decision boundary
    left_right = np.array([axes[0], axes[1]])
    boundary = -(clf.coef_[0][0] * left_right + clf.intercept_[0]) / clf.coef_[0][1]
    # Plot the boundary lines that have been just generated
    plt.clabel(contour, inline=1, fontsize=12) # Print contour lines with label
    plt.plot(left_right, boundary, 'k--', linewidth=3)

def plot_predictions_for_SVC(clf, axes):
    """
    Plots the predictions and decision function values for a SVC model in a 2-D graph.
    """
    # Generate 100 numbers for predictions, from the lowest axes to the highest
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    # Create two matrices with all the numbers generated before
    # x0 represents the first feature, x1 the second feature
    x0, x1 = np.meshgrid(x0s, x1s)
    # Combine the two matrices to generate a combined feature vector
    X = np.c_[x0.ravel(), x1.ravel()]
    # Get the predicitons of the clasifier for our generated feature vector
    y_pred = clf.predict(X).reshape(x0.shape)
    # Get the decision function of the sample for each class in the model
    # (this is the distance of the samples to the separating boundary or hyperplane)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    # Plot the area of the predictions
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2) # Filled countour
    # Plot the are of the decision function
    plt.contourf(x0, x1, y_decision, 10, cmap=plt.cm.brg, alpha=0.1) # Filled contour, 10 data intervals

def plot_predictions_for_KNN(clf, axes):
    """
    Plots the predictions for a KNN model in a 2-D graph.
    """
    # Generate 1000 numbers for predictions, from the lowest axes to the highest
    x0s = np.linspace(axes[0], axes[1], 1000)
    x1s = np.linspace(axes[2], axes[3], 1000)
    # Create two matrices with all the numbers generated before
    # x0 represents the first feature, x1 the second feature
    x0, x1 = np.meshgrid(x0s, x1s)
    # Combine the two matrices to generate a combined feature vector
    X = np.c_[x0.ravel(), x1.ravel()]
    # Get the predicitons of the clasifier for our generated feature vector
    y_pred = clf.predict(X).reshape(x0.shape)
    # Plot a color area around the different neighbors depending on their predicitons
    plt.pcolormesh(x0, x1, y_pred, cmap=plt.cm.binary)


"""
Data preparation functions
"""


def split_train_and_test_sets(data, target_variable, test_size=0.2):
    """
    Splits a given feature vector in a 80% train set and 20% test set.
    Uses Stratified Sampling with the variable passed in "target_variable".
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=13)
    for train_index, test_index in split.split(data, data[target_variable]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
    return train_set, test_set


"""
Validation, evaluation and scoring functions
"""


def cross_validate_models(models, features, labels, scoring, cv=5, n_jobs=-1, return_train_score=False):
    """
    Gets a list of models and the data to train them.
    Returns the results of doing cross validation to all of them using the
    "scoring", "cv" and "n_jobs" passed as parameters.
    """
    results = []
    for model in models:
        cv_results = cross_validate(model, features, labels, scoring=scoring, cv=cv, n_jobs=n_jobs, return_train_score=return_train_score)
        results.append(cv_results)
    return results

def get_cross_validate_scores(cv_results, names, scoring):
    """
    Gets a list of cross validation results and the name of the models that have been used.
    Returns the test scores of these models as given in "scoring".
    It assumes that all the scores in scoring were returned by the cross validation.
    """
    cross_validate_scores = []
    for result, name in zip(cv_results, names):
        scores = {}
        for score in scoring:
            scores[score] = np.mean(result['test_' + score])
        cross_validate_scores.append((name, scores))
    return cross_validate_scores

def evaluate_model_with_precision_and_recall(model, X_test, y_test):
    """
    Evaluates the predictions of a "model" for the data "X_test".
    Returns the precision, recall and F1 scores after comparing the predictions
    to the real values contained in "y_test".
    """
    final_predictions = model.predict(X_test)
    final_precision = precision_score(y_test, final_predictions)
    final_recall = recall_score(y_test, final_predictions)
    final_f1 = f1_score(y_test, final_predictions)
    return final_precision, final_recall, final_f1
