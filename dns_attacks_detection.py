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

# dns_features = create_feature_vector_from_log_file('3JUL.log', extract_features_with_letters_and_numbers)
# print(dns_features[:100])
def split_train_and_test_sets(data, target_variable):
    """
    Splits a given feature vector in a 80% train set and 20% test set.
    Uses Stratified Sampling with the variable passed in "target_variable".
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=13)
    for train_index, test_index in split.split(data, data[target_variable]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
    return train_set, test_set

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plots the precision and recall curves vs threshold.
    Used to analyse the SGD classifiers.
    """
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0, 1])

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
