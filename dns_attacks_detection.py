from feature_vector_creation import *
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier

# dns_features = create_feature_vector_from_log_file('3JUL.log', extract_features_with_letters_and_numbers)
# print(dns_features[:100])
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0, 1])

def evaluate_model_with_precision_and_recall(model, X_test, y_test):
    final_predictions = model.predict(X_test)
    final_precision = precision_score(y_test, final_predictions)
    final_recall = recall_score(y_test, final_predictions)
    final_f1 = f1_score(y_test, final_predictions)
    return final_precision, final_recall, final_f1
