from feature_vector_creation import *
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# dns_features = create_feature_vector_from_log_file('3JUL.log', extract_features_reduced)
# print(dns_features[:100])
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylim([0, 1])
