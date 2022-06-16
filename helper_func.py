import pandas as pd
from multiprocessing import Queue
import numpy as np

intervals_dict = {}
pos_queue = Queue()
neg_queue = Queue()


def plot_roc(clf, X, Y):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_roc_curve, roc_curve, auc

    plot_roc_curve(clf, X, Y)
    plt.show()


def find_Optimal_Cutoff(fpr, tpr, threshold):
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def compute_intervals(intervals_dict, X_train, num_bins=5):
    names = X_train.columns
    for name in names:
        unique_values = X_train[name].unique()
        if len(unique_values) > 2 or max(unique_values) != 1 or min(unique_values) != 0:
            intervals = pd.cut(X_train[name], num_bins)
            intervals_dict[name] = intervals


def get_relevant_features(zipped_data):
    global pos_queue
    global neg_queue

    shap_value, feature_value, feature_name, shap_threshold = zipped_data

    if shap_value != 0:
        if feature_value == 0:
            shap_value = -(shap_value)

        if shap_value > shap_threshold:
            name = format_name(feature_name, feature_value)
            pos_queue.put(name)
        elif shap_value < (-shap_threshold):
            name = format_name(feature_name, feature_value)
            neg_queue.put(name)


def append_to_encoded_vals(class_queue, itemset, encoded_vals):
    labels = {}
    rowset = set()

    while class_queue.qsize() > 0:
        rowset.add(class_queue.get())

    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))

    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)


def format_name(name, feature_value):
    global intervals_dict

    if name in intervals_dict:
        intervals = intervals_dict[name]
        for interval in intervals:
            if interval != interval: continue
            if feature_value in interval:
                left = interval.left
                right = interval.right
                name = f'{left}<{name}<={right}'
                break
    new_name = str(name).replace('less', '<').replace('greater', '>')
    return new_name


def clean_name(feature_name):
    if feature_name.split(' ')[0].strip().replace('.', '').isdigit():
        feature_name = feature_name.split(' ')[2].strip()
    else:
        feature_name = feature_name.split(' ')[0].strip()
    return feature_name
