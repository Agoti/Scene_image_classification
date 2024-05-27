# Metrics.py
# Description: This file is used to define the metrics class for evaluating the model.
# Author: Mingxiao Liu

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import json
import numpy as np

class Metrics:
    '''
    Metrics class
    Methods:
        add_metric: Add a metric to the metrics
        compute: Compute the metrics
        save: Save the metrics to the file
    '''

    # Supported metrics and their corresponding functions
    _all = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'auc': roc_auc_score,
        'confusion_matrix': confusion_matrix,
    }

    def __init__(self, metrics: list = None):
        '''
        Initialize the metrics
        Args:
            metrics: The list of metrics to compute
        '''

        self.metrics = set()

        # Add all metrics if not specified
        if metrics:
            for metric in metrics:
                self.add_metric(metric)

        # Results of the metrics
        # detailed: metrics for each class
        # overall: overall metrics
        self.results = {'overall': {}, 'detailed': {}}


    def add_metric(self, name):
        '''
        Add a metric to the metrics
        Args:
            name: The name of the metric
        '''

        # Add all metrics
        if name == 'all':
            self.metrics = set(self._all.keys())
            return
        
        # Check if the metric is valid
        if name not in self._all:
            raise ValueError(f'Invalid metric name: {name}')

        self.metrics.add(name)
    

    @staticmethod
    def _get_metric_fn(name):
        return Metrics._all[name]
    

    def compute(self, y_true, y_pred):
        '''
        Compute the metrics
        Args:
            y_true: The true labels
            y_pred: The predicted labels
        '''

        classes = np.unique(y_true).tolist()

        for name in self.metrics:

            metric_fn = self._get_metric_fn(name)

            # confusion matrix
            if name == 'confusion_matrix':
                cm = confusion_matrix(y_true, y_pred)
                # convert to list for json serialization
                self.results['detailed'][name] = cm.tolist()

            # auROC
            elif name == 'auc':
                for c in classes:
                    y_true_c = np.where(y_true == c, 1, 0)
                    y_pred_c = np.where(y_pred == c, 1, 0)
                    if name not in self.results['detailed']:
                        self.results['detailed'][name] = {}
                    self.results['detailed'][name][c] = metric_fn(y_true_c, y_pred_c)
                self.results['overall'][name] = sum(self.results['detailed'][name].values()) / len(classes)
            
            # other metrics: accuracy, precision, recall, f1_score
            else:
                # detailed metrics
                for c in classes:
                    y_true_c = np.where(y_true == c, 1, 0)
                    y_pred_c = np.where(y_pred == c, 1, 0)
                    if name not in self.results['detailed']:
                        self.results['detailed'][name] = {}
                    if name == 'precision':
                        self.results['detailed'][name][c] = metric_fn(y_true_c, y_pred_c, zero_division=0)
                    else:
                        self.results['detailed'][name][c] = metric_fn(y_true_c, y_pred_c)
                
                # overall metrics
                # accuracy: overall accuracy
                if name == 'accuracy':
                    self.results['overall'][name] = metric_fn(y_true, y_pred)
                # precision, recall, f1_score: average of the metrics of all classes
                else:
                    self.results['overall'][name] = sum(self.results['detailed'][name].values()) / len(classes)
        
        # verify
        # NOTE: used for testing the correctness of my implementation. will be removed in the final version
        # cm = confusion_matrix(y_true, y_pred)
        # eps = 1e-10
        # precisions = np.diag(cm) / (np.sum(cm, axis=0) + eps)
        # recalls = np.diag(cm) / (np.sum(cm, axis=1) + eps)
        # f1s = 2 * precisions * recalls / (precisions + recalls + eps)
        # self.results['verify'] = {}
        # self.results['verify']['accuracy'] = accuracy_score(y_true, y_pred)
        # self.results['verify']['precision'] = np.mean(precisions)
        # self.results['verify']['recall'] = np.mean(recalls)
        # self.results['verify']['f1_score'] = np.mean(f1s)
        # y_true_bin = label_binarize(y_true, classes=classes)
        # y_pred_bin = label_binarize(y_pred, classes=classes)
        # self.results['verify']['auc'] = roc_auc_score(y_true_bin, y_pred_bin, average='macro')

        return self.results
    
    def save(self, path):
        '''
        Save the metrics to the file
        Args:
            path: The path of the file
        '''

        with open(path, 'w') as f:
            # neat json output
            json.dump(self.results, f, indent=4)

        print(f'Metrics: {path} saved')
