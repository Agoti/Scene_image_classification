from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import json
import numpy as np

# Metrics of multiclass classification

class Metrics:

    _all = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'auc': roc_auc_score,
        'confusion_matrix': confusion_matrix,
    }

    def __init__(self, metrics: list = None):
        self.metrics = set()
        if metrics:
            for metric in metrics:
                self.add_metric(metric)
        self.results = {'overall': {}, 'detailed': {}}

    def add_metric(self, name):
        if name == 'all':
            self.metrics = set(self._all.keys())
            return
        if name not in self._all:
            raise ValueError(f'Invalid metric name: {name}')
        self.metrics.add(name)
    
    @staticmethod
    def _get_metric_fn(name):
        return Metrics._all[name]
    
    def compute(self, y_true, y_pred):

        classes = np.unique(y_true).tolist()
        for name in self.metrics:
            metric_fn = self._get_metric_fn(name)
            if name == 'confusion_matrix':
                cm = confusion_matrix(y_true, y_pred)
                # convert to list for json serialization
                self.results['detailed'][name] = cm.tolist()
            elif name == 'auc':
                for c in classes:
                    y_true_c = np.where(y_true == c, 1, 0)
                    y_pred_c = np.where(y_pred == c, 1, 0)
                    if name not in self.results['detailed']:
                        self.results['detailed'][name] = {}
                    self.results['detailed'][name][c] = metric_fn(y_true_c, y_pred_c)
                self.results['overall'][name] = sum(self.results['detailed'][name].values()) / len(classes)
            else:
                for c in classes:
                    y_true_c = np.where(y_true == c, 1, 0)
                    y_pred_c = np.where(y_pred == c, 1, 0)
                    if name not in self.results['detailed']:
                        self.results['detailed'][name] = {}
                    self.results['detailed'][name][c] = metric_fn(y_true_c, y_pred_c)
                if name == 'accuracy':
                    self.results['overall'][name] = metric_fn(y_true, y_pred)
                else:
                    self.results['overall'][name] = sum(self.results['detailed'][name].values()) / len(classes)
        
        # verify
        cm = confusion_matrix(y_true, y_pred)
        precisions = np.diag(cm) / np.sum(cm, axis=0)
        recalls = np.diag(cm) / np.sum(cm, axis=1)
        f1s = 2 * precisions * recalls / (precisions + recalls)
        self.results['verify'] = {}
        self.results['verify']['accuracy'] = accuracy_score(y_true, y_pred)
        self.results['verify']['precision'] = np.mean(precisions)
        self.results['verify']['recall'] = np.mean(recalls)
        self.results['verify']['f1_score'] = np.mean(f1s)
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)
        self.results['verify']['auc'] = roc_auc_score(y_true_bin, y_pred_bin, average='macro')

        return self.results
    
    def save(self, path):
        with open(path, 'w') as f:
            # neat json output
            json.dump(self.results, f, indent=4)
        print(f'Saved metrics to {path}')
