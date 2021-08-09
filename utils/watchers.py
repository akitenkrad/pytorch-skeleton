from datetime import datetime, timezone, timedelta
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from .logger import get_logger

class SimpleWatcher(object):
    def __init__(self, name, default_value=0, order='ascending', patience=10):
        self.name = name
        self.default_value = default_value
        self.order = order
        self.data = []
        self.jst = timezone(timedelta(hours=9))
        self.patience = patience
        self.counter = 0
        self.best_score = default_value
        self.__is_best = False

    @property
    def early_stop(self):
        return self.counter > self.patience
    @property
    def is_best(self):
        return self.__is_best


    def put(self, x):
        timestamp = datetime.now(self.jst)
        self.data.append((x, timestamp))

        if self.order == 'ascending':
            if self.best_score < x:
                self.best_score = x
                self.counter = 0
                self.__is_best = True
            else:
                self.counter += 1
                self.__is_best = False
        if self.order == 'descending':
            if x < self.best_score:
                self.best_score = x
                self.counter = 0
                self.__is_best = True
            else:
                self.counter += 1
                self.__is_best = False

    def mean(self):
        if len(self.data) < 1:
            return self.default_value
        return np.mean([x[0] for x in self.data])

    def max(self):
        if len(self.data) < 1:
            return self.default_value
        return np.max([x[0] for x in self.data])

    def min(self):
        if len(self.data) < 1:
            return self.default_value
        return np.min([x[0] for x in self.data])

    def fps(self):
        if len(self.data) < 1:
            return self.default_value
        m = np.mean(np.diff([x[1] for x in self.data]))
        return 1.0 / m

class AucWatcher(object):
    def __init__(self, name, threshold=0.5, patience=10):
        self.name = name
        self.threshold = threshold
        self.data = []
        self.jst = timezone(timedelta(hours=9))
        self.patience = patience
        self.counter = 0
        self.best_auc = -1
        self.__is_best = False
        self.__min_eval_count = 4
        self.logger = get_logger('AucWatcher')

    @property
    def early_stop(self):
        return self.counter > self.patience
    @property
    def is_best(self):
        return self.__is_best

    @property
    def precision(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [int(self.threshold < item[0]) for item in self.data]
        y_true = [item[1] for item in self.data]
        score = precision_score(y_true, y_pred)
        return score

    @property
    def recall(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [int(self.threshold < item[0]) for item in self.data]
        y_true = [item[1] for item in self.data]
        score = recall_score(y_true, y_pred)
        return score

    @property
    def auc(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [item[0] for item in self.data]
        y_true = [item[1] for item in self.data]
        try:
            score = roc_auc_score(y_true, y_pred)
            return score
        except Exception as ex:
            self.logger.exception(ex)
            return -1

    def put(self, x, y):
        timestamp = datetime.now(self.jst)

        if isinstance(x, list) or isinstance(x, np.ndarray):
            if x.ndim == 0: x = [x]
            if y.ndim == 0: y = [y]
            for _x, _y in zip(x, y):
                self.data.append((_x, _y, timestamp))
        else:
            self.data.append((x, y, timestamp))

        if len(self.data) < self.__min_eval_count:
            self.__is_best = False
            return

        auc = self.auc
        if self.best_auc < auc:
            self.best_auc = auc
            self.counter = 0
            self.__is_best = True
        else:
            self.counter += 1
            self.__is_best = False
