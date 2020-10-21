# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone


def _mysign(x):
    return 2 * (x >= 0) - 1


class EnsembleIV:

    def __init__(self, adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100):
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        return

    def _check_input(self, Z, T, Y):
        if len(T.shape) == 1:
            T = T.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        return Z, T, Y.flatten()

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def fit(self, Z, T, Y):
        Z, T, Y = self._check_input(Z, T, Y)
        max_value = self.max_abs_value
        adversary = self._get_new_adversary().fit(Z, Y.flatten())
        learners = []
        h = 0
        for it in range(self.n_iter):
            test = adversary.predict(Z).flatten()
            aug_T = np.vstack([np.zeros((2, T.shape[1])), T])
            aug_label = np.concatenate(([-1, 1], _mysign(test)))
            aug_weights = np.concatenate(([0, 0], np.abs(test)))
            learners.append(self._get_new_learner().fit(
                aug_T, aug_label, sample_weight=aug_weights))
            h = h * it / (it + 1)
            h += max_value * _mysign(learners[it].predict_proba(T)[
                :, -1] * learners[it].classes_[-1] - 1 / 2) / (it + 1)
            adversary.fit(Z, Y - h)

        self.learners = learners
        return self

    def predict(self, T):
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)


class EnsembleIVStar:

    def __init__(self, adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100):
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        return

    def _check_input(self, Z, T, Y):
        if len(T.shape) == 1:
            T = T.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        return Z, T, Y.flatten()

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=5, max_depth=2,
                                     bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.0001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def _update_test(self, Z, Y, pred_old, adv):
        best_loss = np.mean((Y - pred_old)**2)
        pred_new = pred_old.copy()
        for gamma in np.linspace(.1, .9, 5):
            adv.fit(Z, Y - gamma * pred_old)
            pred = adv.predict(Z).flatten()
            loss = np.mean(
                (Y - gamma * pred_old - pred)**2)
            if loss <= best_loss:
                pred_new = gamma * pred_old + pred
                best_loss = loss
        return pred_new

    def fit(self, Z, T, Y):
        Z, T, Y = self._check_input(Z, T, Y)
        max_value = self.max_abs_value
        adversary = self._get_new_adversary()
        test = np.zeros(Z.shape[0])
        h = 0
        learners = []
        for it in range(self.n_iter):
            test = self._update_test(Z, Y - h, test, adversary)
            aug_T = np.vstack([np.zeros((2, T.shape[1])), T])
            aug_label = np.concatenate(([-1, 1], _mysign(test)))
            aug_weights = np.concatenate(([0, 0], np.abs(test)))
            learners.append(self._get_new_learner().fit(
                aug_T, aug_label, sample_weight=aug_weights))
            h = h * it / (it + 1)
            h += max_value * _mysign(learners[it].predict_proba(T)[
                :, -1] * learners[it].classes_[-1] - 1 / 2) / (it + 1)

        self.learners = learners
        return self

    def predict(self, T):
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)
