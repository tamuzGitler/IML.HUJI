from __future__ import annotations

from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np

from itertools import product

from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # init
        self.sign_ = 1
        n_samples = X.T.shape[0]
        pos_errors, neg_errors = np.zeros(n_samples), np.zeros(n_samples)
        pos_thresholds, neg_thresholds = np.zeros(n_samples), np.zeros(n_samples)

        # find thresholds and errors for sign and -sign
        for j in range(n_samples):
            pos_thresholds[j], pos_errors[j] = self._find_threshold(X[:, j], y, self.sign_)
            neg_thresholds[j], neg_errors[j] = self._find_threshold(X[:, j], y, -self.sign_)

        # find min error of sign and -sign
        pos_index = np.argmin(pos_errors)
        self.j_, self.threshold_, min_pos_err = pos_index, pos_thresholds[pos_index], pos_errors[pos_index]

        neg_index = np.argmin(neg_errors)
        neg_min_index, min_neg_thr, min_neg_err = neg_index, neg_thresholds[neg_index], neg_errors[neg_index]

        # update to neg
        if min_neg_err < min_pos_err:
            self.threshold_ = min_neg_thr
            self.j_ = neg_min_index
            self.sign_ = -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        y_hat = np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)
        return y_hat

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        thr = values[0]  # init threshold with first value o
        thr_err = 1
        sign_of_labels = np.sign(labels)

        for curThr in values:

            y_predict = sign * np.where(curThr <= values, 1, -1)  # predict y with cur threshold
            epsilons = np.abs(labels[sign_of_labels != y_predict])
            curError = np.sum(epsilons) / np.sum(np.abs(labels))

            # check if we found better threshold
            if thr_err > curError:
                thr = curThr  # update threshold
                thr_err = curError  # update error

        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
