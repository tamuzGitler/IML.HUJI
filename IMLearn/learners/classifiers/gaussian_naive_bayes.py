from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from IMLearn.learners.gaussian_estimators import UnivariateGaussian
class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # self.classes_, counts = np.unique(y, return_counts=True)  # Find the unique elements of an array and frequency
        # num_of_classes = (self.classes_).size[0]

        # self.pi_ = counts / num_of_classes  # calcs class probabilities


        # self.mu_ = (1 / counts) * np.count_nonzero(X[y == counts])
        #
        # self.vars_ = np.diag(np.diag(np.cov(counts, rowvar=False)))
        self.classes_ = np.unique(y)  # Find the unique elements of an array and frequency
        # self.mu_ = dict()
        # self.vars_ = dict()
        # self.pi_ = dict()
        #
        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]
        self.pi_ = np.zeros([n_classes])
        self.mu_ = np.zeros([n_classes , n_features])
        self.vars_ = np.zeros([n_classes, n_features])

        self.s = dict()
        for i,k in enumerate(self.classes_):
            X_of_k = X[y == k] #get all X that belongs to class k
            self.pi_[i] = X_of_k.shape[0] / X.shape[0]  # calcs k probability
            self.mu_[i] = np.mean(X_of_k, axis=0)


        for i in range(n_classes):
            self.vars_[i] = X[(y == self.classes_[i]),:].var(axis=0, ddof=1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_classes = self.classes_.shape[0]
        n_samples = X.shape[0]
        likeli = np.zeros((n_samples, n_classes))
        for j in range(n_classes):
            div1 = (2 * self.vars_[j])
            log_exp = -(np.power(X-self.mu_[j],2)) / div1

            likeli[:, j] = self.pi_[j] * np.prod(np.exp(log_exp) / np.sqrt(self.vars_[j] * 2 * np.pi), axis=1)
        return likeli

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
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        misclassification_loss = misclassification_error(y, y_pred)
        return misclassification_loss