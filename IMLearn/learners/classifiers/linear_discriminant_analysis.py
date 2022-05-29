from typing import NoReturn

from .. import UnivariateGaussian
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True) #Find the unique elements of an array and frequency

        n_features = X.shape[1]
        n_classes = self.classes_.shape[0]
        n_samples = X.shape[0]

        self.pi_ = np.zeros([self.classes_.size])
        self.mu_ = np.zeros([n_classes , n_features])
        self.cov_ = np.zeros([n_features , n_features])

        #calcs mean and pi
        for sample_i,k in enumerate(self.classes_):
            X_of_k = X[y == k]  # get all X that belongs to class k
            self.pi_[sample_i] = X_of_k.shape[0] / X.shape[0]  # cal
            for column in range(X_of_k.shape[1]):
                self.mu_[sample_i][column] = np.mean(X_of_k[:,column])

        #calcs cov
        for sample_i in range(n_samples):
            self.cov_ += np.outer((X[sample_i,:] - self.mu_[y[sample_i],:]), (X[sample_i,:] - self.mu_[y[sample_i],:]))

        self.cov_ = self.cov_ / n_samples
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True


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


        n_samples = X.shape[0]

        pred = np.zeros([n_samples])
        for i, x in enumerate(X):
            y_pred = np.argmax( np.log(self.pi_) + self.likelihood(x))
            pred[i] = y_pred
        return pred



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

        #calculation log-likelihood
        logLike = (X @ self._cov_inv @ self.mu_.T) - 0.5 *np.diag(self.mu_ @ self._cov_inv @ self.mu_.T)
        return logLike


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