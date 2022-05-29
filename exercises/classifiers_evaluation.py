from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

PATH = "..\\datasets/"

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset

        X_sample, y_true = load_dataset(PATH + f)
            # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback1(perceptron, X: np.ndarray, y: int):
            cur_loss = perceptron.loss(X_sample, y_true)
            losses.append(cur_loss)

        # init Perceptron
        perceptron = Perceptron(callback=callback1)
        # fit the Perceptron
        perceptron.fit(X_sample, y_true)

        #figure loss
        fitting_iteration = np.arange(0,len(losses), dtype=int)

        fig = go.Figure((go.Scatter(x=fitting_iteration, y=losses, mode="lines",
                                    marker=dict(color="blue", opacity=.7)),),
                        layout=go.Layout(title=r"$\text{Loss as function of fitting iteration}$",
                                         xaxis={"title": "Fitting Iteration"},
                                         yaxis={"title": "Loss"},
                                         height=400))
        fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_sample, y_true = load_dataset(PATH + f)

        # Fit models and predict over training set
        lda = LDA()
        gaussianNaiveBayes = GaussianNaiveBayes()

        lda.fit(X_sample, y_true)
        gaussianNaiveBayes.fit(X_sample, y_true)
        models = [gaussianNaiveBayes, lda]



        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        model_names = ["GaussianNaiveBayes", "LDA"]
        symbols = np.array(["square", "circle", "triangle-sw-dot"])
        lims = np.array([X_sample.min(axis=0), X_sample.max(axis=0)]).T + np.array([-.4, .4])


        # # Add traces for data-points setting symbols and colors

        fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$ " for m in model_names],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        accuracys = list()
        for j, m in enumerate(models):
            pred = m.predict(X_sample)
            accuracys.append(accuracy(y_true, pred))


            fig.add_traces([decision_surface(m.predict, lims[0], lims[1], showscale=False),

                            go.Scatter(x=X_sample[:, 0], y=X_sample[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y_true, symbol=symbols[y_true], colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))],
                           rows=(j // 2) + 1, cols=(j % 2) + 1)


        fig.update_layout(title="GaussianNaiveBayes with accuracy: " +str(accuracys[0]) + ", LDA with accuracy: " + str(accuracys[1]) +
                          " from " + f ,
                          margin=dict(t=100)) \
            .update_xaxes(visible=True).update_yaxes(visible=True)


        #
        # # Add `X` dots specifying fitted Gaussians' means
        for j, m in enumerate([lda, gaussianNaiveBayes]):
            fig.add_traces([
                go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color="black", symbol="x",
                                       line=dict(color="black", width=1)))],
                rows=1, cols=j + 1
            )
        # # Add ellipses depicting the covariances of the fitted Gaussians
        lda_rows = lda.mu_.shape[0]
        gaus_rows = gaussianNaiveBayes.mu_.shape[0]
        fig.add_traces([get_ellipse(lda.mu_[i, :], lda.cov_) for i in
                        range(lda_rows)], rows=1, cols=1)
        fig.add_traces(
            [get_ellipse(gaussianNaiveBayes.mu_[j, :], np.diag(gaussianNaiveBayes.vars_[j, :])) for j in
             range(gaus_rows)], rows=1, cols=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

