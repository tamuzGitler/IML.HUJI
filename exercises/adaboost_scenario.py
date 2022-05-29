import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import misclassification_error, accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaBoost.fit(train_X, train_y)

    train_error = np.empty(n_learners)
    test_error = np.empty(n_learners)
    for t in range(n_learners):
        train_error[t] = np.array(adaBoost.partial_loss(train_X, train_y, t + 1))
        test_error[t] = np.array(adaBoost.partial_loss(test_X, test_y, t + 1))
    x_axis = np.arange(1, n_learners + 1)
    go.Figure([
        go.Scatter(x=x_axis, y=test_error, mode='lines', name=r'$test loss$'),
        go.Scatter(x=x_axis, y=train_error, mode='lines', name=r'$train loss$')]) \
        .update_layout(
        title=rf"$\text{{Training and Test errors as a function of the number of fitted learners }}$").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:

        learners_name = ["5 learners", "50 learners", "100 learners", "250 learners"]

        # # Add traces for data-points setting symbols and colors

        fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$ " for m in learners_name],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, t in enumerate(T):
            fig.add_traces(
                [decision_surface(lambda X: adaBoost.partial_predict(X, t), lims[0], lims[1], showscale=False),

                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=test_y, colorscale=custom,
                                        line=dict(color="black", width=1)))],
                rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig.show()

    # Question 3: Decision surface of best performing ensemble
    if noise == 0:
        min_err_index = int(np.argmin(test_error)) + 1
        acc = accuracy(test_y, adaBoost.partial_predict(test_X, min_err_index))
        fig3 = go.Figure()
        fig3.add_traces(
            [decision_surface(lambda X: adaBoost.partial_predict(X, min_err_index), lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, colorscale=custom,
                                    line=dict(color="black", width=1)))])
        fig3.update_layout(title="Lowest test error with accuracy " f"{acc} using "  f"{min_err_index}  learners ")
        fig3.show()

    # Question 4: Decision surface with weighted samples
    D = adaBoost.D_ / np.max(adaBoost.D_) * 5
    fig4 = go.Figure()
    fig4.add_traces([decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y, colorscale=custom, size=D,
                                            line=dict(color="black", width=1)))])
    fig4.update_layout(title="Training set with a point size proportional to it’s weight and color ")
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
