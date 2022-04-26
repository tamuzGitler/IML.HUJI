
from IMLearn.utils import split_train_test

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.learners.regressors import LinearRegression


#
PATH = "..\\datasets\house_prices.csv"
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = pd.get_dummies(df , columns=['zipcode'])
    df['house_age'] = df[['yr_built', 'yr_renovated']].max(axis=1)
    df.drop(['yr_built', 'yr_renovated', "id", "date", "lat", "long", "sqft_living15", "sqft_lot15","sqft_lot", "bedrooms"], axis=1, inplace=True)
    df = df.loc[df["price"]>0]
    df = df.loc[df["sqft_living"] > 0]


    X =  df.iloc[:, 1::] # get X
    Y = df["price"] # get Y
    return X, Y

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X =  X.T
    for feature, column in X.iterrows():
        if "zipcode" in feature:
            continue
        cov_mat = np.cov(column, y) #returns 2x2 matrix with cov of [[(x,x), (x,y)],[(y,x)(y,y)]]

        cov_between_x_y = cov_mat[0][1]
        x_sig = np.sqrt(cov_mat[0][0])
        y_sig = np.sqrt(cov_mat[1][1])

        cur_pearson_correlation = cov_between_x_y / (x_sig * y_sig)


        fig = px.scatter(x=column, y= y,
                         title= f"Correlation between price and {feature} = { cur_pearson_correlation.round(5)}",
                         labels = dict(x=feature, y="house price"))

        pio.write_image(fig, (output_path + "/pearson_correlation_" + str(feature) + ".jpeg"))



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, Y = load_data(PATH)

    # # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, Y)
    #
    # # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test =  split_train_test(X,Y)
    #
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    i = 0
    mean_loss, std_loss = np.zeros(91) , np.zeros(91) #91 because in range 10-100% include there are 91 %

    x_train["price"] = y_train #combine

    for p in range(10,101):
        loss = []
        for round in range(10):

            # step 1
            train = x_train.sample(frac=p / 100)  # get p% of train
            y_samples = train["price"]
            x_samples = train.drop(labels="price", axis=1)
            linearRegression = LinearRegression()

            # step 2
            linearRegression._fit(x_samples.to_numpy(), y_samples.to_numpy())

            # step 3
            mse = linearRegression._loss(x_test.to_numpy(), y_test.to_numpy())
            loss.append(mse) #stores the mse

        # step4
        std_loss[i] = np.std(loss) #calc variance of loss
        mean_loss[i] = np.mean(loss) #calc average of loss
        i += 1


    #init variables for plt
    p_percentage = np.arange(10, 101)
    pos_confidence_interval = (mean_loss + 2 * std_loss)
    neg_confidence_interval = (mean_loss - 2 * std_loss)

    #plot
    fig = go.Figure((go.Scatter(x=p_percentage, y=mean_loss, mode="lines+markers", name="Mean Loss", marker=dict(color="blue", opacity=.7)),
                          go.Scatter(x=p_percentage, y=neg_confidence_interval, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=p_percentage, y=pos_confidence_interval, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),),
    layout = go.Layout(title=r"$\text{Mean loss as a function of p%}$",
                       xaxis={"title": "p%"},
                       yaxis={"title": "Mean Loss"},
                       height=400))
    fig.show()









