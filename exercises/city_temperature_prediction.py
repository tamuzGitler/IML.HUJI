import plotly.graph_objects as go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
PATH = "..\\datasets\\City_Temperature.csv"

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.loc[df["Temp"] >-72]

    return df



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(PATH)

    # Question 2 - Exploring data for specific country
    israel_data = data.loc[data["Country"] == "Israel"]

    #plot
    fig1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color=israel_data["Year"].astype(str),
                     title="average daily temperature change as a function of the DayOfYear")
    fig1.show()


    month_std_data = israel_data.groupby(["Month"])["Temp"].agg('std')
    months = np.arange(0,12)
    month_fig = px.bar(month_std_data, x=months, y="Temp",
                     title="Standard deviation of the daily temperatures for each month",
                      labels={"x": "Months", "Temp": "Standard deviation"}

                       )
    month_fig.show()

    # Question 3 - Exploring differences between countries
    county_month_data = data.groupby(['Country', 'Month']).agg(mean=("Temp", "mean"), std=("Temp", "std")).reset_index()
    fig2 = px.line(county_month_data, x="Month", y="mean", error_y="std", color="Country",
                  title="average monthly temperature, with error bars")
    fig2.show()

    # Question 4 - Fitting model for different values of `k`
    column_index = np.arange(0,8)
    column_index = np.delete(column_index, 6) #remove Temp column


    X = israel_data["DayOfYear"]
    Y = israel_data["Temp"]
    x_train, y_train, x_test, y_test =  split_train_test(X,Y)  #splits data by x-DayOfYear, y-temperature


    loss = []
    degrees =  range(1,11)
    for k in degrees:
        polynomialFitting = PolynomialFitting(k)
        polynomialFitting.fit(x_train.to_numpy(), y_train.to_numpy())
        poll_loss = polynomialFitting.loss(x_test.to_numpy(), y_test.to_numpy())
        loss.append(round(poll_loss, 2))

    #4.1
    print(loss)
    #4.2
    months = np.arange(0, 12)
    month_fig = px.bar(x=degrees, y=loss,
                     title= "test error recorded for each value of k",
                      labels={"x": "K - degree", "y": "error recorded"}

                       )
    month_fig.show()


    # Question 5 - Evaluating fitted model on different countries

    #train model base on israel - base on previous calc
    chosen_k = 5
    polynomialFitting = PolynomialFitting(5)
    polynomialFitting.fit(X.to_numpy(), Y.to_numpy())


    country_names = data["Country"].unique()
    country_names = country_names[country_names != "Israel"]
    loss_of_each_country = []

    for country in country_names:
        cur_data = data.loc[data["Country"] == country]
        train, test = cur_data["DayOfYear"], cur_data["Temp"]
        loss = polynomialFitting.loss(train.to_numpy(), test.to_numpy())
        loss_of_each_country.append(loss)

    print(loss_of_each_country)
    print(country_names)
    country_fig = px.bar(x=country_names, y=loss_of_each_country,
                     title= "Model’s error over each of the other countries with k=5",
                      labels={"x": "Country names", "y": "error recorded"}

                       )
    country_fig.show()

