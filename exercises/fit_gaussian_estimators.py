from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X_samples = np.random.normal(loc=10, scale=1,size=1000) #draws 1000 N(1,10) samples
    univariateGaussian = UnivariateGaussian()
    univariateGaussian.fit(X_samples) #fits X_samples and calculates mu & var
    print("(" + str(univariateGaussian.mu_) +" , "+ str(univariateGaussian.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    abs_distance = []
    sample_size = []
    for i in range(10, len(X_samples) + 1, 10):
        mu_hat = np.mean(X_samples[0 : i+1]) #calculates mean for X_samples in range [0:i+1]
        abs_distance.append(abs(mu_hat - univariateGaussian.mu_))
        sample_size.append(i)

    go.Figure([go.Scatter(x=sample_size, y=abs_distance, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{absolute distance between the estimated- and true value of the"
                                     r" expectation as a function of the sample size}$" ,
                               xaxis_title="$m\\text{ - number of samples size}$",
                               yaxis_title="r$|\hat\mu - u|$",
                               height=300)).show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    samples_sort = np.sort(X_samples)
    pdf_y_axis = univariateGaussian.pdf(samples_sort)

    go.Figure([go.Scatter(x=samples_sort, y=pdf_y_axis, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Sample values and their PDF}$",
                               xaxis_title="$\\text{X_samples}$",
                               yaxis_title="$\\text{PDF of the samples}$",
                               height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    u = np.array([0,0,4,0])
    cov = np.array([[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]])

    X_samples = np.random.multivariate_normal(u, cov, 1000)

    multivariateGaussian = MultivariateGaussian()
    multivariateGaussian.fit(X_samples)
    print(multivariateGaussian.mu_)
    print(multivariateGaussian.cov_)
    print(multivariateGaussian.log_likelihood(u,cov,X_samples))
    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = f1
    log_like_array = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            log_like_array[i, j] = multivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]), np.array(cov), X_samples)

    go.Figure(go.Heatmap(x=f1, y=f3, z=log_like_array),layout = go.Layout(title=r"$\text{log_likelihood of X_samples with u=[f1,0,f3,0] and the given Cov Matrix}$",
                       xaxis_title="$\\text{f1}$",
                       yaxis_title="$\\text{f3}$",
                       height=500, width=1000)).show()

    # Question 6 - Maximum likelihood

    argMax = np.argmax(log_like_array, axis=None)
    row = (argMax//200)
    col = argMax % 200
    print(str(f1[row]) +" , "+ str(f3[col]))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    # X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #           -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(UnivariateGaussian.log_likelihood(1,1,X))
    # print(UnivariateGaussian.log_likelihood(10,1,X))
