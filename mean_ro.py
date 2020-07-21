import numpy as np
import pandas as pd
import math
from scipy.special import gamma as gamma_fn
from scipy import stats

def likelihood(X, beta, gamma):
    return ((gamma ** beta) ** (X.shape[0])) * np.exp(np.sum(np.log((X ** (beta - 1)) * np.exp(-gamma * X)))) / (gamma_fn(beta) ** (X.shape[0]))
def log_likelihood(X, beta, gamma):
    n = X.shape[0]
    s = beta * n * np.log(gamma) + (beta - 1) * np.log(X).sum() - gamma * X.sum() - n * np.log(gamma_fn(beta))
    return np.exp(s)

def beta_target(X, beta, gamma):
    return (log_likelihood(X, beta, gamma) + 1e-5) * stats.gamma.pdf(beta, a = 1, scale = 1e-3)
def gamma_target(X, beta, gamma):
    return (log_likelihood(X, beta, gamma) + 1e-5) * stats.gamma.pdf(gamma, a = 1, scale = 1e-3)


def metropolis_hastings(X, beta_target = beta_target, gamma_target = gamma_target, num_samples = 50000):
    beta_0 = np.random.gamma(shape = 1, scale = 1e-3)
    gamma_0 = np.random.gamma(shape = 1, scale = 1e-3)
    ls = [(beta_0, gamma_0)]
    for i in range(num_samples - 1):
        beta_curr, gamma_curr = ls[-1]
        beta_proposed =  beta_curr + np.random.rand()
        gamma_proposed = gamma_curr + np.random.rand()

        beta_ratio = beta_target(X, beta_proposed, gamma_curr) / beta_target(X, beta_curr, gamma_curr)
        gamma_ratio = gamma_target(X, beta_curr, gamma_proposed) / gamma_target(X, beta_curr, gamma_curr)

        beta_next = beta_curr if beta_ratio < np.random.rand() else beta_proposed
        gamma_next = gamma_curr if gamma_ratio < np.random.rand() else gamma_proposed
        ls.append((beta_next, gamma_next))
    return np.array(ls)


data = pd.read_csv("data.csv")
confirmed = data['confirmed'].to_numpy()
recovered = data['recovered'].to_numpy()
data = confirmed - recovered
samplers = metropolis_hastings(data)
print(samplers)
print(samplers[:, 0].mean() / samplers[:, 1].mean())