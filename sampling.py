import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def beta_target(x):
    return stats.gamma.pdf(x, a = 1, scale = .1)
def gamma_target(x):
    return stats.gamma.pdf(x, a = 1, scale = 1e-10)



def metropolis_hastings(beta_target = beta_target, gamma_target = gamma_target, num_samples = 10000):
    beta_0 = np.random.gamma(shape = 1, scale = .1)
    gamma_0 = np.random.gamma(shape = 1, scale = 1e-10)
    ls = [(beta_0, gamma_0)]
    for i in range(num_samples - 1):
        beta_curr, gamma_curr = ls[-1]
        beta_proposed = beta_curr + np.random.randn() * 1e-1
        gamma_proposed = gamma_curr + np.random.randn() * 1e-8
    

        beta_ratio = beta_target(beta_proposed) / beta_target(beta_curr)
        gamma_ratio = gamma_target(gamma_proposed) / gamma_target(gamma_curr)

        beta_next = beta_curr if beta_ratio < np.random.rand() else beta_proposed
        gamma_next = gamma_curr if gamma_ratio < np.random.rand() else gamma_proposed
        ls.append((beta_next, gamma_next))
    return np.array(ls)

# samplers = metropolis_hastings(beta_target, gamma_target)
# y, x  = plt.hist(samplers[:, 1], bins= 200)[:2]
# plt.plot(x[:-1], y)
# plt.show()
# # x = np.linspace (0, 100, 200) 
# # y1 = stats.gamma.pdf(x, a=1, scale =1e-3) #a is alpha, loc is beta???
# # plt.plot(x, y1, "y-", label=(r'$\alpha=29, \beta=3$')) 
# # plt.xlim([0,2])
# # plt.show()