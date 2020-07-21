import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def target(x):
    if isinstance(x, np.ndarray):
        return np.exp(-x)
    return 0 if x < 0 else np.exp(-x)
x = np.empty((100000, ))
x[0] = 5
for i in range(x.size - 1):
    curr_x = x[i]
    proposed_x = curr_x + np.random.randn()
    ratio = target(proposed_x) / target(curr_x)
    if ratio < np.random.rand():
        x[i + 1] = x[i]
    else:
        x[i + 1] = proposed_x
data = plt.hist(x)
print(data[0])
plt.plot(data[0])
plt.show()