import numpy as np
def euler_method(t, coefficient, initial_state):
	""" Find the approximate solution of SIR model
		with Euler method

		Parameters
		----------
		t: int
			calculate S, I, R until time t
		coefficient: list, tuple
			coefficient = [beta, gamma]
			beta - contact coefficient
			gamma - recovery coefficient
		initial_state: list, tuple
			initial_state = [N, I0, R0]
		Returns:
			a array has t + 1 tuples (t, S, I, R) at time t
	"""
	N, I0, R0 = initial_state
	S0 = N - I0 - R0
	beta, gamma = coefficient
	alpha = beta / N
	
	vt1 = np.array([-1, 1, 0])
	vt2 = np.array([0, -1, 1])
	result = np.empty((t + 1, 3))
	result[0] = np.array([S0, I0, R0])
	delta_t = 0.1
	num_steps = int(t / delta_t)
	num_per_steps = int (1 / delta_t)
	# for i in range(1, t + 1):
	# 	delta = alpha * np.prod(result[i - 1, : 2]) * vt1 + gamma * result[i - 1, 1] * vt2
	# 	result[i] = result[i - 1] + 2 * delta

	for i in range(num_per_steps, num_steps + num_per_steps):
		idx = i // num_per_steps
		if i % num_per_steps == 0:
			delta = alpha * np.prod(result[idx - 1, : 2]) * vt1 + gamma * result[idx - 1, 1] * vt2
			result[idx] = result[idx - 1] + delta_t * delta
		else:
			delta = alpha * np.prod(result[idx, : 2]) * vt1 + gamma * result[idx, 1] * vt2
			result[idx] = result[idx] + delta_t * delta

	return result
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
print(euler_method(8, [0.001 * 510, 0.2], [510, 10, 0]))