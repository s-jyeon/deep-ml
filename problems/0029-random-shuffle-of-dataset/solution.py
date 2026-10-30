import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	n = X.shape[0]
	row_indices = np.random.permutation(n)
	return (X[row_indices], y[row_indices])
