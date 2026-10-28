import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mu = np.mean(data, axis = 0)
	sig = np.std(data, axis = 0)
	standardized_data = np.round((data - mu) / sig, decimals = 4)
	normalized_data = np.round((data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0)), decimals = 4)

	return standardized_data, normalized_data