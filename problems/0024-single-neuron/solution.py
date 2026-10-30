import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	z = np.round(features @ weights + bias, decimals = 4)
	probabilities = 1 / (1 + np.exp(-z))
	mse = np.round(np.sum((probabilities-labels) ** 2) / features.shape[0], decimals = 4)
	return probabilities, mse
