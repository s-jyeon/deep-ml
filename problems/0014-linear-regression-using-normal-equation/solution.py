import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	XtX = np.transpose(X) @ X
	XtX_inv = np.linalg.inv(XtX)
	Xty = np.transpose(X) @ y
	theta = XtX_inv @ Xty
	return theta