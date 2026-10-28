import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # 1. Initialize weights to zeros
    # 2. iterations
    for i in range(iterations):
        # 2.1 compute predictions: y_hat
        y_hat = X @ theta
        # 2.2 compute errors
        err = y_hat - y 
        # 2.3 compute gradient 
        grad = 1/m *( X.transpose() @ err) 
        # 2.4 update weights 
        theta = theta - (alpha * grad) 

    return theta.flatten()