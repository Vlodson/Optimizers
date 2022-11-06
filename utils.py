import numpy as np

def tanh(data: np.ndarray) -> np.ndarray:
    return np.tanh(data)

def dtanh(data: np.ndarray) -> np.ndarray:
    return 1 - tanh(data)**2

def sigmoid(data: np.ndarray) -> np.ndarray:
    return np.where(data > 0, 1 / (1 + np.exp(-data)), np.exp(data) / (1 + np.exp(data)))

def dsigmoid(data: np.ndarray) -> np.ndarray:
    return sigmoid(data)*(1 - sigmoid(data))

def MSLoss(y: np.ndarray, yHat: np.ndarray) -> float:
    return np.sum((y - yHat)**2) / (2*y.shape[0])