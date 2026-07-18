import numpy as np
from activation import ActivationRegistry


class Perceptron:
    def __init__(self, input_dim, output_dim, activation='linear', eta=0.01, beta=1.0):
        self.eta = eta
        self.beta = beta
        self.W = np.random.normal(0, 1, (output_dim, input_dim + 1))
        self.activation = ActivationRegistry.get(activation)

    def _add_bias(self, V):
        return np.insert(V, 0, 1)

    def forward(self, V_input):
        V_with_bias = self._add_bias(V_input)
        h = self.W @ V_with_bias
        O = self.activation.compute(h, self.beta)
        return O, h, V_with_bias

    def train_sample(self, V_input, target):
        O, h, V_with_bias = self.forward(V_input)
        error = target - O
        delta = error * self.activation.derivative(h, self.beta)
        self.W += self.eta * np.outer(delta, V_with_bias)
        return np.sum(error ** 2)

    def predict(self, X):
        V_bias = np.insert(X, 0, 1, axis=1)
        h = V_bias @ self.W.T
        return self.activation.compute(h, self.beta)
