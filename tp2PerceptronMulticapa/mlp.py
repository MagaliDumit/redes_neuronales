import numpy as np
from activation import ActivationRegistry


class ModularMultiLayerPerceptron:
    def __init__(self, layer_dims, activations, eta=0.01, beta=1.0):
        self.eta = eta
        self.beta = beta
        self.n_layers = len(layer_dims) - 1
        self.activations = [ActivationRegistry.get(a) for a in activations]
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.W.append(np.random.uniform(-limit, limit, (fan_out, fan_in)))
            self.b.append(np.zeros((fan_out, 1)))

    def forward(self, V_input):
        V = V_input
        self._h_list = []
        self._V_list = [V]
        for i in range(self.n_layers):
            h = self.W[i] @ V + self.b[i].flatten()
            self._h_list.append(h)
            V = self.activations[i].compute(h, self.beta)
            self._V_list.append(V)
        return V

    def backward(self, target):
        O = self._V_list[-1]
        delta = (O - target) * self.activations[-1].derivative(self._h_list[-1], self.beta)
        self._dW = []
        self._db = []
        for i in reversed(range(self.n_layers)):
            V_prev = self._V_list[i]
            self._dW.insert(0, np.outer(delta, V_prev))
            self._db.insert(0, delta.reshape(-1, 1))
            if i > 0:
                delta = (self.W[i].T @ delta) * self.activations[i - 1].derivative(self._h_list[i - 1], self.beta)

    def _update_weights(self):
        for i in range(self.n_layers):
            self.W[i] -= self.eta * self._dW[i]
            self.b[i] -= self.eta * self._db[i]

    def train_sample(self, V_input, target):
        self.forward(V_input)
        self.backward(target)
        self._update_weights()
        O = self._V_list[-1]
        return np.sum((target - O) ** 2)

    def train_batch(self, X, y):
        acc_dW = [np.zeros_like(w) for w in self.W]
        acc_db = [np.zeros_like(b) for b in self.b]
        total_error = 0
        for V, t in zip(X, y):
            self.forward(V)
            self.backward(t)
            for i in range(self.n_layers):
                acc_dW[i] += self._dW[i]
                acc_db[i] += self._db[i]
            O = self._V_list[-1]
            total_error += np.sum((t - O) ** 2)
        n = len(X)
        for i in range(self.n_layers):
            self.W[i] -= (self.eta / n) * acc_dW[i]
            self.b[i] -= (self.eta / n) * acc_db[i]
        return total_error / n

    def predict(self, X):
        return np.array([self.forward(V) for V in X])
