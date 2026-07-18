import numpy as np


class ActivationFunction:
    def __init__(self, name, func, deriv, beta=1.0):
        self.name = name
        self._func = func
        self._deriv = deriv
        self.beta = beta

    def compute(self, h, beta=None):
        if beta is None:
            beta = self.beta
        return self._func(h, beta)

    def derivative(self, h, beta=None):
        if beta is None:
            beta = self.beta
        return self._deriv(h, beta)


class ActivationRegistry:
    _functions = {}

    @classmethod
    def register(cls, act_func):
        cls._functions[act_func.name] = act_func

    @classmethod
    def get(cls, name):
        if name not in cls._functions:
            raise ValueError(f"Activation function '{name}' not found")
        return cls._functions[name]

    @classmethod
    def available(cls):
        return list(cls._functions.keys())


def _step_func(h, beta=1.0):
    return np.where(h >= 0, 1, -1)

def _step_deriv(h, beta=1.0):
    return np.ones_like(h)

def _linear_func(h, beta=1.0):
    return h

def _linear_deriv(h, beta=1.0):
    return np.ones_like(h)

def _sigmoid_func(h, beta=1.0):
    return 1 / (1 + np.exp(-np.clip(2 * beta * h, -500, 500)))

def _sigmoid_deriv(h, beta=1.0):
    gh = _sigmoid_func(h, beta)
    return 2 * beta * gh * (1 - gh)

def _tanh_func(h, beta=1.0):
    return np.tanh(beta * h)

def _tanh_deriv(h, beta=1.0):
    gh = _tanh_func(h, beta)
    return beta * (1 - gh ** 2)

def _relu_func(h, beta=1.0):
    return np.maximum(0, h)

def _relu_deriv(h, beta=1.0):
    return np.where(h > 0, 1, 0)

def _logistic_func(h, beta=1.0):
    return _sigmoid_func(h, beta)

def _logistic_deriv(h, beta=1.0):
    return _sigmoid_deriv(h, beta)


_registry = [
    ActivationFunction('step', _step_func, _step_deriv),
    ActivationFunction('linear', _linear_func, _linear_deriv),
    ActivationFunction('sigmoid', _sigmoid_func, _sigmoid_deriv),
    ActivationFunction('tanh', _tanh_func, _tanh_deriv),
    ActivationFunction('relu', _relu_func, _relu_deriv),
    ActivationFunction('logistic', _logistic_func, _logistic_deriv),
]

for af in _registry:
    ActivationRegistry.register(af)
