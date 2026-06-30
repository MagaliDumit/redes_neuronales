import numpy as np


class ActivationFunction:
    def __init__(self, name, func, deriv):
        self.name = name
        self._func = func
        self._deriv = deriv

    def compute(self, h):
        return self._func(h)

    def derivative(self, h):
        return self._deriv(h)


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


def _step_func(h):
    return np.where(h >= 0, 1, -1)

def _step_deriv(h):
    return np.ones_like(h)

def _linear_func(h):
    return h

def _linear_deriv(h):
    return np.ones_like(h)

def _sigmoid_func(h):
    return 1 / (1 + np.exp(-np.clip(h, -500, 500)))

def _sigmoid_deriv(h):
    s = _sigmoid_func(h)
    return s * (1 - s)

def _tanh_func(h):
    return np.tanh(h)

def _tanh_deriv(h):
    return 1 - np.tanh(h) ** 2

def _relu_func(h):
    return np.maximum(0, h)

def _relu_deriv(h):
    return np.where(h > 0, 1, 0)

def _logistic_func(h):
    return 1 / (1 + np.exp(-np.clip(2 * h, -500, 500)))

def _logistic_deriv(h):
    s = _logistic_func(h)
    return 2 * s * (1 - s)


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
