from numpy_neural_network.tensor import Tensor
import numpy as np

class Module:
    def parameters(self):
        params = []

        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)

            elif isinstance(value, Module):
                params.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class ReLU(Module):
    def __call__(self, x):
        out = None  # создаём переменную для замыкания

        def _backward():
            if x.requires_grad:
                # используем out.grad, которое будет установлено в Tensor
                grad = (x.data > 0) * out.grad
                Tensor._add_grad(x, grad)

        out = x._create_child(
            np.maximum(0, x.data),
            (x,),
            "relu",
            _backward
        )
        return out

class Tanh(Module):
    def __call__(self, x):
        out = None  # placeholder для замыкания

        def _backward():
            if x.requires_grad:
                grad = (1 - np.tanh(x.data) ** 2) * out.grad
                Tensor._add_grad(x, grad)

        out = x._create_child(
            np.tanh(x.data),  # данные после tanh
            (x,),             # prev
            "tanh",           # имя операции
            _backward         # backward функция
        )
        return out


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(
            np.random.randn(in_features, out_features) * 0.1,
            requires_grad=True
        )
        self.b = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

    def forward(self, x):
        return x @ self.W + self.b

    def __call__(self, x):
        return self.forward(x)


class MLP(Module):
    def __init__(self):
        self.l1 = Linear(2, 4)
        self.relu = ReLU()
        self.l2 = Linear(4, 1)

    def __call__(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
