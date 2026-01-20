import numpy as np
class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        _prev=(),
        _op="",
        _backward=None,
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

        # autograd internals
        self._prev = set(_prev)
        self._op = _op
        self._backward = _backward or (lambda: None)

    '''
    -------------
    basic properties
    -------------
    '''
    def shape(self):
        return self.data.shape

    def dtype(self):
        return self.data.dtype

    def size(self):
        return self.data.size

    def ndim(self):
        return self.data.ndim

    def item(self):
        if self.size() != 1:
            raise ValueError("tensor contains more than 1 item or is empty")
        return self.data.item()

    '''
    -------------
    helpers
    -------------
    '''
    @staticmethod
    def _add_grad(tensor, grad):
        if tensor.grad is None:
            tensor.grad = np.zeros_like(tensor.data)
        tensor.grad += grad

    @staticmethod
    def unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def _create_child(self, data, prev, op, backward_fn):
        requires_grad = any(t.requires_grad for t in prev)
        return Tensor(
            data=data,
            requires_grad=requires_grad,
            _prev=prev,
            _op=op,
            _backward=backward_fn if requires_grad else None,
        )

    '''
    -------------
    ops
    -------------
    '''
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(out.grad, self.data.shape),
                )
            if other.requires_grad:
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(out.grad, other.data.shape),
                )

        out = self._create_child(
            self.data + other.data,
            (self, other),
            "+",
            _backward,
        )
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(grad, self.data.shape),
                )
            if other.requires_grad:
                grad = self.data * out.grad
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(grad, other.data.shape),
                )

        out = self._create_child(
            self.data * other.data,
            (self, other),
            "*",
            _backward,
        )
        return out
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            # dA = dC @ B.T
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(grad_self, self.data.shape)
                )

            # dB = A.T @ dC
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(grad_other, other.data.shape)
                )

        out = self._create_child(
            self.data @ other.data,
            (self, other),
            "@",
            _backward,
        )
        return out

    '''
    -------------
    reductions
    -------------
    '''
    def sum(self):
        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) * out.grad
                Tensor._add_grad(self, grad)

        out = Tensor(
            data=np.array(self.data.sum()),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="sum",
            _backward=_backward if self.requires_grad else None,
        )
        return out

    def mean(self):
        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data) / self.data.size
                Tensor._add_grad(self, grad * out.grad)

        out = Tensor(
            data=np.array(self.data.mean()),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op="mean",
            _backward=_backward if self.requires_grad else None,
        )
        return out

    '''
    -------------
    autograd engine
    -------------
    '''
    def zero_grad(self):
        self.grad = None

    def _topo_sort(self):
        visited = set()
        topo = []

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        return topo

    def backward(self):
        if self.data.size != 1:
            raise RuntimeError("backward() only supported for scalar tensors")

        self.grad = np.ones_like(self.data)

        for node in reversed(self._topo_sort()):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

