import numpy as np

class Tensor:
    _NO_GRAD = False
    
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
        if Tensor._NO_GRAD:
            self.requires_grad = False
        else:
            self.requires_grad = requires_grad
        self.grad = None

        self._prev = set(_prev)
        self._op = _op
        self._backward = _backward or (lambda: None)

    @staticmethod
    def no_grad():
        class NoGradContext:
            def __enter__(self):
                self.prev_state = Tensor._NO_GRAD
                Tensor._NO_GRAD = True
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                Tensor._NO_GRAD = self.prev_state
        
        return NoGradContext()
    '''
    -------------
    basic properties 
    -------------
    '''
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def item(self):
        if self.size != 1:
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
        if Tensor._NO_GRAD or not any(t.requires_grad for t in prev):
            requires_grad = False
            backward_fn = None
        else:
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
                    Tensor.unbroadcast(out.grad, self.shape),
                )
            if other.requires_grad:
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(out.grad, other.shape),
                )

        out = self._create_child(
            self.data + other.data,
            (self, other),
            "+",
            _backward,
        )
        return out
    
    def __radd__(self, other):
        """Reverse addition: other + self."""
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(out.grad, self.shape),
                )
            if other.requires_grad:
                Tensor._add_grad(
                    other,
                    -Tensor.unbroadcast(out.grad, other.shape),
                )

        out = self._create_child(
            self.data - other.data,
            (self, other),
            "-",
            _backward,
        )
        return out
    
    def __rsub__(self, other):
        """Reverse subtraction: other - self."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self
    
    def __neg__(self):
        """Negation: -self."""
        return self * -1
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(grad, self.shape),
                )
            if other.requires_grad:
                grad = self.data * out.grad
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(grad, other.shape),
                )

        out = self._create_child(
            self.data * other.data,
            (self, other),
            "*",
            _backward,
        )
        return out
    
    def __rmul__(self, other):
        """Reverse multiplication: other * self."""
        return self * other
    
    def __truediv__(self, other):
        """Division: self / other."""
        other = other if isinstance(other, Tensor) else Tensor(other)

        def _backward():
            if self.requires_grad:
                grad = (1.0 / other.data) * out.grad
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(grad, self.shape),
                )
            if other.requires_grad:
                grad = (-self.data / (other.data ** 2)) * out.grad
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(grad, other.shape),
                )

        out = self._create_child(
            self.data / other.data,
            (self, other),
            "/",
            _backward,
        )
        return out
    
    def __rtruediv__(self, other):
        """Reverse division: other / self."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError(f"Power must be int or float, got {type(power)}")

        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad
                Tensor._add_grad(
                    self,
                    Tensor.unbroadcast(grad, self.shape)
                )

        out = self._create_child(
            self.data ** power,
            (self,),
            f"**{power}",
            _backward
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
                    Tensor.unbroadcast(grad_self, self.shape)
                )

            # dB = A.T @ dC
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                Tensor._add_grad(
                    other,
                    Tensor.unbroadcast(grad_other, other.shape)
                )

        out = self._create_child(
            self.data @ other.data,
            (self, other),
            "@",
            _backward,
        )
        return out
    
    def __rmatmul__(self, other):
        """Reverse matmul: other @ self."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self

    '''
    -------------
    reductions 
    -------------
    '''
    def sum(self, axis=None, keepdims=False):
        def _backward():
            if self.requires_grad:
                if axis is None:
                    grad = np.ones_like(self.data) * out.grad
                else:
                    grad = np.ones_like(self.data) * out.grad
                    if not keepdims:
                        grad = grad.reshape(self.shape)
                Tensor._add_grad(self, grad)

        out = Tensor(
            data=np.array(self.data.sum(axis=axis, keepdims=keepdims)),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op=f"sum(axis={axis})",
            _backward=_backward if self.requires_grad else None,
        )
        return out

    def mean(self, axis=None, keepdims=False):
        def _backward():
            if self.requires_grad:
                if axis is None:
                    n = self.size
                else:
                    if isinstance(axis, int):
                        n = self.shape[axis]
                    else:
                        n = np.prod([self.shape[i] for i in axis])
                grad = np.ones_like(self.data) * out.grad / n
                if not keepdims:
                    grad = grad.reshape(self.shape)
                Tensor._add_grad(self, grad)

        out = Tensor(
            data=np.array(self.data.mean(axis=axis, keepdims=keepdims)),
            requires_grad=self.requires_grad,
            _prev=(self,),
            _op=f"mean(axis={axis})",
            _backward=_backward if self.requires_grad else None,
        )
        return out
    
    def relu(self):
        out_data = np.maximum(0, self.data)

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0) * out.grad
                Tensor._add_grad(self, grad)

        return self._create_child(
            out_data,
            (self,),
            "relu",
            _backward
        )
    
    def sigmoid(self):
        def sigmoid(x):
            x = np.clip(x, -50, 50)
            return 1 / (1 + np.exp(-x))
        
        out_data = sigmoid(self.data)

        def _backward():
            if self.requires_grad:
                grad = out_data * (1 - out_data) * out.grad
                Tensor._add_grad(self, grad)

        return self._create_child(
            out_data,
            (self,),
            "sigmoid",
            _backward
        )
    
    def tanh(self):
        out_data = np.tanh(self.data)

        def _backward():
            if self.requires_grad:
                grad = (1 - out_data ** 2) * out.grad
                Tensor._add_grad(self, grad)

        return self._create_child(
            out_data,
            (self,),
            "tanh",
            _backward
        )

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

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() called on tensor with requires_grad=False")
        
        if self.size != 1 and grad is None:
            raise RuntimeError(
                "backward() requires grad for non-scalar tensors or "
                "scalar output. Use backward(grad=...) for non-scalar outputs."
            )
        
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            self.grad = np.array(grad, dtype=np.float64)
        
        for node in reversed(self._topo_sort()):
            if node._backward is not None:
                node._backward()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)
    
    @property
    def T(self):
        return self.transpose()
    
    def transpose(self, *axes):
        if not axes:
            axes = None
        elif len(axes) == 1:
            axes = axes[0]
        
        out_data = self.data.transpose(axes)
        
        def _backward():
            if self.requires_grad:
                if axes is None:
                    inv_axes = tuple(reversed(range(self.ndim)))
                else:
                    inv_axes = tuple(np.argsort(axes))
                grad = out.grad.transpose(inv_axes)
                Tensor._add_grad(self, grad)
        
        return self._create_child(
            out_data,
            (self,),
            f"transpose{axes}",
            _backward,
        )
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        out_data = self.data.reshape(shape)
        
        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                Tensor._add_grad(self, grad)
        
        return self._create_child(
            out_data,
            (self,),
            f"reshape{shape}",
            _backward,
        )

    def __repr__(self):
        grad_info = ""
        if self.grad is not None:
            grad_norm = np.linalg.norm(self.grad)
            grad_info = f", grad_norm={grad_norm:.4f}"
        
        return (f"Tensor(shape={self.shape}, "
                f"requires_grad={self.requires_grad}{grad_info}, "
                f"op='{self._op}')")
    
    def __str__(self):
        return f"Tensor{self.shape}"