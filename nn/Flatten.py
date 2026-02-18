import numpy as np
from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import Module

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        
        if self.end_dim == -1:
            self.end_dim = len(x.shape) - 1
        
        flattened_size = 1
        for i in range(self.start_dim, self.end_dim + 1):
            flattened_size *= x.shape[i]
        
        new_shape = (
            x.shape[:self.start_dim] + 
            (flattened_size,) + 
            x.shape[self.end_dim + 1:]
        )
        
        out_data = x.data.reshape(new_shape)
        
        def _backward():
            if x.requires_grad:
                grad = out.grad.reshape(self.input_shape)
                Tensor._add_grad(x, grad)
        
        return x._create_child(out_data, (x,), "flatten", _backward)
    
    def __call__(self, x):
        return self.forward(x)
    
    def extra_repr(self):
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"