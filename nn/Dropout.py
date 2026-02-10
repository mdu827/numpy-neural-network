import numpy as np
from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import Module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True  
        self.mask = None     
    
    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        
        self.mask = (np.random.random(x.data.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * self.mask * scale
        
        def _backward():
            if x.requires_grad:
                grad = out.grad * self.mask * scale
                Tensor._add_grad(x, grad)
        
        out = x._create_child(
            out_data,
            (x,),
            f"dropout(p={self.p})",
            _backward
        )
        return out
    
    def eval(self):
        self.training = False
    
    def train(self, mode=True):
        self.training = mode
    
    def __repr__(self):
        return f"Dropout(p={self.p}, training={self.training})"