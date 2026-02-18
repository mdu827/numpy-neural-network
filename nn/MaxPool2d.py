import numpy as np
from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import Module

class MaxPool2d(Module):
    """
    2D Max Pooling layer.
    """
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride) if stride is not None else self.kernel_size
        self.padding = self._pair(padding)
    
    def _pair(self, x):
        if isinstance(x, (int, float)):
            return (x, x)
        return x
    
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"MaxPool2d extected 4D input, got {x.ndim}D instead")
        
        batch_size, channels, height, width = x.shape
        
        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode='constant',
                constant_values=-np.inf
            )
        else:
            x_padded = x.data
        
        out_h = (height + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (width + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride[0]
                        w_start = j * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch = x_padded[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        output[b, c, i, j] = patch[max_idx]
                        max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])
        
        def _backward():
            if x.requires_grad:
                grad_input = np.zeros_like(x_padded)
                
                for b in range(batch_size):
                    for c in range(channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                h, w = max_indices[b, c, i, j]
                                grad_input[b, c, h, w] += out.grad[b, c, i, j]
                
                if self.padding != (0, 0):
                    pad_h, pad_w = self.padding
                    grad_input = grad_input[:, :, pad_h:-pad_h, pad_w:-pad_w]
                
                Tensor._add_grad(x, grad_input)
        
        return x._create_child(output, (x,), "maxpool2d", _backward)
    
    def __call__(self, x):
        return self.forward(x)