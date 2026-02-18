import numpy as np
from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import Module

class Conv2d(Module):
    """
    2D convolution layer
        out_height = floor((height + 2*padding - kernel_size) / stride + 1)
        out_width = floor((width + 2*padding - kernel_size) / stride + 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        
        scale = np.sqrt(2.0 / (in_channels * np.prod(self.kernel_size)))
        
        self.weight = Tensor(
            np.random.randn(
                out_channels, 
                in_channels, 
                self.kernel_size[0], 
                self.kernel_size[1]
            ) * scale,
            requires_grad=True
        )
        
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True
        )
        
        print(f"  Conv2d: {in_channels}→{out_channels}, kernel={kernel_size}, stride={stride}")
    
    def _pair(self, x):
        if isinstance(x, (int, float)):
            return (x, x)
        return x
    
    def _get_output_shape(self, input_shape):
        batch, in_c, h, w = input_shape
        out_h = (h + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (w + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        return (batch, self.out_channels, out_h, out_w)
    
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Conv2d expected 4D input got {x.ndim} instead")
        
        batch_size, in_channels, height, width = x.shape
        
        if in_channels != self.in_channels:
            raise ValueError(f"Channel number mismatch, expected {self.in_channels}, got {in_channels}")
        
        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode='constant'
            )
        else:
            x_padded = x.data
        
        out_h = (height + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (width + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride[0]
                        w_start = j * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        output[b, oc, i, j] = np.sum(patch * self.weight.data[oc]) + self.bias.data[oc]
        
        def _backward():
            if x.requires_grad:
                grad_input = np.zeros_like(x_padded)
                
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_start = i * self.stride[0]
                                w_start = j * self.stride[1]
                                h_end = h_start + self.kernel_size[0]
                                w_end = w_start + self.kernel_size[1]
                                
                                grad_input[b, :, h_start:h_end, w_start:w_end] += \
                                    self.weight.data[oc] * out.grad[b, oc, i, j]
                
                if self.padding != (0, 0):
                    pad_h, pad_w = self.padding
                    grad_input = grad_input[:, :, pad_h:-pad_h, pad_w:-pad_w]
                
                Tensor._add_grad(x, grad_input)
            
            if self.weight.requires_grad:
                grad_weight = np.zeros_like(self.weight.data)
                
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        for i in range(out_h):
                            for j in range(out_w):
                                h_start = i * self.stride[0]
                                w_start = j * self.stride[1]
                                h_end = h_start + self.kernel_size[0]
                                w_end = w_start + self.kernel_size[1]
                                
                                patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                                grad_weight[oc] += patch * out.grad[b, oc, i, j]
                
                Tensor._add_grad(self.weight, grad_weight)
            
            if self.bias.requires_grad:
                grad_bias = np.zeros_like(self.bias.data)
                
                for oc in range(self.out_channels):
                    grad_bias[oc] = out.grad[:, oc, :, :].sum()
                
                Tensor._add_grad(self.bias, grad_bias)
        
        return x._create_child(output, (x,), "conv2d", _backward)
    
    def __call__(self, x):
        return self.forward(x)
    
    def extra_repr(self):
        return f"{self.in_channels}→{self.out_channels}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}"