import numpy as np
class SGD:
    def __init__(self, params, lr=3e-4):  
        self.params = list(params)
        self.lr = lr
        
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            
            if np.any(np.isnan(grad)):
                p.grad = None
                continue
            if np.any(np.isinf(grad)):
                grad = np.clip(grad, -100, 100)
            grad = np.clip(grad, -1000, 1000)
            p.data -= self.lr * grad
            
            if np.any(np.isnan(p.data)):
                p.data += self.lr * grad  
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None