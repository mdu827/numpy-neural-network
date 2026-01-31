import numpy as np 
class RMSprop:
    def __init__(self, params, lr=3e-4, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.cache = [np.zeros_like(p.data) for p in params]
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.cache[i] = self.alpha * self.cache[i] + (1 - self.alpha) * (p.grad ** 2)
            p.data -= self.lr * p.grad / (np.sqrt(self.cache[i]) + self.eps)