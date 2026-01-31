import numpy as np
class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
                
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]
