import numpy as np

class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]  
        self.v = [np.zeros_like(p.data) for p in params]  
        self.t = 0
    
    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None