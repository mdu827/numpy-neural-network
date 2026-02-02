import numpy as np
class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(X)
    
    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]
