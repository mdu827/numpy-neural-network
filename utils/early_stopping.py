class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights and model is not None:
                self.best_state = self._save_model_state(model)
            
            return False
        else:
            self.counter += 1
            print(f"EarlyStopping no update:  {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                
                if self.restore_best_weights and self.best_state is not None:
                    self._load_model_state(model, self.best_state)
                    print(f" Best model parameters from epoch (loss={self.best_loss:.4f})")
            
            return self.early_stop
    
    def _save_model_state(self, model):
        state = []
        for param in model.parameters():
            state.append(param.data.copy())
        return state
    
    def _load_model_state(self, model, state):
        for param, saved_data in zip(model.parameters(), state):
            param.data = saved_data
    
    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None