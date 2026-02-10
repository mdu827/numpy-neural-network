from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import *
from numpy_neural_network.nn.Dropout import *
from numpy_neural_network.optimizer.adam import Adam
from numpy_neural_network.optimizer.sgd import SGD
from numpy_neural_network.optimizer.momentum import Momentum
from numpy_neural_network.optimizer.rmsprop import RMSprop
from numpy_neural_network.utils.early_stopping import EarlyStopping
from numpy_neural_network.utils.data import DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
data = load_diabetes()
X = data.data
y = data.target.reshape(-1, 1)

X = (X - X.mean(axis=0)) / (X.std(axis=0))
y = (y - y.mean(axis=0)) / (y.std(axis=0))

# X_t = Tensor(X, requires_grad=False)
# y_t = Tensor(y, requires_grad=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)


X_val = Tensor(X_val, requires_grad=False)
y_val = Tensor(y_val, requires_grad=False)
X_test = Tensor(X_test, requires_grad=False)
y_test = Tensor(y_test, requires_grad=False)


batch_size = 32
train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

model = Sequential(
    Linear(X.shape[1], 128),
    ReLU(),
    Dropout(p=0.3),
    Linear(128, 256),
    ReLU(),
    Dropout(p=0.3),
    Linear(256,1),
)

with Tensor.no_grad():
    prev_pred = model(X_val)
    mse = ((prev_pred - y_val) ** 2).mean().item()
    mae = np.abs(prev_pred.data - y_val.data).mean()

print("MSE and MAE before train:")
print(f"MSE: {mse:,.1f}")
print(f"MAE: {mae:.1f}")
optimizer = Adam(model.parameters())
early_stopper = EarlyStopping(
    patience=250, 
    min_delta=0.01,
    restore_best_weights=True
)

for epoch in range(25_000):
    epoch_loss = 0
    n_batches = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch_t = Tensor(X_batch, requires_grad=False)
        y_batch_t = Tensor(y_batch, requires_grad=False)

        y_pred = model(X_batch_t)
        loss = ((y_pred - y_batch_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
    if epoch % 50 == 0:
        with Tensor.no_grad():
            val_loss = ((model(X_val) - y_val) ** 2).mean().item()
        
        if early_stopper(val_loss, model):
            print(f"\n Early stopping on epoch: {epoch}")
            break
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: Avg Loss = {avg_loss:,.3f} "
              f"(batches: {n_batches}, total: {epoch_loss:,.3f})")

with Tensor.no_grad():
    final_pred = model(X_test)
    mse = ((final_pred - y_test) ** 2).mean().item()
    mae = np.abs(final_pred.data - y_test.data).mean()
print("MSE and MAE after train:")
print(f"MSE: {mse:,.1f}")
print(f"MAE: {mae:.1f}")