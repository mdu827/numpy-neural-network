from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import *
from numpy_neural_network.optimizer.adam import Adam
from numpy_neural_network.optimizer.sgd import SGD
from numpy_neural_network.optimizer.momentum import Momentum
from numpy_neural_network.optimizer.rmsprop import RMSprop
from sklearn.datasets import load_diabetes
import numpy as np
data = load_diabetes()
X = data.data
y = data.target.reshape(-1, 1)

X = (X - X.mean(axis=0)) / (X.std(axis=0))

X_t = Tensor(X, requires_grad=False)
y_t = Tensor(y, requires_grad=False)

model = Sequential(
    Linear(X.shape[1], 32),
    ReLU(),
    Linear(32, 1)
)
optimizer = SGD(model.parameters())
for epoch in range(10000):
    y_pred = model(X_t)
    loss = ((y_pred - y_t) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: mse loss = {loss.item():,.1f}")

with Tensor.no_grad():
    final_pred = model(X_t)
    mse = ((final_pred - y_t) ** 2).mean().item()
    mae = np.abs(final_pred.data - y.data).mean()

print(f"MSE: {mse:,.1f}")
print(f"MAE: {mae:.1f}")