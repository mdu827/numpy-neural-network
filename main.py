from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import *
from numpy_neural_network.optimizer.sgd import *
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

data = load_diabetes()
X_np, Y_np = data.data, data.target.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)
X_train = Tensor(X_train, requires_grad=False)
Y_train = Tensor(Y_train, requires_grad=False)
X_test = Tensor(X_test, requires_grad=False)
Y_test = Tensor(Y_test, requires_grad=False)
model = Sequential(
    Linear(X_train.data.shape[1], 16),
    Tanh(),
    Linear(16,1)
)

y_pred = model(X_test)
mse = ((y_pred - Y_test) ** 2).mean().data.item()
print("Test MSE before training:", mse)

optimizer = SGD(model.parameters(), lr=0.001)
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Прямой проход
    y_pred = model(X_train)
    
    # MSE Loss
    loss = ((y_pred - Y_train) ** 2).mean()
    
    # Обратный проход
    loss.backward()
    
    # Шаг оптимизатора
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss: {loss.data}")

y_pred = model(X_test)
mse = ((y_pred - Y_test) ** 2).mean().data.item()
print("Test MSE after training:", mse)
