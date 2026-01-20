from numpy_neural_network.tensor import Tensor
from numpy_neural_network.nn.Module import Linear

x = Tensor([[1., 2., 3.]], requires_grad=True)  # (1,3)

lin = Linear(3, 2)

y = lin(x)
loss = y.sum()
loss.backward()

print("y:", y.data)
print("x.grad:", x.grad)
print("W.grad:", lin.W.grad)
print("b.grad:", lin.b.grad)
