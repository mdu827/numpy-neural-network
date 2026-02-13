# Numpy-only Neural Network
### Purpose and Scope

This repo supposed to be a from-scratch deep learning framework implemented using only NumPy. The project demonstrates fundamental neural network concepts including automatic differentiation, modular layer architecture, optimization algorithms, and training utilities without relying on high-level frameworks like PyTorch or TensorFlow.
***
## Project Overview

The numpy-neural-network framework consists of three foundational layers:
| Layer | Purpose | Key Components |
|-------|---------|----------------|
| Core Foundation | Automatic differentiation engine | `Tensor` class with gradient tracking |
| Neural Network Abstraction | Building blocks for model construction | `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Dropout`, `Sequential` |
| Optimization | Parameter update algorithms | `SGD`, `Adam`, `Momentum`, `RMSprop` |
| Utilities | Training support tools | `DataLoader`, `EarlyStopping`, `MSELoss` |
## Core Concepts
#### Tensor
The `Tensor` class provides the computational substrate for the entire framework. It wraps NumPy arrays with gradient tracking capabilities and implements reverse-mode automatic differentiation through stored backward functions.

Key responsibilities:

- Store data as numpy.ndarray
- Track gradients in .grad attribute
- Build computational graphs via operation overloading (__add__, __mul__, __matmul__, __pow__)
- Execute backpropagation through .backward() method
- Provide gradient-free contexts via no_grad() context manager
#### Module
The `Module` base class defines the interface for all neural network components. All layers, activations, and containers inherit from Module.

Key responsibilities:

- Abstract forward() method defining computation
- parameters() method for recursive parameter collection
- zero_grad() for clearing accumulated gradients
- __call__() enabling direct invocation (model(x))

Components include:

- Layers: Linear (fully connected)
- Activations: ReLU, Sigmoid, Tanh
- Regularization: Dropout
- Containers: Sequential, MLP
- Loss Functions like MSELoss and MAELoss/L1Loss
#### Optimizers
`Optimizers` implement algorithms for updating model parameters using computed gradients. All optimizers share a common interface:
| Optimizer | Algorithm | Key Parameters |
|-----------|-----------|----------------|
| SGD | Stochastic Gradient Descent | `lr` (learning rate) |
| Adam | Adaptive Moment Estimation | `lr`, `betas=(0.9, 0.999)`, `eps=1e-8` |
| Momentum | SGD with momentum | `lr`, `momentum=0.9` |
| RMSprop | Root Mean Square Propagation | `lr`, `alpha=0.99`, `eps=1e-8` |
Common interface methods:

- step(): Makes parameter updates
- zero_grad(): Clear gradients
***
## Framework Capabilities

**Model Construction**:
- Feedforward architectures via Sequential container
- Custom layer configurations using Linear layers
- Multiple activation functions: ReLU, Sigmoid, Tanh
- Regularization through Dropout

**Training Features**
- Batch processing via DataLoader
- Four optimization algorithms: SGD, Adam, Momentum, RMSprop
- Validation monitoring
- Early stopping with best weight restoration
- Gradient clipping in optimizers (NaN/Inf protection)

**Loss Functions Compute**
- Mean Squared Error (MSELoss) for regression tasks
***
## Application Example
```python
model = Sequential(
    Linear(X.shape[1], 128),
    ReLU(),
    Dropout(p=0.3),
    Linear(128, 256),
    ReLU(),
    Dropout(p=0.3),
    Linear(256, 1),
)

# Optimizer configuration
optimizer = Adam(model.parameters())

# Training utility setup
early_stopper = EarlyStopping(
    patience=250, 
    min_delta=0.01,
    restore_best_weights=True
)
```
***
### Installation
Step 1: Clone the repository
```
git clone https://github.com/mdu827/numpy-neural-network.git
cd numpy-neural-network
```

Step 2: Install dependencies
```
pip install numpy scikit-learn
```
Running the Example
```
python main.py
```
Expected Output:
```
MSE and MAE before train:
MSE: 1.0
MAE: 0.8

Epoch    0: Avg Loss = 1.234 (batches: 10, total: 12.340)
Epoch  200: Avg Loss = 0.678 (batches: 10, total: 6.780)
Epoch  400: Avg Loss = 0.456 (batches: 10, total: 4.560)
...
Early stopping on epoch: 1850

MSE and MAE after train:
MSE: 0.3
MAE: 0.5
```