# NeuralNetwork
Neural Network for recognizing numerical digits

# MNIST Digit Recognition Neural Network

A simple two-layer neural network implementation from scratch using NumPy for recognizing handwritten digits from the MNIST dataset.

## Note

The training set being used is the MNIST Digit Recognition Neural Network from Kraggle('/kaggle/input/mnist-digit-recognizer/train.csv')

A simple two-layer neural network implementation from scratch using NumPy for recognizing handwritten digits from the MNIST dataset.

## Overview

This implementation creates a neural network with:
- **Input layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 neurons with softmax activation (one for each digit 0-9)

The network uses gradient descent optimization to learn digit classification through backpropagation.

## Dataset

The code expects the MNIST dataset in CSV format with:
- First column: labels (digits 0-9)
- Remaining 784 columns: pixel values (0-255)

Dataset split:
- **Development set**: First 1,000 samples
- **Training set**: Remaining samples

## Architecture Details

### Network Structure
```
Input (784) → Hidden (10, ReLU) → Output (10, Softmax)
```

### Key Components

**Activation Functions:**
- **ReLU**: `max(0, z)` for hidden layer
- **Softmax**: Normalized exponential for output probabilities

**Loss Function:**
- Cross-entropy loss (implemented through one-hot encoding)

**Optimization:**
- Gradient descent with configurable learning rate

## Code Structure

### Core Functions

- `init_params()`: Initialize weights and biases randomly
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Backpropagation to compute gradients
- `update_params()`: Update parameters using gradients
- `gradient_descent()`: Main training loop

### Utility Functions

- `one_hot()`: Convert labels to one-hot encoded vectors
- `get_predictions()`: Extract predicted classes from output
- `get_accuracy()`: Calculate classification accuracy

## Usage

```python
# Train the network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)

# The trained parameters can then be used for predictions
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_test)
predictions = get_predictions(A2)
```

## Hyperparameters

- **Learning rate (alpha)**: 0.10
- **Iterations**: 500
- **Architecture**: 784 → 10 → 10

## Features

- **Data preprocessing**: Pixel normalization (divide by 255)
- **Progress tracking**: Accuracy printed every 10 iterations
- **Vectorized operations**: Efficient NumPy-based computations

## Requirements

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Notes

- Weights are initialized randomly between -0.5 and 0.5
- Input images are normalized to [0,1] range
- The network processes batches of data simultaneously for efficiency
- Training progress is monitored through accuracy on the training set

## Expected Performance

The network should achieve reasonable accuracy on MNIST digit classification, though performance may vary based on random initialization and the relatively simple architecture.

## Future Improvements

- Add validation accuracy tracking
- Implement different initialization schemes (Xavier, He)
- Add regularization techniques
- Experiment with different architectures
- Add early stopping based on validation performance

## Overview

This implementation creates a neural network with:
- **Input layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 neurons with softmax activation (one for each digit 0-9)

The network uses gradient descent optimization to learn digit classification through backpropagation.

## Dataset

The code expects the MNIST dataset in CSV format with:
- First column: labels (digits 0-9)
- Remaining 784 columns: pixel values (0-255)

Dataset split:
- **Development set**: First 1,000 samples
- **Training set**: Remaining samples

## Architecture Details

### Network Structure
```
Input (784) → Hidden (10, ReLU) → Output (10, Softmax)
```

### Key Components

**Activation Functions:**
- **ReLU**: `max(0, z)` for hidden layer
- **Softmax**: Normalized exponential for output probabilities

**Loss Function:**
- Cross-entropy loss (implemented through one-hot encoding)

**Optimization:**
- Gradient descent with configurable learning rate

## Code Structure

### Core Functions

- `init_params()`: Initialize weights and biases randomly
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Backpropagation to compute gradients
- `update_params()`: Update parameters using gradients
- `gradient_descent()`: Main training loop

### Utility Functions

- `one_hot()`: Convert labels to one-hot encoded vectors
- `get_predictions()`: Extract predicted classes from output
- `get_accuracy()`: Calculate classification accuracy

## Usage

```python
# Train the network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)

# The trained parameters can then be used for predictions
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_test)
predictions = get_predictions(A2)
```

## Hyperparameters

- **Learning rate (alpha)**: 0.10
- **Iterations**: 500
- **Architecture**: 784 → 10 → 10

## Features

- **Data preprocessing**: Pixel normalization (divide by 255)
- **Progress tracking**: Accuracy printed every 10 iterations
- **Vectorized operations**: Efficient NumPy-based computations

## Requirements

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Notes

- Weights are initialized randomly between -0.5 and 0.5
- Input images are normalized to [0,1] range
- The network processes batches of data simultaneously for efficiency
- Training progress is monitored through accuracy on the training set

## Expected Performance

The network should achieve reasonable accuracy on MNIST digit classification, though performance may vary based on random initialization and the relatively simple architecture.

## Future Improvements

- Add validation accuracy tracking
- Implement different initialization schemes (Xavier, He)
- Add regularization techniques
- Experiment with different architectures
- Add early stopping based on validation performance
