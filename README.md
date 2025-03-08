# Neural Network Classifier with R-Functions

## Overview
This project implements a neural network-based classifier using R-functions to combine outputs from multiple decision boundaries. The classifier is trained on a synthetic dataset generated using `sklearn.datasets.make_blobs`. The primary goal is to demonstrate how R-functions can be used to merge multiple decision boundaries in a classification task.

## Example
[example](example.png)

## Features
- Uses two linear classifiers combined using an R-function.
- Applies a sigmoid activation function for classification.
- Implements gradient-based weight updates.
- Visualizes decision boundaries and error progression during training.
- Includes an animated visualization of the classification process.

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install numpy matplotlib scikit-learn
```

## How It Works
1. **Data Generation**: The dataset is created with three cluster centers, and labels are adjusted to fit a binary classification task.
2. **Neural Network Model**: The model consists of two linear classifiers, each with its own weight vector. The outputs are combined using an R-function.
3. **Activation Function**: A sigmoid function is used for classification.
4. **Training**:
   - Each sample is evaluated using the two classifiers.
   - The R-function is used to combine their outputs.
   - Weights are updated using gradient descent with a custom influence factor.
   - The error is tracked over multiple epochs.
5. **Visualization**:
   - The decision boundary evolution is shown during training.
   - An animation visualizes the classifier’s learning process.

## Usage
Run the Python script to train the classifier and visualize the training process:
```sh
python pwlc.py
```

## Key Functions
- `sigmoid(x)`: Computes the sigmoid activation.
- `sigmoid_derivative(x)`: Computes the derivative of the sigmoid function.
- `r_function(f1, f2)`: Merges two classifier outputs.
- `influence_factor(net1, net2)`: Computes a factor used in weight updates.
- `decision_boundary(weights, x)`: Calculates decision boundary based on weights.
- `plot_decision_boundary(ax, weights1, weights2)`: Plots the classification regions.

## Animation
The training process is visualized using `matplotlib.animation.FuncAnimation`, showing how the decision boundaries evolve over time.
This implementation was developed as an educational demonstration of neural network classification using R-functions.

ç
