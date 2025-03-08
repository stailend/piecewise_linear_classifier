import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation

# Define cluster centers
cluster_centers = [[-1, 0], [0, 0], [1, 0]]

# Generate synthetic dataset with three clusters
X, true_labels = make_blobs(n_samples=100, centers=cluster_centers, cluster_std=0.1, random_state=2)

# Adjust labels: convert -1 labels to 1
true_labels = true_labels - 1
true_labels = [1 if i == -1 else int(i) for i in true_labels]
y = true_labels

# Add bias term (column of ones) to feature matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize two sets of weights randomly
weights_1 = np.random.rand(3)
weights_2 = np.random.rand(3)

# Learning rate and number of epochs
learning_rate = 0.02
num_epochs = 50

# Store weight history and errors for visualization
weights_1_history = []
weights_2_history = []
errors_list = []

# Define decision boundary equation
def decision_boundary(w, x):
    return -(w[1] * x + w[0]) / w[2]

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Auxiliary function to adjust learning influence
def influence_factor(net1, net2):
    return 1 - net2 / np.sqrt(net1**2 + net2**2)

# Combination function for two decision boundaries
def combined_decision(net1, net2):
    return net1 + net2 + np.sqrt(net1**2 + net2**2)

# Training loop
for epoch in range(num_epochs):
    total_error = 0
    for i in range(X.shape[0]):
        # Compute net inputs for both perceptrons
        net_1 = np.dot(weights_1, X[i])
        net_2 = np.dot(weights_2, X[i])
        
        # Compute combined output
        net_combined = combined_decision(net_1, net_2)
        prediction = np.round(sigmoid(net_combined))
        error = y[i] - prediction
        
        # Update weights using gradient descent
        weights_1 += learning_rate * error * X[i] * sigmoid_derivative(net_1) * influence_factor(net_1, net_2)
        weights_2 += learning_rate * error * X[i] * sigmoid_derivative(net_2) * influence_factor(net_2, net_1)
        
        total_error += abs(error)
    
    # Store history for visualization
    weights_1_history.append(weights_1.copy())
    weights_2_history.append(weights_2.copy())
    errors_list.append(total_error)

# Set up plot for animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_decision_boundary(ax, weights1, weights2):
    x_min, x_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    y_min, y_max = X[:, 2].min() - 2.5, X[:, 2].max() + 2.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    
    net1 = np.dot(grid, weights1)
    net2 = np.dot(grid, weights2)
    net = combined_decision(net1, net2)
    Z = np.round(sigmoid(net))
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#ccccff", "#ffcccc"], alpha=0.6)

# Initialize plot elements
scatter = ax1.scatter(X[:, 1], X[:, 2], c=true_labels, cmap="bwr", edgecolor="k", alpha=0.7)
ax1.set_title("Classifier Training")
ax2.set_title("Error Evolution")
ax2.set_xlim(0, num_epochs)
ax2.set_ylim(0, len(true_labels))
error_line, = ax2.plot([], [], 'r-', lw=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Error")

def init_animation():
    ax1.set_xlim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax1.set_ylim(X[:, 2].min() - 0.5, X[:, 2].max() + 0.5)
    error_line.set_data([], [])
    return error_line,

def update_animation(frame):
    ax1.clear()
    plot_decision_boundary(ax1, weights_1_history[frame], weights_2_history[frame])
    ax1.scatter(X[:, 1], X[:, 2], c=true_labels, cmap="bwr", edgecolor="k", alpha=0.7)
    ax1.set_title("Classifier Training")
    ax1.set_xlim(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5)
    ax1.set_ylim(X[:, 2].min() - 2.5, X[:, 2].max() + 2.5)
    
    x_vals = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    y_vals1 = decision_boundary(weights_1_history[frame], x_vals)
    y_vals2 = decision_boundary(weights_2_history[frame], x_vals)
    ax1.plot(x_vals, y_vals1, 'b--', label='Boundary 1')
    ax1.plot(x_vals, y_vals2, 'g--', label='Boundary 2')
    ax1.legend()
    ax1.grid()
    
    error_line.set_data(range(frame + 1), errors_list[:frame + 1])
    ax2.set_xlim(0, num_epochs)
    ax2.set_ylim(0, max(errors_list) * 1.1)
    
    return error_line,

# Create animation
animation = FuncAnimation(fig, update_animation, frames=len(weights_1_history), blit=False, interval=100)
plt.show()
