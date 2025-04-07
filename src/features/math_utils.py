"""
Neural Network Activation Functions and Propagation Reference
============================================================

This reference document contains formulas and explanations for:
1. Common activation functions and their derivatives
2. Forward propagation computation
3. Backward propagation (gradient descent) computation

Each section includes mathematical formulas, properties, and implementation notes.
"""

import numpy as np
import matplotlib.pyplot as plt


# ========== ACTIVATION FUNCTIONS ==========

def plot_activation_functions():
    """Plot common activation functions and their derivatives for visualization."""
    x = np.linspace(-5, 5, 1000)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Activation Functions and Their Derivatives', fontsize=16)
    
    # Sigmoid
    axs[0, 0].plot(x, sigmoid(x))
    axs[0, 0].set_title('Sigmoid')
    axs[0, 0].grid(True)
    axs[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axs[0, 1].plot(x, sigmoid_derivative(x))
    axs[0, 1].set_title('Sigmoid Derivative')
    axs[0, 1].grid(True)
    axs[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # ReLU
    axs[1, 0].plot(x, relu(x))
    axs[1, 0].set_title('ReLU')
    axs[1, 0].grid(True)
    axs[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axs[1, 1].plot(x, relu_derivative(x))
    axs[1, 1].set_title('ReLU Derivative')
    axs[1, 1].grid(True)
    axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tanh
    axs[2, 0].plot(x, tanh(x))
    axs[2, 0].set_title('Tanh')
    axs[2, 0].grid(True)
    axs[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[2, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axs[2, 1].plot(x, tanh_derivative(x))
    axs[2, 1].set_title('Tanh Derivative')
    axs[2, 1].grid(True)
    axs[2, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[2, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# ===== 1. SIGMOID ACTIVATION =====

def sigmoid(x):
    """
    Sigmoid Activation Function: σ(x) = 1 / (1 + e^(-x))
    
    Properties:
    - Output range: (0, 1)
    - Smooth and differentiable
    - Historically popular but less used in hidden layers now
    - Suffers from vanishing gradient problem for deep networks
    
    Use cases:
    - Binary classification (output layer)
    - When output needs to be interpreted as probability
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """
    Derivative of Sigmoid: σ'(x) = σ(x) * (1 - σ(x))
    
    Properties:
    - Maximum value of 0.25 at x=0
    - Approaches 0 as |x| increases
    - Contributes to vanishing gradient problem
    """
    s = sigmoid(x)
    return s * (1 - s)


# ===== 2. RELU ACTIVATION =====

def relu(x):
    """
    Rectified Linear Unit: ReLU(x) = max(0, x)
    
    Properties:
    - Output range: [0, ∞)
    - Not differentiable at x=0
    - Non-saturating for positive values
    - Solves vanishing gradient problem for positive inputs
    - Suffers from "dying ReLU" problem
    
    Use cases:
    - Default choice for hidden layers in many networks
    - Convolutional Neural Networks (CNNs)
    - Deep networks
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU:
        ReLU'(x) = 1 if x > 0
        ReLU'(x) = 0 if x ≤ 0
    
    Properties:
    - Binary output (0 or 1)
    - Gradient doesn't vanish for positive inputs
    - Zero gradient for negative inputs (can cause neurons to "die")
    """
    return np.where(x > 0, 1, 0)


# ===== 3. TANH ACTIVATION =====

def tanh(x):
    """
    Hyperbolic Tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Properties:
    - Output range: (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Smooth and differentiable
    - Still suffers from vanishing gradient for large |x|
    
    Use cases:
    - When zero-centered output is needed
    - Recurrent Neural Networks (RNNs)
    - When sigmoid is too restrictive
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of tanh: tanh'(x) = 1 - tanh²(x)
    
    Properties:
    - Maximum value of 1 at x=0
    - Approaches 0 as |x| increases
    - Still contributes to vanishing gradient but less severe than sigmoid
    """
    return 1 - np.tanh(x)**2


# ===== 4. LEAKY RELU =====

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: LReLU(x) = max(αx, x) where α is a small positive constant
    
    Properties:
    - Output range: (-∞, ∞)
    - Allows small negative values (doesn't "die")
    - Differentiable everywhere except at x=0
    - α typically set to 0.01
    
    Use cases:
    - When dying ReLU is a concern
    - Alternative to ReLU in deep networks
    """
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU:
        LReLU'(x) = 1 if x > 0
        LReLU'(x) = α if x ≤ 0
    
    Properties:
    - Output is either 1 or α
    - No zero gradients (mitigates dying neurons)
    """
    return np.where(x > 0, 1, alpha)


# ===== 5. SOFTMAX ACTIVATION =====

def softmax(x):
    """
    Softmax: softmax(x)_i = e^(x_i) / Σ(e^(x_j)) for all j
    
    Properties:
    - Outputs sum to 1.0 (probability distribution)
    - Emphasizes the largest values
    - Typically applied to entire vector/array at once
    - Not element-wise like other activations
    
    Use cases:
    - Multi-class classification (output layer)
    - When output needs to be probability distribution
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# ========== PROPAGATION FORMULAS ==========

def forward_propagation_formulas():
    """
    Forward Propagation Formulas
    ============================
    
    For a layer with weights W, bias b, input x, and activation function f:
    
    1. Linear transformation: z = W·x + b
        - W is the weight matrix
        - x is the input vector
        - b is the bias vector
    
    2. Activation: a = f(z)
        - f is the activation function
        - a is the output of the layer
    
    For a network with L layers:
    - a⁰ = x (input)
    - z^l = W^l·a^(l-1) + b^l for l ∈ {1, 2, ..., L}
    - a^l = f^l(z^l) for l ∈ {1, 2, ..., L}
    - a^L = output of the network
    """
    pass


def backward_propagation_formulas():
    """
    Backward Propagation Formulas
    =============================
    
    Using the chain rule of calculus to compute gradients for
    parameters (weights W and biases b) with respect to the loss function.
    
    For the output layer L:
    1. Error (delta): δ^L = ∇_a L ⊙ f'(z^L)
        - ∇_a L is the gradient of the loss with respect to the output
        - f'(z^L) is the derivative of the activation function
        - ⊙ represents element-wise multiplication
    
    For hidden layers l ∈ {L-1, L-2, ..., 1}:
    2. Error (delta): δ^l = ((W^(l+1))^T · δ^(l+1)) ⊙ f'(z^l)
        - (W^(l+1))^T is the transpose of the weight matrix of the next layer
        - δ^(l+1) is the error of the next layer
    
    For all layers l ∈ {1, 2, ..., L}:
    3. Weight gradients: ∇_W^l L = δ^l · (a^(l-1))^T
    4. Bias gradients: ∇_b^l L = δ^l
    
    Parameter update using gradient descent:
    - W^l = W^l - α·∇_W^l L
    - b^l = b^l - α·∇_b^l L
    
    Where α is the learning rate.
    """
    pass


def common_loss_functions():
    """
    Common Loss Functions
    ====================
    
    1. Mean Squared Error (MSE):
       L(y, ŷ) = (1/n) Σ (y_i - ŷ_i)²
       Derivative: dL/dŷ_i = (2/n) (ŷ_i - y_i)
       
       - Used for regression problems
       - Heavily penalizes large errors
    
    2. Binary Cross-Entropy:
       L(y, ŷ) = -(1/n) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
       Derivative: dL/dŷ_i = -(y_i/ŷ_i - (1-y_i)/(1-ŷ_i))
       
       - Used for binary classification
       - Outputs interpreted as probabilities
    
    3. Categorical Cross-Entropy:
       L(y, ŷ) = -(1/n) Σ Σ y_ij log(ŷ_ij)
       Derivative: dL/dŷ_ij = -y_ij/ŷ_ij
       
       - Used for multi-class classification
       - Works with one-hot encoded targets
    """
    pass


if __name__ == "__main__":
    print("Neural Network Activation Functions and Propagation Reference")
    print("------------------------------------------------------------")
    print("This module provides reference information about activation functions")
    print("and propagation formulas used in neural networks.")
    print("\nTo visualize activation functions, call plot_activation_functions()")
    
    # Uncomment to visualize the activation functions:
    # plot_activation_functions()
