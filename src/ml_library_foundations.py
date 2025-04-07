# Import statement syntax: 'import [package] as [alias]'
import numpy as np  # Uses common convention of 'np' as alias for numpy


# Class definition syntax: 'class [ClassName]:'
class Neuron:
    # Docstring syntax: Triple quotes for multi-line documentation
    """A single neuron implementation with detailed syntax explanations."""
    
    # PURPOSE: Constructor - Creates a neuron object with weights, bias, and activation function
    # Initializes weights based on strategy and prepares internal state for computations
    def __init__(self, n_inputs, activation='sigmoid', weight_init='random'):
        """Initialize a neuron with weights and bias."""
        # Instance variable assignment syntax: 'self.attribute = value'
        self.n_inputs = n_inputs
        
        # Conditional branching syntax: 'if condition:'
        if weight_init == 'random':
            # NumPy function call syntax: 'np.[function_name]([arguments])'
            # Scalar multiplication syntax: 'array * scalar'
            self.weights = np.random.randn(n_inputs) * 0.01
        # Alternative condition syntax: 'elif condition:'
        elif weight_init == 'xavier':
            # Mathematical function application syntax: 'np.[function](expression)'
            # Division operator syntax: '/'
            self.weights = np.random.randn(n_inputs) * np.sqrt(1 / n_inputs)
        elif weight_init == 'he':
            # Similar mathematical expression with different constant (2 instead of 1)
            self.weights = np.random.randn(n_inputs) * np.sqrt(2 / n_inputs)
        # Default case syntax: 'else:'
        else:
            # Exception raising syntax: 'raise [ExceptionType](message)'
            # F-string syntax: f"text {variable} more text"
            raise ValueError(f"Unknown weight initialization method: {weight_init}")
            
        # Initializing an attribute with a literal value
        self.bias = 0
        
        # Method calling syntax: 'self.[method_name]([arguments])'
        # Underscore prefix indicates a "private" method (convention, not enforced)
        self._set_activation(activation)
        
        # Initializing attributes with None (Python's null value)
        self.last_input = None
        self.last_output = None
        self.last_activation_input = None
    
    # PURPOSE: Helper method to set the activation function and its derivative
    # Maps string activation names to their corresponding implementation methods
    def _set_activation(self, activation):
        """Set the activation function and its derivative."""
        # Equality comparison operator: '=='
        if activation == 'sigmoid':
            # Attribute assignment to method reference (no parentheses)
            # Functions are first-class objects in Python
            self.activation_fn = self._sigmoid
            self.activation_fn_derivative = self._sigmoid_derivative
        elif activation == 'relu':
            # Similar method reference assignments for ReLU
            self.activation_fn = self._relu
            self.activation_fn_derivative = self._relu_derivative
        elif activation == 'tanh':
            # Similar method reference assignments for tanh
            self.activation_fn = self._tanh
            self.activation_fn_derivative = self._tanh_derivative
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    # PURPOSE: Sigmoid activation function - Maps any input to a value between 0 and 1
    # Used to model probability-like outputs or introduce non-linearity
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        # SYNTAX: np.exp(-x) - Element-wise exponential function e^(-x)
        # WHY: np.exp is used for numerical stability instead of direct math.e**x
        # PATTERN: np.exp(expression) - Takes any array-like input
        
        # SYNTAX: np.clip(x, min, max) - Constrains values to specified range
        # WHY: Prevents numerical overflow in exp() with large negative or positive inputs
        # PATTERN: np.clip(array, min_value, max_value) - Always takes three arguments
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # PURPOSE: Calculates the derivative of sigmoid function for backpropagation
    # Used to determine how much to adjust weights during learning
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        # SYNTAX: variable = self.method(args) - Calling instance methods and storing result
        s = self._sigmoid(x)
        # SYNTAX: * for element-wise multiplication with arrays (not matrix multiplication)
        return s * (1 - s)
    
    # PURPOSE: ReLU (Rectified Linear Unit) activation - Returns max(0,x)
    # Popular in deep networks for faster learning and sparsity
    def _relu(self, x):
        """ReLU activation function."""
        # SYNTAX: np.maximum(a, b) - Element-wise maximum between two arrays or values
        # WHY: Vectorized approach is much faster than loops or conditionals
        # PATTERN: np.maximum(array1, array2) - Works with arrays of same shape or array and scalar
        return np.maximum(0, x)
    
    # PURPOSE: Calculates the derivative of ReLU function for backpropagation
    # Returns 1 for positive inputs, 0 otherwise
    def _relu_derivative(self, x):
        """Derivative of ReLU function."""
        # SYNTAX: np.where(condition, value_if_true, value_if_false) - Conditional element selection
        # WHY: Vectorized alternative to list comprehension or loops
        # PATTERN: np.where(boolean_array, if_true_values, if_false_values) - Always 3 arguments
        return np.where(x > 0, 1, 0)
    
    # PURPOSE: Hyperbolic tangent activation - Maps inputs to values between -1 and 1
    # Similar to sigmoid but with outputs centered around 0
    def _tanh(self, x):
        """Tanh activation function."""
        # SYNTAX: np.tanh(x) - Element-wise hyperbolic tangent
        # WHY: NumPy's version is optimized for array operations
        # PATTERN: np.tanh(array) - Acts element-wise on arrays
        return np.tanh(x)
    
    # PURPOSE: Calculates the derivative of tanh function for backpropagation
    # Used to determine gradient direction and magnitude during learning
    def _tanh_derivative(self, x):
        """Derivative of tanh function."""
        # SYNTAX: ** operator - Element-wise exponentiation
        # WHY: a**2 is more readable than a*a or np.power(a,2) for squaring
        # PATTERN: array**power - Raises each element to the specified power
        return 1 - np.tanh(x) ** 2
    
    # PURPOSE: Forward propagation - Computes neuron's output given input values
    # Calculates weighted sum of inputs plus bias, then applies activation function
    def forward(self, inputs):
        """Forward pass: compute weighted sum and apply activation function."""
        # SYNTAX: len(sequence) - Returns number of items in a sequence
        # WHY: Python built-in function, not a method of the object (unlike array.length in other languages)
        # PATTERN: len(object) - Works on any sequence type (list, array, string, etc.)
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}")
        
        self.last_input = inputs
        
        # SYNTAX: np.dot(a, b) - Performs dot product between two arrays
        # WHY: Dot product is optimal way to compute weighted sum (w₁x₁ + w₂x₂ + ... + wₙxₙ)
        # PATTERN: np.dot(array1, array2) - For 1D arrays, returns scalar dot product
        z = np.dot(self.weights, inputs) + self.bias
        self.last_activation_input = z
        
        # SYNTAX: variable = self.attribute(arguments) - Calling a function stored as attribute
        # WHY: We stored function references earlier, now we're calling them with ()
        # PATTERN: self.function_attribute(args) - Parentheses trigger the function call
        output = self.activation_fn(z)
        self.last_output = output
        
        return output
    
    # PURPOSE: Backward propagation - Updates weights based on error gradient
    # Core of the learning process, adjusts weights and bias to minimize error
    def backward(self, error_gradient, learning_rate=0.01):
        """Backward pass: compute gradients and update weights."""
        # SYNTAX: parameter=value in function definition - Default parameter value
        # WHY: Makes parameter optional, uses default if not provided
        # PATTERN: def function(required_param, optional_param=default_value)
        
        # SYNTAX: value is None - Identity comparison with None (not equality ==)
        # WHY: "is" checks for exact same object in memory, more appropriate for None
        # PATTERN: variable is None - Preferred way to check for None
        if self.last_input is None or self.last_activation_input is None:
            raise ValueError("Forward pass must be called before backward pass")
        
        activation_gradient = self.activation_fn_derivative(self.last_activation_input)
        
        delta = error_gradient * activation_gradient
        
        # SYNTAX: array * array - Element-wise multiplication between arrays
        # WHY: NumPy overloads * operator for arrays to do element-wise operations
        # PATTERN: array1 * array2 - Must have compatible shapes for broadcasting
        weight_gradients = delta * self.last_input
        
        bias_gradient = delta
        
        backward_gradient = delta * self.weights
        
        # SYNTAX: variable -= value - Compound assignment operator (shorthand for variable = variable - value)
        # WHY: More concise way to update a variable based on its current value
        # PATTERN: x -= y is equivalent to x = x - y but more efficient
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient
        
        return backward_gradient
    
    # PURPOSE: Creates string representation of neuron for debugging/display
    # Shows neuron's configuration and current state
    def __repr__(self):
        """String representation of the neuron."""
        # SYNTAX: __repr__ - Special method for object representation
        # WHY: Called by repr() and when object is printed in interactive mode
        # PATTERN: def __repr__(self): return f"ClassName(attributes...)"
        return f"Neuron(n_inputs={self.n_inputs}, weights={self.weights}, bias={self.bias})"


class Layer:
    """A layer of neurons in a neural network."""
    
    # PURPOSE: Constructor - Creates a layer containing multiple neurons
    # Sets up a collection of neurons that will process inputs in parallel
    def __init__(self, n_inputs, n_neurons, activation='sigmoid', weight_init='random'):
        """Initialize a layer with multiple neurons."""
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        # SYNTAX: list comprehension - [expression for item in iterable]
        # WHY: More concise and faster than creating an empty list and appending
        # PATTERN: [create_item(args) for _ in range(n)] - Create n items with the same constructor
        self.neurons = [Neuron(n_inputs, activation, weight_init) for _ in range(n_neurons)]
        # Note: _ is a convention for loop variables that aren't used in the body
        
        # Will store the output of the layer
        self.output = None
    
    # PURPOSE: Forward propagation through layer - Processes inputs through all neurons
    # Collects outputs from all neurons into an array for next layer
    def forward(self, inputs):
        """Forward pass through all neurons in the layer."""
        # SYNTAX: list comprehension with method call - [obj.method(arg) for obj in objects]
        # WHY: Apply same operation to all objects in a collection
        # PATTERN: [item.method(args) for item in items] - Call same method on multiple objects
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        
        # SYNTAX: np.array(sequence) - Convert sequence (like list) to NumPy array
        # WHY: NumPy arrays support vectorized operations, more efficient for math
        # PATTERN: np.array(list_or_tuple) - Creates array with same shape as input sequence
        self.output = np.array(outputs)
        
        return self.output
    
    # PURPOSE: Backward propagation through layer - Updates all neurons' weights
    # Propagates error gradients back to previous layer after updating weights
    def backward(self, error_gradients, learning_rate=0.01):
        """Backward pass through all neurons in the layer."""
        # SYNTAX: len(obj) - Get number of items in a collection
        if len(error_gradients) != self.n_neurons:
            # SYNTAX: raising exceptions with context - Provides clear error message about what went wrong
            raise ValueError(f"Expected {self.n_neurons} error gradients, got {len(error_gradients)}")
        
        # SYNTAX: zip(sequence1, sequence2) - Creates iterator of tuples with items from both sequences
        # WHY: Allows parallel iteration over two related sequences
        # PATTERN: for item1, item2 in zip(sequence1, sequence2): - Process pairs of related items
        backward_gradients = [neuron.backward(gradient, learning_rate) 
                             for neuron, gradient in zip(self.neurons, error_gradients)]
        
        # SYNTAX: np.array(list).T - Create NumPy array and transpose it
        # WHY: Converting operation results to array allows mathematical operations
        # PATTERN: np.array(list_of_lists).T - Useful for turning row vectors into column vectors
        backward_gradients = np.array(backward_gradients)
        
        # SYNTAX: np.sum(array, axis=0) - Sum array elements along specified axis
        # WHY: axis=0 means sum along first dimension (vertically in 2D array)
        # PATTERN: np.sum(array, axis=int) - Reduces dimensionality by summing along axis
        if len(backward_gradients.shape) > 1:  # Check if we have a 2D array
            return np.sum(backward_gradients, axis=0)
        else:
            return backward_gradients
    
    # PURPOSE: Creates string representation of layer for debugging/display
    # Shows layer's configuration (input size and neuron count)
    def __repr__(self):
        """String representation of the layer."""
        return f"Layer(n_inputs={self.n_inputs}, n_neurons={self.n_neurons})"


class MLP:
    """Multi-Layer Perceptron (a basic neural network)."""
    
    # PURPOSE: Constructor - Creates a complete neural network with multiple layers
    # Establishes network architecture by connecting layers of appropriate sizes
    def __init__(self, layer_sizes, activation='sigmoid', weight_init='random'):
        """Initialize a multi-layer neural network.
        
        Args:
            layer_sizes: List of integers representing [input_size, hidden_layer1_size, ..., output_size]
            activation: Activation function to use ('sigmoid', 'relu', 'tanh')
            weight_init: Weight initialization strategy ('random', 'xavier', 'he')
        """
        # SYNTAX: assert condition, message - Runtime verification of condition
        # WHY: Catch configuration errors early with descriptive message
        # PATTERN: assert len(x) > value, "message" - Validate input before processing
        assert len(layer_sizes) >= 2, "Need at least input and output layer sizes"
        
        self.layer_sizes = layer_sizes
        
        # SYNTAX: range(start, stop) - Generate sequence of integers from start to stop-1
        # WHY: Used for iterating a specific number of times
        # PATTERN: range(len(sequence)-1) - Iterate up to second-to-last element
        # SYNTAX: list comprehension with values from multiple lists - Complex object creation in one line
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activation, weight_init)
                      for i in range(len(layer_sizes) - 1)]
        
        # Store layer outputs for backpropagation
        self.layer_outputs = []
        
    # PURPOSE: Forward propagation through entire network - Processes inputs to get predictions
    # Sequentially passes data through all layers and returns final output
    def forward(self, inputs):
        """Forward pass through the entire network."""
        # SYNTAX: input validation with type check and shape check
        # WHY: Early validation prevents cryptic errors later
        # PATTERN: if not isinstance(x, type) or x.shape != expected: raise ValueError()
        if not isinstance(inputs, np.ndarray):
            # SYNTAX: Converting list to numpy array if needed
            inputs = np.array(inputs)
            
        if inputs.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {inputs.shape[0]}")
        
        # Reset layer outputs
        # SYNTAX: self.attribute = [] - Reinitialize list attribute
        self.layer_outputs = [inputs]  # Include input as first layer output
        
        # Feed forward through each layer
        current_inputs = inputs
        
        # SYNTAX: for item in sequence: - Standard Python iteration
        # WHY: Process each item in a sequence one at a time
        # PATTERN: for obj in self.objects: result = obj.method(args)
        for layer in self.layers:
            current_outputs = layer.forward(current_inputs)
            self.layer_outputs.append(current_outputs)
            current_inputs = current_outputs
        
        # Return the output of the last layer
        return self.layer_outputs[-1]  # SYNTAX: sequence[-1] - Access last element of sequence
    
    # PURPOSE: Backward propagation through entire network - Updates all weights
    # Propagates error gradients backwards through all layers to update weights
    def backward(self, error_gradients, learning_rate=0.01):
        """Backward pass through the entire network."""
        if not isinstance(error_gradients, np.ndarray):
            error_gradients = np.array(error_gradients)
            
        n_layers = len(self.layers)
        
        # Check that we have the expected number of error gradients
        if error_gradients.shape[0] != self.layer_sizes[-1]:
            raise ValueError(f"Expected {self.layer_sizes[-1]} error gradients, got {error_gradients.shape[0]}")
        
        # Backpropagate through each layer in reverse order
        # SYNTAX: reversed(sequence) - Iterate through sequence in reverse order
        # WHY: For backpropagation, we need to go from last layer to first
        # PATTERN: for item in reversed(sequence): - Process items in reverse order
        current_gradients = error_gradients
        
        # SYNTAX: enumerate(sequence) - Get (index, item) pairs when iterating
        # WHY: When both the index and the value are needed in the loop
        # PATTERN: for i, item in enumerate(sequence): - Access index and value together
        for i, layer in enumerate(reversed(self.layers)):
            # Get the corresponding layer inputs from our saved outputs
            layer_input_idx = n_layers - i - 1  # Index into layer_outputs
            
            current_gradients = layer.backward(current_gradients, learning_rate)
    
    # PURPOSE: Training method - Repeatedly updates weights to minimize prediction errors
    # Core learning algorithm that handles both forward and backward passes
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the neural network using gradient descent.
        
        Args:
            X: Training inputs, shape (n_samples, n_features)
            y: Target outputs, shape (n_samples, n_outputs)
            epochs: Number of training iterations
            learning_rate: Step size for gradient descent
            verbose: Whether to print progress
        """
        X = np.array(X)
        y = np.array(y)
        
        # SYNTAX: assert condition, message - Check preconditions
        assert X.shape[1] == self.layer_sizes[0], f"Input features {X.shape[1]} don't match network input size {self.layer_sizes[0]}"
        assert y.shape[1] == self.layer_sizes[-1], f"Output features {y.shape[1]} don't match network output size {self.layer_sizes[-1]}"
        
        # SYNTAX: range with step - range(start, stop, step)
        # WHY: Process every nth item or skip items
        # PATTERN: range(0, max, interval) - Generate sequence with specific interval
        for epoch in range(0, epochs):
            total_error = 0
            
            # SYNTAX: zip for parallel iteration over multiple sequences
            # WHY: Process corresponding elements from multiple sequences together
            # PATTERN: for item1, item2 in zip(seq1, seq2): - Process pairs of items
            for x_sample, y_target in zip(X, y):
                # Forward pass
                y_pred = self.forward(x_sample)
                
                # Compute error (for reporting)
                # SYNTAX: np.mean(array) - Compute mean of all elements
                # WHY: Reduce array to single scalar value
                # PATTERN: np.mean(function_of_array) - Common reduction operation
                sample_error = np.mean((y_pred - y_target) ** 2)
                total_error += sample_error
                
                # Compute gradients for output layer
                # SYNTAX: 2 * (pred - target) - Derivative of squared error
                # WHY: Chain rule application: d(error)/d(pred) = d((pred-target)²)/d(pred) = 2(pred-target)
                # PATTERN: factor * (array1 - array2) - Common gradient calculation pattern
                output_gradients = 2 * (y_pred - y_target)
                
                # Backward pass
                self.backward(output_gradients, learning_rate)
            
            # Print progress
            # SYNTAX: if condition and verbose: - Conditional log/print statement
            # WHY: Only print if verbose flag is True (common pattern for debugging/logging)
            # PATTERN: if condition and verbose_flag: print(message)
            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}, Error: {total_error:.6f}")
                
        return total_error
    
    # PURPOSE: Prediction method - Makes predictions for multiple input samples
    # Convenience method that runs forward propagation on multiple inputs
    def predict(self, X):
        """Make predictions for multiple samples."""
        # SYNTAX: list comprehension with method call
        # WHY: Apply same operation to all items in collection
        # PATTERN: [self.method(item) for item in items] - Process all items with same method
        X = np.array(X)
        return np.array([self.forward(x) for x in X])
    
    # PURPOSE: Creates string representation of network for debugging/display
    # Shows network's layer structure
    def __repr__(self):
        """String representation of the network."""
        return f"MLP(layers={self.layer_sizes})"


# Example usage
if __name__ == "__main__":
    # PURPOSE: Demo section - Shows how to use the neural network implementation
    # Creates and trains a network to solve the XOR problem as a demonstration
    
    # Create a neural network with 2 inputs, 3 hidden neurons, and 1 output
    # SYNTAX: Class instantiation with keyword arguments
    # WHY: Makes configuration explicit
    # PATTERN: Object = ClassName(param1=value1, param2=value2)
    mlp = MLP(layer_sizes=[2, 3, 1], activation='sigmoid', weight_init='xavier')
    
    # XOR problem inputs and outputs
    # SYNTAX: np.array creation with nested lists
    # WHY: Create 2D array (matrix) from list of lists
    # PATTERN: np.array([[row1], [row2], ...]) - Each inner list is a row
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Train the network
    mlp.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Test the network
    predictions = mlp.predict(X)
    print("\nPredictions:")
    
    # SYNTAX: enumerate() for getting index and value in loop
    # WHY: Process items with their positions
    # PATTERN: for i, item in enumerate(sequence): - Access position and value
    for i, (x_input, y_target, y_pred) in enumerate(zip(X, y, predictions)):
        # SYNTAX: String formatting with f-strings and formatting specifiers
        # WHY: Control precision of floating point numbers in output
        # PATTERN: f"{value:.precision}" - Format with specific decimal precision 
        print(f"Input: {x_input}, Target: {y_target[0]}, Prediction: {y_pred[0]:.4f}")