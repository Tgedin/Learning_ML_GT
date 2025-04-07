import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicAttention:
    """A basic attention mechanism implementation with detailed explanations."""
    
    # PURPOSE: Constructor - Creates an attention module for sequence processing
    # Initializes parameters for computing alignment scores between queries and keys
    def __init__(self, hidden_dim):
        """Initialize attention mechanism with specified hidden dimension."""
        # Instance variable assignment for dimension tracking
        self.hidden_dim = hidden_dim
        
        # SYNTAX: np.random.randn(dim) / math.sqrt(dim) - Initialize with scaled normal distribution
        # WHY: Scaling by sqrt(dim) helps maintain proper variance in the forward pass
        # PATTERN: Common weight initialization pattern for stable training
        self.query_weights = np.random.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
        self.key_weights = np.random.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
        self.value_weights = np.random.randn(hidden_dim, hidden_dim) / math.sqrt(hidden_dim)
        
        # History tracking for learning and debugging
        self.last_attention_weights = None
        self.last_context_vector = None
    
    # PURPOSE: Score computation - Calculates alignment between query and key
    # Measures how relevant each element in the sequence is to the current focus
    def _compute_scores(self, query, keys):
        """
        Compute attention scores between query and keys.
        
        Args:
            query: Vector representing the current focus point, shape (hidden_dim,)
            keys: Matrix of sequence elements to attend to, shape (seq_len, hidden_dim)
            
        Returns:
            scores: Raw alignment scores, shape (seq_len,)
        """
        # SYNTAX: np.dot(a, b) - Matrix multiplication between two arrays
        # WHY: Transforms query into the appropriate attention space
        # PATTERN: query transformation is a crucial first step in attention calculation
        transformed_query = np.dot(query, self.query_weights)  # shape: (hidden_dim,)
        
        # SYNTAX: np.dot(keys, weights) - Batch matrix multiplication
        # WHY: Transforms all keys at once for efficiency
        # PATTERN: Vectorized operation to avoid explicit loops
        transformed_keys = np.dot(keys, self.key_weights)  # shape: (seq_len, hidden_dim)
        
        # SYNTAX: scores = np.dot(key, query) - Dot product as similarity measure
        # WHY: Dot product captures how aligned two vectors are in direction and magnitude
        # PATTERN: This is the core of the attention mechanism - measuring relevance
        scores = np.dot(transformed_keys, transformed_query)  # shape: (seq_len,)
        
        return scores
    
    # PURPOSE: Attention weights - Convert raw scores to probability distribution
    # Creates a weighting scheme that sums to 1 across the sequence
    def _get_attention_weights(self, scores):
        """
        Convert raw scores to probability distribution using softmax.
        
        Args:
            scores: Raw alignment scores, shape (seq_len,)
            
        Returns:
            attention_weights: Normalized attention distribution, shape (seq_len,)
        """
        # SYNTAX: scores - np.max(scores) - Numerical stability technique
        # WHY: Prevents overflow in exp() by subtracting maximum score
        # PATTERN: Standard practice when implementing softmax
        shifted_scores = scores - np.max(scores)
        
        # SYNTAX: np.exp(x) - Element-wise exponential function
        # WHY: Converts arbitrary values to strictly positive numbers
        # PATTERN: First step of softmax calculation
        exp_scores = np.exp(shifted_scores)
        
        # SYNTAX: exp_scores / np.sum(exp_scores) - Normalization step
        # WHY: Creates a proper probability distribution that sums to 1
        # PATTERN: Second step of softmax calculation
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Store for later analysis (common in attention visualization)
        self.last_attention_weights = attention_weights
        
        return attention_weights
    
    # PURPOSE: Context vector - Creates a weighted sum of values based on attention
    # Produces a single vector that captures relevant information from the sequence
    def _create_context_vector(self, values, attention_weights):
        """
        Create context vector as weighted sum of values.
        
        Args:
            values: Value vectors to combine, shape (seq_len, hidden_dim)
            attention_weights: Weights determining importance, shape (seq_len,)
            
        Returns:
            context_vector: Attention-weighted representation, shape (hidden_dim,)
        """
        # SYNTAX: np.dot(values.T, weights) - Weighted sum using matrix multiplication
        # WHY: Efficiently combines values based on their importance
        # PATTERN: Core operation that creates the "focused" representation
        
        # First transform values through the value projection
        transformed_values = np.dot(values, self.value_weights)  # (seq_len, hidden_dim)
        
        # SYNTAX: weights[:, np.newaxis] - Reshape vector for broadcasting
        # WHY: Allows element-wise multiplication with 2D array
        # PATTERN: Common reshaping technique for working with attention weights
        weighted_values = transformed_values * attention_weights[:, np.newaxis]  # (seq_len, hidden_dim)
        
        # SYNTAX: np.sum(weighted_values, axis=0) - Sum along sequence dimension
        # WHY: Combines all weighted values into a single context vector
        # PATTERN: Final step in attention mechanism that produces the output
        context_vector = np.sum(weighted_values, axis=0)  # (hidden_dim,)
        
        # Store for visualization or analysis
        self.last_context_vector = context_vector
        
        return context_vector
    
    # PURPOSE: Forward pass - Executes complete attention mechanism
    # Combines all steps: score computation, weight normalization, and context creation
    def forward(self, query, keys, values):
        """
        Perform full attention operation.
        
        Args:
            query: Current focus point, shape (hidden_dim,)
            keys: Sequence elements to match against, shape (seq_len, hidden_dim)
            values: Information to extract from sequence, shape (seq_len, hidden_dim)
            
        Returns:
            context_vector: Attention-weighted information, shape (hidden_dim,)
        """
        # Step 1: Compute raw alignment scores
        scores = self._compute_scores(query, keys)
        
        # Step 2: Convert scores to attention weights (probabilities)
        attention_weights = self._get_attention_weights(scores)
        
        # Step 3: Create context vector as weighted combination of values
        context_vector = self._create_context_vector(values, attention_weights)
        
        return context_vector, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism as introduced in "Attention Is All You Need" paper.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    
    # PURPOSE: Constructor - Sets up multiple parallel attention heads
    # Allows the model to focus on different aspects of the sequence simultaneously
    def __init__(self, model_dim, num_heads):
        """
        Initialize multi-head attention with model dimension and number of attention heads.
        
        Args:
            model_dim: Hidden dimension of the model
            num_heads: Number of parallel attention mechanisms
        """
        # Validate that model dimension is divisible by number of heads
        # SYNTAX: assert condition, message - Runtime verification
        # WHY: Ensures clean division of embedding space across heads
        # PATTERN: Common validation technique for model configuration
        assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # SYNTAX: np.random.randn(*dims) - Create random array with specified dimensions
        # WHY: Initialize weights with Gaussian distribution for training stability
        # PATTERN: Common initialization pattern for neural network weights
        scale = np.sqrt(self.head_dim)
        
        # Combined projection matrices for efficiency
        # Rather than creating separate instances of BasicAttention
        self.wq = np.random.randn(model_dim, model_dim) / scale
        self.wk = np.random.randn(model_dim, model_dim) / scale
        self.wv = np.random.randn(model_dim, model_dim) / scale
        self.wo = np.random.randn(model_dim, model_dim) / scale
        
        # Keep track of attention weights for visualization
        self.last_attention_weights = None
    
    # PURPOSE: Split heads - Reshapes tensors for parallel attention computation
    # Divides embedding dimension across attention heads for specialized focus
    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, model_dim)
            batch_size: Number of sequences in batch
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        # SYNTAX: x.reshape(dims) - Reshape array to new dimensions
        # WHY: Reorganizes the embedding dimension for parallel processing by heads
        # PATTERN: Critical transformation for multi-head architecture
        
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # SYNTAX: np.transpose(x, axes) - Rearrange dimensions
        # WHY: Brings head dimension earlier for easier per-head operations
        # PATTERN: Common reshaping operation in attention mechanisms
        return np.transpose(x, (0, 2, 1, 3))  # (batch, heads, seq_len, head_dim)
    
    # PURPOSE: Combine heads - Merges parallel attention results
    # Integrates information from multiple attention heads back into a single representation
    def _combine_heads(self, x, batch_size):
        """
        Combine heads back into original shape.
        
        Args:
            x: Multi-head tensor of shape (batch_size, num_heads, seq_len, head_dim)
            batch_size: Number of sequences in batch
            
        Returns:
            Combined tensor of shape (batch_size, seq_len, model_dim)
        """
        # SYNTAX: np.transpose(x, axes) - Rearrange dimensions
        # WHY: Prepares tensor for recombination of head dimensions
        # PATTERN: Inverse operation of _split_heads
        x = np.transpose(x, (0, 2, 1, 3))  # (batch, seq_len, heads, head_dim)
        
        seq_len = x.shape[1]
        
        # SYNTAX: x.reshape(dims) - Change shape without changing data
        # WHY: Recombines separate head representations into single embedding
        # PATTERN: Final step in multi-head processing
        return x.reshape(batch_size, seq_len, self.model_dim)
    
    # PURPOSE: Scaled dot-product attention - Core attention calculation
    # Computes attention weights and applies them to values across all heads
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention for all heads at once.
        
        Args:
            q: Query tensor (batch_size, num_heads, q_len, head_dim)
            k: Key tensor (batch_size, num_heads, k_len, head_dim)
            v: Value tensor (batch_size, num_heads, v_len, head_dim)
            mask: Optional mask for attention weights (batch_size, 1, 1, k_len)
            
        Returns:
            output: Attention output (batch_size, num_heads, q_len, head_dim)
            attention_weights: Attention distribution (batch_size, num_heads, q_len, k_len)
        """
        # SYNTAX: np.matmul(a, b.transpose) - Batch matrix multiplication
        # WHY: Computes similarity between queries and keys
        # PATTERN: Core operation in attention that creates attention matrix
        
        # Transpose k for matrix multiplication with q
        # Need to move head_dim to the end for matmul
        k_transposed = np.transpose(k, (0, 1, 3, 2))  # (batch, heads, head_dim, k_len)
        
        # Compute attention scores
        # SYNTAX: matmul followed by scaling - Matches paper formulation
        # WHY: Scaling prevents dot products from growing too large with increasing dimension
        # PATTERN: Critical scaling that gives the mechanism its name
        scores = np.matmul(q, k_transposed) / np.sqrt(self.head_dim)  # (batch, heads, q_len, k_len)
        
        # Apply mask if provided (e.g., for causal/padding attention)
        if mask is not None:
            # SYNTAX: x + very_negative_number - Masking technique
            # WHY: Makes masked positions have nearly zero weight after softmax
            # PATTERN: Standard approach for preventing attention to certain positions
            scores = scores + (mask * -1e9)
        
        # Apply softmax to get attention weights
        # SYNTAX: softmax on last dimension - Creates probability distribution
        # WHY: Normalizes scores to sum to 1 across key dimension
        # PATTERN: Standard softmax implementation for attention
        
        # Compute softmax along last axis (key length dimension)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Store for visualization
        self.last_attention_weights = attention_weights
        
        # Apply attention weights to values
        # SYNTAX: np.matmul(weights, values) - Weighted combination
        # WHY: Creates context vectors based on attention distribution
        # PATTERN: Final step that produces attention-weighted representation
        output = np.matmul(attention_weights, v)  # (batch, heads, q_len, head_dim)
        
        return output, attention_weights
    
    # PURPOSE: Forward pass - Complete multi-head attention operation
    # Projects inputs, computes parallel attention, and combines results
    def forward(self, q, k, v, mask=None):
        """
        Perform full multi-head attention operation.
        
        Args:
            q: Query tensor (batch_size, q_len, model_dim)
            k: Key tensor (batch_size, k_len, model_dim)
            v: Value tensor (batch_size, v_len, model_dim)
            mask: Optional mask tensor
            
        Returns:
            output: Final attention result (batch_size, q_len, model_dim)
        """
        batch_size = q.shape[0]
        
        # Linear projections
        # SYNTAX: np.matmul(x, weights) - Linear transformation
        # WHY: Projects inputs into subspaces for different attention heads
        # PATTERN: Initial transformation in attention pipeline
        q_projected = np.matmul(q, self.wq)  # (batch, q_len, model_dim)
        k_projected = np.matmul(k, self.wk)  # (batch, k_len, model_dim)
        v_projected = np.matmul(v, self.wv)  # (batch, v_len, model_dim)
        
        # Split into multiple heads
        q_heads = self._split_heads(q_projected, batch_size)  # (batch, heads, q_len, head_dim)
        k_heads = self._split_heads(k_projected, batch_size)  # (batch, heads, k_len, head_dim)
        v_heads = self._split_heads(v_projected, batch_size)  # (batch, heads, v_len, head_dim)
        
        # Apply scaled dot-product attention to each head
        attn_output, attention_weights = self._scaled_dot_product_attention(
            q_heads, k_heads, v_heads, mask)  # (batch, heads, q_len, head_dim)
        
        # Combine the heads back
        combined_output = self._combine_heads(attn_output, batch_size)  # (batch, q_len, model_dim)
        
        # Final linear projection
        # SYNTAX: np.matmul(x, weights) - Final transformation
        # WHY: Integrates information from all heads into output space
        # PATTERN: Standard practice to project combined attention back to model dimension
        output = np.matmul(combined_output, self.wo)  # (batch, q_len, model_dim)
        
        return output, attention_weights


class SelfAttention:
    """
    Self-attention layer where queries, keys, and values all come from the same source.
    This is the building block of transformer architectures.
    """
    
    # PURPOSE: Constructor - Creates a self-attention module
    # Special case of attention where query, key, and value come from same sequence
    def __init__(self, model_dim, num_heads=8):
        """Initialize self-attention with model dimension and number of heads."""
        self.multi_head = MultiHeadAttention(model_dim, num_heads)
    
    # PURPOSE: Forward pass - Applies multi-head attention using same input for Q, K, V
    # Allows sequence elements to attend to each other
    def forward(self, x, mask=None):
        """
        Apply self-attention to input sequence.
        
        Args:
            x: Input sequence (batch_size, seq_len, model_dim)
            mask: Optional attention mask
            
        Returns:
            output: Self-attention output (batch_size, seq_len, model_dim)
        """
        # SYNTAX: self.multi_head.forward(x, x, x) - Same input for Q, K, V
        # WHY: In self-attention, we want elements to attend to other elements in same sequence
        # PATTERN: Distinctive feature of self-attention versus cross-attention
        return self.multi_head.forward(x, x, x, mask)


class PositionalEncoding:
    """
    Adds positional information to input embeddings.
    Since attention has no inherent notion of position, this is critical for sequence processing.
    """
    
    # PURPOSE: Constructor - Creates position encodings
    # Generates sinusoidal patterns that uniquely identify positions in sequence
    def __init__(self, d_model, max_seq_length=5000):
        """Initialize positional encoding for transformer models."""
        # Create a position encoding matrix
        # SYNTAX: np.zeros((max_len, d_model)) - Initialize empty tensor
        # WHY: Will be filled with sinusoidal position encodings
        # PATTERN: Standard approach for positional encoding in transformers
        self.encoding = np.zeros((max_seq_length, d_model))
        
        # Create position indices
        # SYNTAX: np.arange(max_len)[:, np.newaxis] - Column vector of positions
        # WHY: Each row corresponds to a position in sequence
        # PATTERN: Setup for broadcasting in subsequent calculations
        position = np.arange(max_seq_length)[:, np.newaxis]  # (max_len, 1)
        
        # Create dimension indices for sinusoidal patterns
        # SYNTAX: np.arange(0, d_model, 2) - Even indices only
        # WHY: Even dimensions get cosine, odd dimensions get sine
        # PATTERN: Pattern from "Attention Is All You Need" paper
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Set encodings using sine and cosine patterns
        # SYNTAX: encoding[:, 0::2] = np.sin(...) - Set even columns
        # WHY: Creates unique pattern for each position that varies smoothly
        # PATTERN: Implementation directly from transformer paper
        self.encoding[:, 0::2] = np.sin(position * div_term)  # Even dimensions
        self.encoding[:, 1::2] = np.cos(position * div_term)  # Odd dimensions
        
        # Add batch dimension for easier addition to embeddings
        # SYNTAX: encoding[np.newaxis, :, :] - Add dimension at front
        # WHY: Matches batch dimension of input for broadcasting
        # PATTERN: Reshaping for compatibility with batch processing
        self.encoding = self.encoding[np.newaxis, :, :]  # (1, max_len, d_model)
    
    # PURPOSE: Apply encodings - Adds positional information to embeddings
    # Injects position awareness into otherwise position-agnostic attention
    def forward(self, x):
        """
        Add positional encodings to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            x: Embeddings with positional information added
        """
        # SYNTAX: x + self.encoding[:, :seq_len, :] - Element-wise addition
        # WHY: Adds unique positional signature to each token embedding
        # PATTERN: Simple addition is sufficient due to clever design of encodings
        seq_len = x.shape[1]
        return x + self.encoding[:, :seq_len, :]


class TransformerEncoderLayer:
    """
    A single layer of transformer encoder.
    Combines self-attention with a position-wise feed-forward network.
    """
    
    # PURPOSE: Constructor - Creates a transformer encoder layer
    # Implements the full architecture from "Attention Is All You Need"
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        """Initialize a transformer encoder layer."""
        self.self_attn = SelfAttention(model_dim, num_heads)
        
        # Feed-forward network
        # SYNTAX: Dictionary of weight matrices - Setup for two-layer FFN
        # WHY: Applies non-linear transformation after attention
        # PATTERN: Standard architecture of transformer encoder
        self.ff = {
            'w1': np.random.randn(model_dim, ff_dim) / np.sqrt(model_dim),
            'b1': np.zeros(ff_dim),
            'w2': np.random.randn(ff_dim, model_dim) / np.sqrt(ff_dim),
            'b2': np.zeros(model_dim)
        }
        
        # Layer normalization parameters
        # SYNTAX: np.ones(dim), np.zeros(dim) - Initialize scale and bias
        # WHY: Helps training stability and convergence
        # PATTERN: Critical component in modern deep learning
        self.layer_norm1 = {'gamma': np.ones(model_dim), 'beta': np.zeros(model_dim)}
        self.layer_norm2 = {'gamma': np.ones(model_dim), 'beta': np.zeros(model_dim)}
        
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
    
    # PURPOSE: Layer normalization - Normalizes activations
    # Stabilizes hidden states by normalizing mean and variance
    def _layer_norm(self, x, norm_params):
        """Apply layer normalization to input."""
        # SYNTAX: Compute statistics along last dimension
        # WHY: Normalizes each feature independently across batch and sequence
        # PATTERN: Standard implementation of layer normalization
        
        # Compute mean and variance for normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize and apply scale and shift
        # SYNTAX: (x - mean) / sqrt(var + epsilon) * gamma + beta
        # WHY: Centers and scales the data, then applies learnable transformation
        # PATTERN: Standard formula for layer normalization
        return norm_params['gamma'] * (x - mean) / np.sqrt(var + 1e-6) + norm_params['beta']
    
    # PURPOSE: Feed-forward network - Applies non-linear transformation
    # Processes each position independently with shared weights
    def _feed_forward(self, x):
        """Apply position-wise feed-forward network."""
        # First linear transformation followed by ReLU
        # SYNTAX: np.matmul(x, w1) + b1 - Linear layer
        # WHY: Increases model capacity through non-linear transformation
        # PATTERN: First half of standard two-layer FFN in transformers
        h = np.maximum(0, np.matmul(x, self.ff['w1']) + self.ff['b1'])  # ReLU
        
        # Second linear transformation
        # SYNTAX: np.matmul(h, w2) + b2 - Second linear layer
        # WHY: Projects back to model dimension after expansion
        # PATTERN: Second half of standard two-layer FFN
        return np.matmul(h, self.ff['w2']) + self.ff['b2']
    
    # PURPOSE: Dropout - Randomly zeros elements during training
    # Reduces overfitting by preventing co-adaptation
    def _dropout(self, x, training=True):
        """Apply dropout regularization."""
        if not training or self.dropout_rate == 0:
            return x
        
        # SYNTAX: np.random.binomial - Generate binary mask
        # WHY: Randomly drops out units during training
        # PATTERN: Standard dropout implementation
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * mask
    
    # PURPOSE: Forward pass - Complete transformer encoder layer
    # Processes sequence through attention and feed-forward networks
    def forward(self, x, mask=None, training=True):
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, model_dim)
            mask: Optional attention mask
            training: Whether in training mode (for dropout)
            
        Returns:
            output: Processed tensor (batch_size, seq_len, model_dim)
        """
        # Step 1: Self-attention sub-layer
        # SYNTAX: Residual connection - x + sublayer(x)
        # WHY: Helps gradient flow and enables deeper networks
        # PATTERN: Critical architectural pattern in transformers
        
        # Self-attention block
        attn_output, _ = self.self_attn.forward(x, mask)
        attn_output = self._dropout(attn_output, training)
        x = self._layer_norm(x + attn_output, self.layer_norm1)  # Add & Norm
        
        # Step 2: Feed-forward sub-layer
        # SYNTAX: Another residual connection with different sublayer
        # WHY: Same benefits as first residual connection
        # PATTERN: Repeated structure that defines transformer architecture
        
        # Feed-forward block
        ff_output = self._feed_forward(x)
        ff_output = self._dropout(ff_output, training)
        output = self._layer_norm(x + ff_output, self.layer_norm2)  # Add & Norm
        
        return output


# Example usage
if __name__ == "__main__":
    # Create a simple sequence for testing
    batch_size = 2
    seq_len = 5
    model_dim = 64
    
    # Create random input sequence
    x = np.random.randn(batch_size, seq_len, model_dim)
    
    # Add positional encoding
    pos_encoder = PositionalEncoding(model_dim)
    x_with_pos = pos_encoder.forward(x)
    
    # Create transformer encoder layer
    encoder_layer = TransformerEncoderLayer(
        model_dim=model_dim,
        num_heads=8,
        ff_dim=256
    )
    
    # Process sequence
    output = encoder_layer.forward(x_with_pos)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer successfully processed the sequence!")