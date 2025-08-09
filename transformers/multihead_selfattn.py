import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention:
    """Multi-Head Self-Attention mechanism.
    
    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        self.dropout = dropout
        
        # Combined QKV projection for efficiency
        self.qkv_proj = self._init_linear(d_model, 3 * d_model, bias)
        self.out_proj = self._init_linear(d_model, d_model, bias)
        
        # Dropout layers
        self.attn_dropout = dropout
        self.resid_dropout = dropout
    
    def _init_linear(self, in_dim: int, out_dim: int, bias: bool) -> dict:
        """Initialize linear layer with proper scaling."""
        # Xavier/Glorot initialization
        std = np.sqrt(2.0 / (in_dim + out_dim))
        weight = np.random.normal(0, std, (in_dim, out_dim))
        bias_term = np.zeros(out_dim) if bias else None
        return {'weight': weight, 'bias': bias_term}
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len) or (seq_len, seq_len)
                  1 = can attend, 0 = cannot attend
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Generate Q, K, V in one shot for efficiency
        qkv = x @ self.qkv_proj['weight']
        if self.qkv_proj['bias'] is not None:
            qkv += self.qkv_proj['bias']
        
        # Reshape and split into Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        
        if mask is not None:
            # Expand mask to match attention scores dimensions
            if mask.ndim == 2:  # (seq_len, seq_len)
                mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
            elif mask.ndim == 3:  # (batch_size, seq_len, seq_len)  
                mask = mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)
            
            # Set masked positions to very negative value
            attn_scores = np.where(mask == 0, -1e9, attn_scores)
        
        # Softmax + dropout
        attn_weights = self._softmax(attn_scores)
        attn_weights = self._dropout(attn_weights, self.attn_dropout)
        
        # Apply attention to values
        attn_output = attn_weights @ v  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        # Final projection + dropout
        output = attn_output @ self.out_proj['weight']
        if self.out_proj['bias'] is not None:
            output += self.out_proj['bias']
        
        output = self._dropout(output, self.resid_dropout)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _dropout(self, x: np.ndarray, p: float, training: bool = True) -> np.ndarray:
        """Apply dropout during training."""
        if not training or p <= 0.0:
            return x
        
        # Generate random mask: 1 = keep, 0 = drop
        keep_prob = 1.0 - p
        mask = np.random.binomial(1, keep_prob, x.shape).astype(x.dtype)
        
        # Scale by keep_prob to maintain expected value
        return x * mask / keep_prob


class FeedForward:
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        # Two linear layers
        self.linear1 = self._init_linear(d_model, d_ff)
        self.linear2 = self._init_linear(d_ff, d_model)
    
    def _init_linear(self, in_dim: int, out_dim: int) -> dict:
        """Initialize with proper scaling."""
        std = np.sqrt(2.0 / in_dim)  # He initialization for ReLU-family
        weight = np.random.normal(0, std, (in_dim, out_dim))
        bias = np.zeros(out_dim)
        return {'weight': weight, 'bias': bias}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # First linear + activation
        hidden = x @ self.linear1['weight'] + self.linear1['bias']
        hidden = self._activation(hidden)
        hidden = self._dropout(hidden, self.dropout)
        
        # Second linear
        output = hidden @ self.linear2['weight'] + self.linear2['bias']
        output = self._dropout(output, self.dropout)
        
        return output
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            # Approximate GELU
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _dropout(self, x: np.ndarray, p: float) -> np.ndarray:
        if p > 0.0:
            mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
            return x * mask
        return x


class LayerNorm:
    """Layer Normalization with learnable parameters."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.weight = np.ones(d_model)
        self.bias = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.weight * normalized + self.bias


class TransformerBlock:
    """Single Transformer Block with Pre-Layer Normalization."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-layer norm + self-attention + residual
        attn_input = self.norm1.forward(x)
        attn_output = self.attention.forward(attn_input, mask)
        x = x + attn_output
        
        # Pre-layer norm + feed-forward + residual  
        ff_input = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(ff_input)
        x = x + ff_output
        
        return x


class Transformer:
    """Full Transformer model."""
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int, 
                 d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        
        logger.info(f"Initialized Transformer: {d_model}d, {num_heads}h, {num_layers}L")
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through all transformer blocks."""
        for block in self.blocks:
            x = block.forward(x, mask)
        
        return self.final_norm.forward(x)
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        # This would iterate through all parameters in a real implementation
        params_per_block = (
            4 * self.d_model * self.d_model +  # QKV + output projections
            2 * self.d_model * 4 * self.d_model +  # FFN layers
            4 * self.d_model  # Layer norm parameters
        )
        return self.num_layers * params_per_block