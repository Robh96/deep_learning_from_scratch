import numpy as np
# This module will include a standard implementation of self attention
# n_dims is number of features in embedding vector (i.e. its size)
# Q_matrix = input_embeddings * W_Q
# K_matrix = input_embeddings * W_K
# V_matrix = input_embeddings * W_V
# self_attention = softmax(QK^T/sqrt(D))*V

class self_attn:
    def __init__(self, n_dims):
        # 
        self.wq = np.random.randn(n_dims, n_dims)
        self.wk = np.random.randn(n_dims, n_dims)
        self.wv = np.random.randn(n_dims, n_dims)
        self.n_dims = n_dims

    def self_attention(self, X):
        q = X @ self.wq
        k = X @ self.wk
        v = X @ self.wv

        logits = (q @ k.T) / np.sqrt(self.n_dims)
        attention_weights = self.softmax(logits)
        attention_output = attention_weights @ v
        return attention_output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
