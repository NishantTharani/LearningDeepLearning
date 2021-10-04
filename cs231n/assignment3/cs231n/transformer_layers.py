import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        ############################################################################
        # TODO: Initialize any remaining layers and parameters to perform the      #
        # attention operation as defined in Transformer_Captioning.ipynb. We will  #
        # also apply dropout just after the softmax step. For reference, our       #
        # solution is less than 5 lines.                                           #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.dropout = dropout
        self.num_heads = num_heads
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, T, D))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        H = self.num_heads
        E = D
        
        # Compute the transformations of the inputs
        key_out = self.key(key)
        value_out = self.value(value)
        query_out = self.query(query)
        
        # Reshape the transformations
        key_split = torch.reshape(key_out, shape=(N, T, H, E//H))
        value_split = torch.reshape(value_out, shape=(N, T, H, E//H))
        query_split = torch.reshape(query_out, shape=(N, S, H, E//H))
        
        # Swap axes to prepare matrices for matmul
        query_swapped = torch.swapaxes(query_split, 1, 2)  # -> (N, H, S, E//H)
        key_swapped = torch.swapaxes(key_split, 1, 2)  # -> (N, H, T, E//H)
        key_swapped = torch.swapaxes(key_swapped, 2, 3)  # -> (N, H, E//H, T)
        
        # Obtain the alignment scores
        alignment_scores = torch.matmul(query_swapped, key_swapped)  # -> (N, H, S, T)
        
        # Swap axes so that masking will work
        alignment_swapped = torch.swapaxes(alignment_scores, 2, 3)  # -> (N, H, T, S)

        # Apply mask to alignment scores
        if attn_mask is not None:
            bool_mask = torch.tensor(attn_mask == 0)
        else:
            bool_mask = torch.full((T, S), False)

        alignment_swapped = alignment_swapped.masked_fill(bool_mask, float('-inf'))
        
        # Scale and get softmax scores
        scaling_term = math.sqrt(E // H)
        alignment_scaled = alignment_swapped / scaling_term

        attn_scores = F.softmax(alignment_scaled, dim=2)  # -> (N, H, T, S) 
                
        # attn_scores are (N, H, T, S)
        # values are (N, T, H, E//H)
        attn_scores = torch.unsqueeze(attn_scores, dim=2)  # -> (N, H, 1, T, S)
        attn_scores = attn_scores.repeat(1, 1, E//H, 1, 1)  # -> (N, H, E//H, T, S)
        value_swapped = torch.swapaxes(value_split, 1, 2)  # -> (N, H, T, E//H)
        value_swapped = torch.swapaxes(value_swapped, 2, 3)  # -> (N, H, E//H, T)
        value_swapped = torch.unsqueeze(value_swapped, dim=4)  # -> (N, H, E//H, T, 1)
        value_swapped = value_swapped.repeat(1, 1, 1, 1, S)  # -> (N, H, E//H, T, S)
        
        outputs = attn_scores * value_swapped  # -> (N, H, E//H, T, S)
        outputs = torch.sum(outputs, dim=3)  # -> (N, H, E//H, S)
        outputs = torch.swapaxes(outputs, 2, 3)  # -> (N, H, S, E//H)
        
        # Apply dropout
        outputs = F.dropout(outputs, p=self.dropout)
        
        # Concatenate the outputs and project to the final output
        outputs_swapped = torch.swapaxes(outputs, 1, 2)  # -> (N, S, H, E//H)
        outputs_squished = torch.reshape(outputs_swapped, shape=(N, S, E))
        output = self.proj(outputs_squished)
        
        """
        # Get softmax scores
        attn_scores = F.softmax(alignment_swapped, dim=2)  # -> (N, H, T, S) 
                
        # Swap axes to prepare for matmul
        attn_swapped = torch.swapaxes(attn_scores, 2, 3)  # -> (N, H, S, T)
        value_swapped = torch.swapaxes(value_split, 1, 2)  # -> (N, H, T, E//H)
        outputs = torch.matmul(attn_swapped, value_swapped)  # -> (N, H, S, E//H)
        
        # Apply dropout
        outputs = F.dropout(outputs, p=self.dropout)
        
        # Concatenate the outputs and project to the final output
        outputs_swapped = torch.swapaxes(outputs, 1, 2)  # -> (N, S, H, E//H)
        outputs_squished = torch.reshape(outputs_swapped, shape=(N, S, E))
        output = self.proj(outputs_squished)
        """
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


