"""Hybrid CNN-Transformer model for remaining useful life prediction.

Typical hyperparameters:
    d_model = 128  # dimension in encoder
    heads = 4      # number of heads in multi-head attention
    N = 2          # number of encoder layers
    m = 14         # number of features
"""

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """Full model: gated convolutional input stage followed by a Transformer encoder."""

    def __init__(self, m: int, d_model: int, N: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.gating = Gating(d_model, m)
        self.encoder = Encoder(d_model, N, heads, m, dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor, t: int) -> torch.Tensor:
        """Predict the RUL for one time step given a (1, 1, 3, m) input window."""
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i, t)
        output = self.out(e_outputs)

        return output.reshape(1)


class Gating(nn.Module):
    """Gated convolutional unit with reset and update gates applied to the input window."""

    def __init__(self, d_model: int, m: int) -> None:  # 128,14
        super().__init__()
        self.m = m

        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=1),
        )

    def init_weights(self) -> None:
        """Initialize all parameters uniformly in [-1/sqrt(m), 1/sqrt(m)]."""
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gating mechanism and project to the encoder dimension."""
        x_i = x[:, :, 1:2, :]  # only applying the gating on the current row even with the stack of 3 rows cames as input (1,1,3,14)
        h_i = self.cnn_layers(x)  # shape becomes 1,1,1,14 as the nn.conv2d has output channel as 1 but the convolution is applied on whole past input (stack of three)

        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)

        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e  # (the final output is 1,1,1,128 as the encoder has size of 128.)


class Encoder(nn.Module):
    """Stack of N encoder layers with positional encoding and a final normalization."""

    def __init__(self, d_model: int, N: int, heads: int, m: int, dropout: float) -> None:
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, t: int) -> torch.Tensor:
        """Encode a single time step embedding through the layer stack."""
        src = src.reshape(1, self.d_model)  # this 128 is changed according to d_model
        x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](x, None)
        return self.norm(x)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding based on the current time step."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Scale the input and add the sinusoidal encoding for time step ``t``."""
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        pe = np.zeros(self.d_model)

        for i in range(0, self.d_model, 2):
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i + 1)) / self.d_model)))

        x = x + torch.Tensor(pe)
        return x


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Return a ModuleList of ``N`` deep copies of ``module``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    """Encoder layer with one multi-head attention layer and one feed-forward layer."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply self-attention and feed-forward sublayers with residual connections."""
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Norm(nn.Module):
    """Layer normalization with learnable gain and bias."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``."""
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, heads: int, d_model: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention over ``h`` heads and project back to ``d_model``."""
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int,
              mask: Optional[torch.Tensor] = None,
              dropout: Optional[nn.Dropout] = None) -> torch.Tensor:
    """Scaled dot-product attention with optional dropout."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with ReLU and dropout."""

    def __init__(self, d_model: int, d_ff: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation."""
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
