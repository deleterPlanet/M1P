import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import math
from tqdm import tqdm

from .coefs import make_coefs_dict_from_list
from .grad_desc import get_grad_indices


# -------------------------- Simple Self-Attention --------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.net(x))


# -------------------------- Main Model --------------------------
class TaskToB(nn.Module):
    def __init__(
        self,
        input_dim,
        dim=128,
        num_heads=4,
        num_layers=3,
        emb_dim=256,
        u_dim=256,
        b_dim=20000,
        dropout=0.1,
        is_full=True,
        M=5,
        N=20,
        n_coefs_folds=100
    ):
        super().__init__()

        self.is_full = is_full
        self.M = M
        self.N = N
        self.n_coefs_folds = n_coefs_folds

        self.input_proj = nn.Linear(input_dim, dim)

        # self-attention блоки
        blocks = [TransformerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        self.encoder = nn.Sequential(*blocks)

        # усреднение по токенам
        self.pool = lambda x: x.mean(dim=1)

        # task embedding MLP
        self.task_mlp = nn.Sequential(
            nn.Linear(dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            ResidualBlock(emb_dim),
            nn.Dropout(dropout),
            ResidualBlock(emb_dim),
            nn.Dropout(dropout),
            ResidualBlock(emb_dim),
            nn.Dropout(dropout),
            ResidualBlock(emb_dim),
            nn.Dropout(dropout),

            nn.LayerNorm(emb_dim)
        )

        # small MLP to produce u
        self.u_net = nn.Sequential(
            nn.Linear(emb_dim, u_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(u_dim, u_dim)
        )

        # low-rank final projection
        self.b_dim = b_dim
        self.u_dim = u_dim
        self.W = nn.Parameter(torch.randn(b_dim, u_dim) * (1.0 / math.sqrt(u_dim)))
        self.b_bias = nn.Parameter(torch.zeros(b_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X, y, operator=None, operator_args=None):
        """
        X: (n, d)
        y: (n,)
        operator: callable(coefs_dict, operator_args) -> scalar
        """
        single_batch = False
        if X.dim() == 2:
            single_batch = True
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)

        B, n, d = X.shape

        if y.dim() == 2 and y.shape[-1] != 1:
            y = y.unsqueeze(-1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)

        XY = torch.cat([X, y], dim=-1)
        H = self.input_proj(XY)

        H = self.encoder(H)
        P = self.pool(H)

        emb = self.task_mlp(P)
        u = self.u_net(emb)

        b = torch.matmul(u, self.W.t()) + self.b_bias.unsqueeze(0)

        if single_batch:
            b = b.squeeze(0)

        if self.is_full:
            if operator is None:
                raise ValueError("operator must be provided when is_full=True")
            output = OperatorFunction.apply(b, operator, operator_args, self.M, self.N, get_grad_indices, self.n_coefs_folds)

            b_norm = torch.norm(b, p=2)
            output = torch.log1p(output) + b_norm
            return output, b

        return b


# -------------------------- Custom autograd wrapper --------------------------
class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, operator, operator_args, M, N, get_grad_indices, n_coefs_folds):
        ctx.operator = operator
        ctx.operator_args = operator_args
        ctx.M = M
        ctx.N = N
        ctx.get_grad_indices = get_grad_indices
        ctx.n_coefs_folds = n_coefs_folds

        ctx.save_for_backward(b)

        b_np = b.detach().cpu().numpy().reshape(-1)
        coefs = make_coefs_dict_from_list(b_np, M, N)
        score = operator(coefs, **operator_args)

        return torch.tensor(score, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_output):
        (b,) = ctx.saved_tensors
        b_np = b.detach().cpu().numpy().reshape(-1)

        coefs_keys = list(make_coefs_dict_from_list(b_np, ctx.M, ctx.N).keys())
        coefs = {k: b_np[i] for i, k in enumerate(coefs_keys)}

        indices_list = ctx.get_grad_indices(coefs_keys, ctx.n_coefs_folds)

        eps = 1e-3
        grad_b = np.zeros_like(b_np)

        F_w = ctx.operator(coefs, **ctx.operator_args)

        for indices in tqdm(indices_list):
            mean_coef = np.mean([coefs[coefs_keys[idx]] for idx in indices])

            copy_plus = coefs.copy()
            copy_minus = coefs.copy()
            for idx in indices:
                copy_plus[coefs_keys[idx]]  += mean_coef * eps
                copy_minus[coefs_keys[idx]] -= mean_coef * eps

            F_plus  = ctx.operator(copy_plus,  **ctx.operator_args)
            #F_minus = ctx.operator(copy_minus, **ctx.operator_args)

            grad_dir  = (F_plus  - F_w) / (mean_coef * eps * (len(indices) ** 1.5))
            #grad_dir_minus = (F_minus - F_w) / (mean_coef * eps * (len(indices) ** 1.5))

            #grad_dir = (grad_dir_plus + grad_dir_minus) / 2

            for idx in indices:
                grad_b[idx] = grad_dir

        grad_b = torch.tensor(grad_b, dtype=b.dtype, device=b.device).reshape_as(b)
        return grad_output * grad_b, None, None, None, None, None, None