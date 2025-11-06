import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import math
from tqdm import tqdm


from .coefs import make_coefs_dict_from_list
from .grad_desc import get_grad_indices

# -------------------------- Set Transformer and helpers --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        unsqueezed = False
        if q.dim() == 2:
            unsqueezed = True
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, Nq, _ = q.shape
        Nk = k.shape[1]

        q = self.q_lin(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.dim)
        out = self.out_lin(out)

        if unsqueezed:
            out = out.squeeze(0)
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


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.ln(x)


class SAB(nn.Module):
    def __init__(self, dim, num_heads=4, ln=True, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.ln1 = LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = LayerNorm(dim) if ln else nn.Identity()

    def forward(self, X):
        H = self.ln1(X + self.mha(X, X, X))
        return self.ln2(H + self.ff(H))


class ISAB(nn.Module):
    def __init__(self, dim, num_heads=4, m=32, ln=True, dropout=0.1):
        super().__init__()
        self.m = m
        self.inducing_points = nn.Parameter(torch.randn(1, m, dim))
        self.mha1 = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.mha2 = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.ln1 = LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = LayerNorm(dim) if ln else nn.Identity()

    def forward(self, X):
        unsqueezed = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            unsqueezed = True

        B = X.size(0)
        I = self.inducing_points.expand(B, -1, -1)

        H = self.mha1(I, X, X)
        H = self.ln1(I + H)
        H2 = self.mha2(X, H, H)
        H2 = self.ln2(X + H2)
        H2 = H2 + self.ff(H2)

        if unsqueezed:
            H2 = H2.squeeze(0)
        return H2


class PMA(nn.Module):
    def __init__(self, dim, num_heads=4, k=1, ln=True, dropout=0.1):
        super().__init__()
        self.k = k
        self.seed_vectors = nn.Parameter(torch.randn(1, k, dim))
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ln = LayerNorm(dim) if ln else nn.Identity()

    def forward(self, X):
        unsqueezed = False
        if X.dim() == 2:
            X = X.unsqueeze(0)
            unsqueezed = True

        B = X.size(0)
        S = self.seed_vectors.expand(B, -1, -1)
        H = self.mha(S, X, X)
        H = self.ln(H + S)

        if unsqueezed:
            H = H.squeeze(0)
        return H


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


# -------------------------- Main model --------------------------
class TaskToB(nn.Module):
    def __init__(
            self,
            input_dim,
            dim = 128,
            num_heads = 4,
            num_inds = 32,
            num_sab = 2,
            emb_dim = 256,
            u_dim = 256,
            b_dim = 20000,
            dropout = 0.1,
            is_full = True,
            M = 5,
            N = 20,
            n_coefs_folds = 100
            ):
        """
        Args:
            input_dim: dimensionality of per-element vector (features + target) i.e. d+1
            dim: hidden dim inside Transformer blocks
            num_heads: attention heads
            num_inds: number of inducing points for ISAB
            num_sab: number of stacked ISAB/SAB layers
            emb_dim: final pooled task embedding dim
            u_dim: dimension of intermediate vector u
            b_dim: final size of b
            dropout: dropout for all
            is_full: if True, forward() will apply the operator and return (output, b);
                     if False, forward() returns b only.
            M: number of differentials
            N: number of trees
            n_coefs_folds: number of folds when finding the gradient of the operator
        """
        super().__init__()

        self.is_full = is_full

        self.M = M
        self.N = N
        self.n_coefs_folds = n_coefs_folds

        self.input_proj = nn.Linear(input_dim, dim)

        # encoder: stack of ISAB blocks + a final SAB
        enc_blocks = []
        for i in range(num_sab):
            enc_blocks.append(ISAB(dim=dim, num_heads=num_heads, m=num_inds, ln=True, dropout=dropout))
        enc_blocks.append(SAB(dim=dim, num_heads=num_heads, ln=True, dropout=dropout))
        self.encoder = nn.Sequential(*enc_blocks)

        self.pma = PMA(dim=dim, num_heads=num_heads, k=1, ln=True, dropout=dropout)

        # task embedding head
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

    def forward(self, X, y, operator=None, operator_args = None):
        """
        X: (n, d)
        y: (n,)
        operator: callable that accepts (coefs_dict, **operator_args) and returns scalar loss
        operator_args: dict of additional arguments to pass to operator

        Returns:
            if self.is_full: (output_tensor, b) else: b
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
        P = self.pma(H)
        P = P.view(B, -1)

        emb = self.task_mlp(P)
        u = self.u_net(emb)

        b = torch.matmul(u, self.W.t()) + self.b_bias.unsqueeze(0)

        b = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        if single_batch:
            b = b.squeeze(0)

        if self.is_full:
            if operator is None:
                raise ValueError("operator must be provided when is_full=True")
            output = OperatorFunction.apply(b, operator, operator_args, self.M, self.N, get_grad_indices, self.n_coefs_folds)
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