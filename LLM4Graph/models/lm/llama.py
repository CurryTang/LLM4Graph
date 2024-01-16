import torch 
import torch.nn as nn
from typing import Optional, Tuple
import math
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class CustomLLaMAConfig:
    dim: int = 4096
    multiple_of: int = 256 
    n_heads: int = 32
    n_layers: int = 32
    norm_eps: float = 1e-6
    vocab_size: int = -1
    first_layer: int = 31

    w_lora: bool = False
    kv_cache: bool = False 
    target_modules: list[str] = field(default_factory=list)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, lora_r, lora_alpha, lora_dropout=0.05,
    ):
        super().__init__()

        if lora_r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {min(in_features, out_features)}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        self.lora_up = nn.Linear(lora_r, out_features, bias=False)
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.lora_up.weight.dtype)
        result = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        result = result.to(previous_dtype)
        return result




class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_local_heads = args.n_heads 
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        self.q_proj = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        
        self.k_proj = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.w_lora = args.w_lora
        self.target_modules = args.target_modules
        self.debug = None

        if self.w_lora:
            if 'q_proj' in args.target_modules:
                self.lora_wq = LoraInjectedLinear(self.q_proj.in_features, self.q_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'k_proj' in args.target_modules:
                self.lora_wk = LoraInjectedLinear(self.k_proj.in_features, self.k_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

            if 'v_proj' in args.target_modules:
                self.lora_wv = LoraInjectedLinear(self.v_proj.in_features, self.v_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

            if 'o_proj' in args.target_modules:
                self.lora_wo = LoraInjectedLinear(self.o_proj.in_features, self.o_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        
        self.kv_cache = args.kv_cache
        if args.kv_cache:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_heads,
                    self.head_dim,
                )
            ).cuda()
        else:
            self.cache_k, self.cache_v = None, None 
    

    def forward(self, x, start_pos = -1, freqs_cis = None, mask = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.w_lora:
            if 'q_proj' in self.target_modules:
                xq = xq + self.lora_wq(x)
            if 'k_proj' in self.target_modules:
                xk = xk + self.lora_wk(x)
            if 'v_proj' in self.target_modules:
                xv = xv + self.lora_wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk 
        values = xv 

        if self.kv_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys, values
            else:
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # if DEBUG: 
        #     import time
        #     torch.save(scores, '/tmp/scores_{}.pt'.format(time.time()))
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)
    

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, args: CustomLLaMAConfig
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.params = args

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False,)    # gate_proj
        self.up_proj = nn.Linear(hidden_dim, dim, bias=False,)    # down_proj
        self.down_proj = nn.Linear(dim, hidden_dim, bias=False,)    # up_proj

        if self.params.w_lora:
            if 'up_proj' in args.target_modules:
                self.lora_w3 = LoraInjectedLinear(self.down_proj.in_features, self.down_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'down_proj' in args.target_modules:
                self.lora_w2 = LoraInjectedLinear(self.up_proj.in_features, self.up_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            if 'gate_proj' in args.target_modules:
                self.lora_w1 = LoraInjectedLinear(self.gate_proj.in_features, self.gate_proj.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    
    def forward(self, x):
        up_x = self.down_proj(x)
        gate_x = self.gate_proj(x)

        if self.params.w_lora:
            if 'up_proj' in self.params.target_modules:
                up_x = up_x + self.lora_w3(x)

            if 'gate_proj' in self.params.target_modules:
                gate_x = gate_x + self.lora_w1(x)

        down_input = F.silu(gate_x) * up_x
        out = self.up_proj(down_input)

        if self.params.w_lora:
            if 'down_proj' in self.params.target_modules:
                out = out + self.lora_w2(down_input)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: CustomLLaMAConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self, x: torch.Tensor, start_pos: int = -1, freqs_cis: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out


class LLaMATransformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        ## this is a special argument
        ## determine we start from which layer
        self.first_layer = args.first_layer

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.first_layer, self.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id, args
                )
            )
        
        self.norm = RMSNorm(
            args.dim, 
            eps = args.norm_eps
        )

    
    def forward(self, tokens):
        # bsz, token_num, hidden_dim = tokens.shape
        h = tokens
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return h.float()


    def custom_load_state_dict(self, checkpoint, tail = False, strict = False):
        if tail:
            # import ipdb; ipdb.set_trace()
            for i in range(self.first_layer, self.n_layers):
                layer_checkpoint_keys = [k for k in checkpoint.keys() if f'layers.{i}.' in k]
                layer_checkpoint_keys = [k.replace(f'model.layers.{i}.', '') for k in layer_checkpoint_keys]
                layer_checkpoint = {f'layers.{i - self.first_layer}.{k}' : checkpoint[f'model.layers.{i}.{k}'] for k in layer_checkpoint_keys}
                self.layers[i - self.first_layer].load_state_dict(
                    layer_checkpoint, strict=strict)