import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class SmolLM2Config:
    # Model architecture params from config_smollm2_135M.yaml
    hidden_size: int = 576  # matches n_embd in S13
    num_attention_heads: int = 9
    num_key_value_heads: int = 3  # New: uses grouped-query attention
    num_hidden_layers: int = 30
    intermediate_size: int = 1536
    vocab_size: int = 49152
    max_position_embeddings: int = 2048
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.041666666666666664
    use_cache: bool = True
    pad_token_id: int = None
    bos_token_id: int = 0
    eos_token_id: int = 0
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    rope_scaling = None
    rope_interleaved: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMSNorm
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return self.weight * x

class CausalSelfAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        
        # Grouped-query attention setup
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Compute query, key, value projections with grouped-query attention
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, 
                               self.head_dim * config.num_key_value_heads, 
                               bias=False)
        self.v_proj = nn.Linear(config.hidden_size, 
                               self.head_dim * config.num_key_value_heads, 
                               bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Initialize rotary embeddings
        self.rope_theta = config.rope_theta
        self.register_buffer(
            "rope_cache",
            self._build_rope_cache(config.max_position_embeddings),
            persistent=False,
        )

    def _build_rope_cache(self, seq_len):
        theta = self.rope_theta
        pos = torch.arange(0, seq_len).float()
        dim = self.head_dim // 2  # Important: use half dimension
        
        # Create position frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim).float() / dim))
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        
        # Create rotation matrices
        sin = sinusoid_inp.sin()  # [seq_len, dim]
        cos = sinusoid_inp.cos()  # [seq_len, dim]
        
        # Cache both sin and cos values
        return torch.stack([cos, sin], dim=0)  # [2, seq_len, dim]

    def _apply_rope(self, x, rope_cache):
        B, T, H, D = x.shape  # [batch, seq_len, heads, head_dim]
        D_half = D // 2  # Rotary embeddings apply to half the dimension

        # Get relevant part of rope cache
        cos, sin = rope_cache[:, :T, :D_half]  # Extract only necessary dimensions

        # Debugging shapes before reshaping
        #print(f"x shape: {x.shape}, cos shape: {cos.shape}, sin shape: {sin.shape}")

        # Reshape cos/sin to match `x` shape
        cos = cos.view(1, T, 1, D_half).expand(B, T, H, D_half)  # Shape [B, T, H, D//2]
        sin = sin.view(1, T, 1, D_half).expand(B, T, H, D_half)  # Shape [B, T, H, D//2]

        # Reshape x into pairs for rotation
        x_reshape = x.view(B, T, H, D_half, 2)

        # Debugging shapes after reshaping
        #print(f"x_reshape shape: {x_reshape.shape}, cos shape after expand: {cos.shape}")

        # Apply rotation using pairs
        x_out = torch.empty_like(x_reshape)
        x_out[..., 0] = x_reshape[..., 0] * cos - x_reshape[..., 1] * sin
        x_out[..., 1] = x_reshape[..., 0] * sin + x_reshape[..., 1] * cos

        return x_out.flatten(-2)  # Restore original shape

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        q = self.q_proj(x)  # [B, T, H*D]
        k = self.k_proj(x)  # [B, T, Hkv*D]
        v = self.v_proj(x)  # [B, T, Hkv*D]
        
        # Reshape and apply rotary embeddings
        q = q.view(B, T, self.num_attention_heads, self.head_dim)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings before transpose
        q = self._apply_rope(q, self.rope_cache)
        k = self._apply_rope(k, self.rope_cache)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, Hkv, T, D]
        v = v.transpose(1, 2)  # [B, Hkv, T, D]
        
        # Repeat k/v heads if needed for grouped-query attention
        if self.num_key_value_heads != self.num_attention_heads:
            k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
        
        # Compute attention with flash attention when available
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config, device=None):
        super().__init__()
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lazy load embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Lazy load layers
        self.layers = nn.ModuleList([None for _ in range(config.num_hidden_layers)])
        
        # Load first layer eagerly (others load on demand)
        self.load_layer(0)
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def load_layer(self, layer_idx):
        """Lazy load a specific transformer layer"""
        if isinstance(self.layers[layer_idx], type(None)):
            self.layers[layer_idx] = Block(self.config).to(self.device)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        #print(f"Input idx shape: {idx.shape}")  # Debug print

        x = self.embed_tokens(idx)
        #print(f"Embedded x shape: {x.shape}")  # Debug print

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, type(None)):
                self.load_layer(i)
            x = self.layers[i](x)

        x = self.norm(x)
        logits = self.lm_head(x)
        #print(f"Logits shape: {logits.shape}")  # Debug print

        loss = None
        if targets is not None:
            #print(f"Targets shape: {targets.shape}")  # Debug print
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size tokens if needed
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            
            # Get predictions
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx 