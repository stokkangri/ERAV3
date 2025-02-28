import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from smollm2 import SmolLM2Config, RMSNorm, MLP  # Import the existing classes

debug = False
def debug_print(*args, **kargs):
    if debug:
        print(*args, **kargs)

@dataclass
class DeepSeekConfig(SmolLM2Config):
    # Additional configs for DeepSeek architecture
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k_experts: int = 2
    expert_capacity_factor: float = 1.25
    moe_layer_freq: int = 2  # Apply MoE every n layers
    num_latent_heads: int = 4  # Number of latent heads for MLHA
    latent_dim: int = 64  # Dimension of latent space
    compression_ratio: int = 8  # Compression ratio for MLHA
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001

class MLHAAttention(nn.Module):
    """Multi-head Latent Attention implementation"""
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_latent_heads = config.num_latent_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.latent_dim = config.latent_dim // config.compression_ratio  # Apply compression
        
        # Main attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Latent attention components
        self.latent_q = nn.Parameter(torch.randn(self.num_latent_heads, self.latent_dim))
        self.latent_k = nn.Linear(config.hidden_size, self.num_latent_heads * self.latent_dim, bias=False)
        self.latent_v = nn.Linear(config.hidden_size, self.num_latent_heads * self.latent_dim, bias=False)
        
        # Add latent output projection to match hidden size
        self.latent_out_proj = nn.Linear(self.num_latent_heads * self.latent_dim, config.hidden_size, bias=False)
        
        # RoPE embeddings
        self.rope_theta = config.rope_theta
        self.register_buffer("rope_cache", self._build_rope_cache(config.max_position_embeddings))

    def _build_rope_cache(self, seq_len):
        theta = self.rope_theta
        pos = torch.arange(0, seq_len).float()
        dim = self.head_dim // 2
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim).float() / dim))
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        
        sin = sinusoid_inp.sin()
        cos = sinusoid_inp.cos()
        
        return torch.stack([cos, sin], dim=0)

    def _apply_rope(self, x, rope_cache):
        """Apply rotary positional embeddings to the input."""
        # x shape: [batch, seq_len, heads, head_dim]
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

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        
        # Main attention path
        q = self.q_proj(x).view(B, T, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_attention_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_attention_heads, self.head_dim)
        
        # Apply RoPE
        q = self._apply_rope(q, self.rope_cache)
        k = self._apply_rope(k, self.rope_cache)
        
        # Compute main attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        latent_k = self.latent_k(x)    # => shape [B, T, 32]
        latent_k = latent_k.view(B, T, self.num_latent_heads, self.latent_dim) # .transpose(1,2)  # => [B, T, num_latent_heads=4, latent_dim=8] # => [B, 4, T, 64]

        # Latent attention path
        # v
        latent_v = self.latent_v(x)  # [B, T, 32]
        latent_v = latent_v.view(B, T, self.num_latent_heads, self.latent_dim).transpose(1, 2)  # => [B, 4, T, 8]


        # Expand latent queries to [B, num_attention_heads, T, latent_dim]
        # Good for 4-latent-heads approach
        latent_q = self.latent_q.unsqueeze(0).unsqueeze(2)   # => [1, 4, 1, 8]
        latent_q = latent_q.expand(B, self.num_latent_heads, T, self.latent_dim)
        # => final shape [16, 4, 1024, 8]


        debug_print(f"latent_q shape: {latent_q.shape}")  # e.g. [16, 8, 1024, 4]
        debug_print(f"latent_k shape before view: {latent_k.shape}")  # e.g. [16, 8, 1024, 4]

        latent_k = latent_k.view(B, self.num_latent_heads, T, self.latent_dim)
        latent_v = latent_v.view(B, self.num_latent_heads, T, self.latent_dim)

        debug_print(f"latent_q shape: {latent_q.shape}")  # [B, self.num_latent_heads, T, latent_dim]
        debug_print(f"latent_k shape: {latent_k.shape}")  # [B, self.num_latent_heads, T, latent_dim]

        
        # Compute latent attention
        latent_scores = torch.matmul(latent_q, latent_k.transpose(-2, -1)) / math.sqrt(self.latent_dim)

        latent_probs = F.softmax(latent_scores, dim=-1)
        latent_out = torch.matmul(latent_probs, latent_v)
        debug_print(f"latent_out shape {latent_out.shape}")
        latent_out = latent_out.transpose(1, 2).contiguous().view(B, T, self.num_latent_heads*self.latent_dim )

        # Combine main and latent attention
        main_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        main_out = main_out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Project outputs
        #output = self.o_proj(main_out) + latent_out.view(B, T, -1)

        # Then if you want to add to main_out [B,T,768], do:
        latent_out = self.latent_out_proj(latent_out)  # a new nn.Linear(256, 768)
        # => [B, T, 768]

        output = self.o_proj(main_out) + latent_out
        return output

class MoELayer(nn.Module):
    """Mixture of Experts with Loss-less Load Balancing"""
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k_experts = config.top_k_experts
        self.hidden_size = config.hidden_size
        self.capacity_factor = config.expert_capacity_factor
        
        # Expert networks
        #self.experts = nn.ModuleList([
        #    MLP(config) for _ in range(self.num_experts)
        #])
        
        # Unique experts
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_experts - self.num_shared_experts)
        ])
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            MLP(config) for _ in range(self.num_shared_experts)
        ])
        
        # Router
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.router_weights = None
        
    def forward(self, x):

        #  ---- DEBUG 1: Input shape ----
        debug_print(f"[MoELayer] Input x shape: {x.shape}", flush=True)

        B, T, C = x.shape
        
        # Get router logits
        router_logits = self.router(x)  # [B, T, num_experts]
        # ---- DEBUG 2: Router logits shape ----
        debug_print(f"[MoELayer] router_logits shape: {router_logits.shape}", flush=True)
        
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        # ---- DEBUG 3: router_probs shape & sum check ----
        debug_print(f"[MoELayer] router_probs shape: {router_probs.shape}", flush=True)
        debug_print(f"[MoELayer] router_probs sum across experts (mean): {router_probs.sum(dim=-1).mean().item():.4f}", flush=True)

        # Calculate capacity
        capacity = int(self.capacity_factor * T * B / self.num_experts)
        debug_print(f"[MoELayer] capacity per expert: {capacity}", flush=True)

        # Route tokens to experts
        routes = torch.topk(router_probs, k=self.top_k_experts, dim=-1)
        scores, expert_indices = routes.values, routes.indices
        # ---- DEBUG 4: shapes of scores/indices ----
        debug_print(f"[MoELayer] scores shape: {scores.shape}, expert_indices shape: {expert_indices.shape}", flush=True)

      
        # Normalize scores
        scores_sum = scores.sum(dim=-1, keepdim=True)
        scores = scores / scores_sum
        
        # Dispatch tokens to experts
        final_output = torch.zeros_like(x)
        debug_print(f"[MoELayer] final_output init shape: {final_output.shape}", flush=True)
        
        router_z_loss = torch.zeros(1, device=x.device)
        router_aux_loss = torch.zeros(1, device=x.device)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            expert_input = x[expert_mask]
            
            if expert_input.shape[0] > 0:
                # ---- DEBUG 6: expert_input shape ----
                debug_print(f"    [MoELayer] expert_input shape: {expert_input.shape}", flush=True)
                
                # Process tokens with expert
                #expert_output = self.experts[expert_idx](expert_input)
                if expert_idx < self.num_shared_experts:
                    expert_output = self.shared_experts[expert_idx](expert_input)
                else:
                    expert_output = self.experts[expert_idx - self.num_shared_experts](expert_input)
                # ---- DEBUG 7: expert_output shape ----
                debug_print(f"    [MoELayer] expert_output shape: {expert_output.shape}", flush=True)
              

                # Combine expert outputs weighted by router scores
                expert_scores = scores[expert_mask][expert_indices[expert_mask] == expert_idx]
                final_output[expert_mask] += expert_output * expert_scores.unsqueeze(-1)
            
            # Calculate auxiliary losses
            router_z_loss += torch.mean(torch.square(router_logits))
            router_probs_for_aux = router_probs.mean(dim=(0, 1))
            router_aux_loss += self.num_experts * torch.mean(router_probs_for_aux * torch.log(router_probs_for_aux + 1e-9))
        # ---- DEBUG 8: final_output shape ----
        debug_print(f"[MoELayer] final_output shape after combining: {final_output.shape}", flush=True)
        #raise RuntimeError("DEBUG CRASH – verifying final_output shape shown above")

        
        return final_output, router_z_loss, router_aux_loss

class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = (layer_idx + 1) % config.moe_layer_freq == 0
        
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MLHAAttention(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if self.use_moe:
            self.mlp = MoELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        debug_print(f"[DeepSeekBlock] x shape before attn: {x.shape}")
        y = self.ln_1(x)
        debug_print(f"[DeepSeekBlock] y shape before attn: {y.shape}")
        z = self.attn(y)
        debug_print(f"[DeepSeekBlock] z shape before attn: {z.shape}")
        x = x + y # self.attn(self.ln_1(x))
        debug_print(f"[DeepSeekBlock] x shape after attn:  {x.shape}")

        if self.use_moe:
            mlp_out, z_loss, aux_loss = self.mlp(self.ln_2(x))
            x = x + mlp_out
            return x, z_loss, aux_loss
        else:
            x = x + self.mlp(self.ln_2(x))
            return x, None, None

class DeepSeekModel(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.embed_tokens(idx)
        
        # Track MoE losses
        total_z_loss = 0
        total_aux_loss = 0
        
        # Process through layers
        for layer in self.layers:
            if isinstance(layer, DeepSeekBlock):
                x, z_loss, aux_loss = layer(x)
                if z_loss is not None:
                    total_z_loss += z_loss
                if aux_loss is not None:
                    total_aux_loss += aux_loss
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Combine losses
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            z_loss_scaled = self.config.router_z_loss_coef * total_z_loss
            aux_loss_scaled = self.config.router_aux_loss_coef * total_aux_loss
            loss = ce_loss + z_loss_scaled + aux_loss_scaled
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generation with DeepSeek architecture"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_position_embeddings:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx 
