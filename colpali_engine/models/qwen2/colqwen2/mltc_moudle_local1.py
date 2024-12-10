import torch
import einops
import torch.nn.functional as F
import torch.nn as nn
    
class MLTC(nn.Module):
    
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.moudel_name = "MLTC"
        self.pool_kernel_size = 4
        self.mlp_depth = 2
        self.num_heads = 8
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.q_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.q_proj.append(nn.GELU())
        self.q_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.q_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.q_proj = nn.Sequential(*self.q_proj)

        self.k_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)

        self.q_pool = nn.AvgPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size, padding=self.pool_kernel_size // 2)

        modules = []
        for _ in range(self.mlp_depth):
            modules.append(nn.Linear(self.embed_dim, self.embed_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(self.embed_dim, self.output_dim))
        self.mlp = nn.Sequential(*modules)


    def forward(self, hidden_states, attention_mask = None):
        if torch.isnan(hidden_states).any():
            raise ValueError(f"NaN detected in hidden_states!")
        # pool_out =self.q_pool(hidden_states.transpose(2, 1))
        # pool_out = pool_out.transpose(1, 2)
        # q = self.q_proj(pool_out)

        q = self.q_proj(hidden_states)
        
        k = self.k_proj(hidden_states)  # Shape: [batch_size, seq_len, embed_dim]
        v = self.v_proj(hidden_states)  # Shape: [batch_size, seq_len, embed_dim]
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()  # Convert to bool type

        attn_output, attn_weights = self.clip_attn(q, k, v, key_padding_mask = attention_mask)  # Shape: [batch_size, seq_len, embed_dim]
        output = self.mlp(attn_output)  # Shape: [batch_size, seq_len, output_dim]
        output = F.normalize(output, p=2, dim=-1)
        if torch.isnan(output).any():
            raise ValueError(f"NaN detected!")
        return output