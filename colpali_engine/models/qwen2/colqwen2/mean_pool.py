import torch
import einops
import torch.nn.functional as F
import torch.nn as nn
    
class MeanPool(nn.Module):
    
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.moudel_name = "MLTC"
        self.pool_kernel_size = 4
        self.mlp_depth = 2
        self.embed_dim = embed_dim
        self.output_dim = output_dim
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
        pool_out =self.q_pool(hidden_states.transpose(2, 1))
        pool_out = pool_out.transpose(1, 2)

        output = self.mlp(pool_out)  # Shape: [batch_size, seq_len, output_dim]
        output = F.normalize(output, p=2, dim=-1)
        if torch.isnan(output).any():
            raise ValueError(f"NaN detected!")
        return output