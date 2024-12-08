import torch
import einops
import torch.nn as nn
    
class MLTC(nn.Module):
    
    def __init__(self, embed_dim, output_dim):
        super().__init__()

        self.kernel_size = (16, 64, 128)
        self.stride = (16, 64, 128)
        self.mlp_depth = 2
        self.hidden_size = 4096
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

        self.k_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)
        
        self.q_downsample_convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                      kernel_size=self.kernel_size[i], stride=self.stride[i], groups=self.embed_dim)
            for i in range(len(self.kernel_size))
        ])
        
        modules = []
        for _ in range(self.mlp_depth):
            modules.append(nn.Linear(self.embed_dim, self.embed_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(self.embed_dim, self.output_dim))
        self.mlp = nn.Sequential(*modules)
        

    def forward(self, hidden_states):
        # x = hidden_states[2] #orignal single level image features
        # text_features = hidden_states[1]
        # x_multi = hidden_states[0] #multi level image features
        x = hidden_states
        # add in query now
        # text_features = self.text_projection(text_features)
        # x = x + text_features 
        
        #add the query feature to the self
        query_states_1d = einops.rearrange(self.q_proj(x), 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        downsampled_q = self.q_downsample(query_states_1d)
        b, _, h, w = downsampled_q.size()

        # makes it so each grid counts as a separate batch
        query_states = einops.rearrange(downsampled_q, 'b d h w -> (b h w) 1 d')

        key_states = self.k_proj(x) # b x 576 x d
        value_states = self.v_proj(x)

        # for "chunking" a 2d tensor into a 2d grid (a c) (b d) -> (a b) c d gives use a*b cxd tensors
        # e.g., setting a,b=2, allows use to segment the tensor into 4 quadrants
        k = self.token_grid_size // h
        l = self.token_grid_size // w
        key_states = einops.rearrange(key_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        value_states = einops.rearrange(value_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        
        # attention is now from each convolution "grid" to the respective tokens
        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        output = einops.rearrange(attn_output, "(b t) 1 d -> b t d", b=b)

        return output