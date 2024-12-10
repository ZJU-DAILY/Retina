import torch
import einops
import torch.nn.functional as F
import torch.nn as nn
    
class MLTC(nn.Module):
    
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.moudel_name = "MLTC"
        # self.kernel_size = (16, 64, 128)
        # self.stride = (8, 32, 64)
        self.kernel_size = [3]
        self.stride = [1]
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
        
        # self.q_downsample_convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
        #               kernel_size=self.kernel_size[i], stride=self.stride[i], groups=self.embed_dim, padding=self.kernel_size[i]//2)
        #     for i in range(len(self.kernel_size))
        # ])
        # self.q_pools = nn.ModuleList([
        #     nn.AvgPool1d(kernel_size=self.kernel_size[i], stride=self.stride[i], padding=self.kernel_size[i] // 2)
        #     for i in range(len(self.kernel_size))
        # ])
        self.q_pool = nn.AvgPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
        self.q_downsample_convs = nn.ModuleList(
             [nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=(k, self.embed_dim), stride=s) for k,s in zip(self.kernel_size, self.stride)]
        )
    #     self.q_downsample_convs = nn.ModuleList([
    #                 nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=(k, self.embed_dim), stride=s),
    # #                              nn.BatchNorm1d(num_features=config.feature_size), 
    #                             nn.ReLU())
    #                     for k,s in zip(self.kernel_size, self.stride)
    #                     ])
     
        
        
        
        modules = []
        for _ in range(self.mlp_depth):
            modules.append(nn.Linear(self.embed_dim, self.embed_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(self.embed_dim, self.output_dim))
        self.mlp = nn.Sequential(*modules)
        # def init_weights(m):
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

        # self.apply(init_weights)
    def conv_and_pool(self, x, conv):
        # x = conv(x)#  最后一个维度为1   ：(input_height-kenl_size+padding*2)/stride[0]
        # x = F.relu(x)
        x = self.q_pool(x.transpose(3, 2).squeeze(1))
        return x
    def forward(self, hidden_states, attention_mask = None):
        # x = hidden_states[2] #orignal single level image features
        # text_features = hidden_states[1]
        # x_multi = hidden_states[0] #multi level image features
        # x = hidden_states
        # # add in query now
        # # text_features = self.text_projection(text_features)
        # # x = x + text_features 
        
        # #add the query feature to the self
        # query_states_1d = einops.rearrange(self.q_proj(x), 'b (h w) d -> b d h w',
        #                                     h = self.token_grid_size,
        #                                     w = self.token_grid_size)
        # downsampled_q = self.q_downsample(query_states_1d)
        # b, _, h, w = downsampled_q.size()

        # # makes it so each grid counts as a separate batch
        # query_states = einops.rearrange(downsampled_q, 'b d h w -> (b h w) 1 d')

        # key_states = self.k_proj(x) # b x 576 x d
        # value_states = self.v_proj(x)

        # # for "chunking" a 2d tensor into a 2d grid (a c) (b d) -> (a b) c d gives use a*b cxd tensors
        # # e.g., setting a,b=2, allows use to segment the tensor into 4 quadrants
        # k = self.token_grid_size // h
        # l = self.token_grid_size // w
        # key_states = einops.rearrange(key_states, "b (i k j l) d -> (b i j) (k l) d",
        #                  i=h, j=w, k=k, l=l)
        # value_states = einops.rearrange(value_states, "b (i k j l) d -> (b i j) (k l) d",
        #                  i=h, j=w, k=k, l=l)
        
        # # attention is now from each convolution "grid" to the respective tokens
        # attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        # output = einops.rearrange(attn_output, "(b t) 1 d -> b t d", b=b)
        
        # batch_size, seq_len, embed_dim = hidden_states.shape
        # Project hidden_states to q
        
        if torch.isnan(hidden_states).any():
            raise ValueError(f"NaN detected in hidden_states!")
        q = self.q_proj(hidden_states)  # Shape: [batch_size, seq_len, embed_dim]
        q = q.unsqueeze(1) # Shape: [batch_size, 1, seq_len, embed_dim]
        # Pass q through three 1D convolutions and concatenate results
        # q_convs = [conv(q.transpose(1, 2)) for conv in self.q_downsample_convs]
        q_convs = []
        for conv in self.q_downsample_convs:
            conv_out = self.conv_and_pool(q, conv)
            # pooled = torch.mean(conv_out, dim=2).values
            conv_out = conv_out.transpose(1, 2)
            q_convs.append(conv_out)
        # for i, conv in enumerate(self.q_downsample_convs):
        #     kernel_size = conv.kernel_size[0]
        #     seq_len = q.size(2)

        #     # 检查卷积核大小和序列长度
        #     print(f"Conv {i}: kernel_size={kernel_size}, seq_len={seq_len}")

        #     q_conv = conv(q.transpose(1, 2) )

        #     # 检查卷积输出范围
        #     print(f"Conv {i} output range: min={q_conv.min().item()}, max={q_conv.max().item()}")

        #     # 检查是否存在 NaN
        #     if torch.isnan(q_conv).any():
        #         raise ValueError(f"NaN detected in Conv {i} output!")

        #     q_convs.append(q_conv)
        q = torch.cat(q_convs, dim=1)  # Concatenate on the last dimension
        # q_pools = []
        # for i, pool in enumerate(self.q_downsample_convs):
        #     seq_len = q.size(1)  # Sequence length
        #     q_pool = pool(q.transpose(1, 2))  # Apply pooling (B, D, L)
        #     q_pools.append(q_pool.transpose(1, 2))  # Transpose back to (B, L, D)

        #     # Debugging output
        #     print(f"Pooling {i}: kernel_size={pool.kernel_size}, stride={pool.stride}")
        #     print(f"Output range: min={q_pool.min().item()}, max={q_pool.max().item()}")
        #     if torch.isnan(q_pool).any():
        #         raise ValueError(f"NaN detected in Conv {i} output!")

        # q = torch.cat(q_pools, dim=-1)  # Concatenate on the last dimension

        # Project hidden_states to k and v
        # k = self.k_proj(hidden_states)  # Shape: [batch_size, seq_len, embed_dim]
        # v = self.v_proj(hidden_states)  # Shape: [batch_size, seq_len, embed_dim]
        # if attention_mask is not None:
        #     attention_mask = attention_mask.bool()  # Convert to bool type
        # # Compute attention using clip_attn
        # attn_output, _ = self.clip_attn(q, k, v, key_padding_mask = attention_mask)  # Shape: [batch_size, seq_len, embed_dim]

        # # Final output through MLP
        # output = self.mlp(attn_output)  # Shape: [batch_size, seq_len, output_dim]
        q = F.normalize(q, p=2, dim=-1)
        if torch.isnan(q).any():
            raise ValueError(f"NaN detected!")
        return q