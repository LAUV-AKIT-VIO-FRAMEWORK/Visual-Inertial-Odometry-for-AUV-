import torch
import torch.nn as nn
import torch.nn.functional as F

class MAB(nn.Module):
    """Multihead Attention Block: Generic block used by ISAB and PMA"""
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, queries, keys, key_padding_mask=None):
        # Attention on keys, normalized over queries
        attn, _ = self.mha(queries, keys, keys, key_padding_mask=key_padding_mask)
        x = self.norm1(queries + self.dropout(attn))
        
        # Feedforward part
        ff = self.fc(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class ISAB(nn.Module):
    """Induced Self-Attention Block: Uses inducing points for efficiency"""
    def __init__(self, dim, num_inducing=32, heads=4, dropout=0.1):
        super().__init__()
        # Learnable inducing points (M in the paper)
        self.inducing_points = nn.Parameter(torch.Tensor(1, num_inducing, dim))
        nn.init.xavier_uniform_(self.inducing_points)
        self.mab1 = MAB(dim, heads, dropout)
        self.mab2 = MAB(dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        inducing_pts = self.inducing_points.repeat(batch_size, 1, 1)
        
        # Step 1: Inducing points attend to the input set
        H = self.mab1(inducing_pts, x, key_padding_mask=mask)
        H = self.dropout(H)
        
        # Step 2: Input set attends to the summarized inducing points
        return self.mab2(x, H, key_padding_mask=mask)

class PMA(nn.Module):
    """Pooling by Multihead Attention: Learnable aggregation instead of mean()"""
    def __init__(self, dim, num_seeds=1, heads=4, dropout=0.1):
        super().__init__()
        # Learnable seed vector (S in the paper)
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, heads, dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seed = self.S.repeat(batch_size, 1, 1)
        
        # Seed attends to the set to produce a fixed-size summary
        return self.mab(seed, x, key_padding_mask=mask)

class AKITTransformer(nn.Module):
    """Adaptive Kalman Filter Tuning with Set Transformer"""
    def __init__(self, context_dim=7, set_dim=12, hidden_dim=128, 
                 num_inducing=32, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Context encoder (processes temporal IMU data)
        self.ctx_enc = nn.LSTM(
            context_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.ctx_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Set encoder for visual features (ISAB blocks)
        self.set_input_proj = nn.Sequential(
            nn.Linear(set_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.isab1 = ISAB(hidden_dim, num_inducing=num_inducing, 
                          heads=num_heads, dropout=dropout)
        self.isab2 = ISAB(hidden_dim, num_inducing=num_inducing, 
                          heads=num_heads, dropout=dropout)
        self.isab3 = ISAB(hidden_dim, num_inducing=num_inducing, 
                          heads=num_heads, dropout=dropout)
        
        # Multi-seed pooling for richer representation
        self.pma = PMA(hidden_dim, num_seeds=4, heads=num_heads, dropout=dropout)
        
        # Feature fusion with gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Output heads with proper activations
        # Q (process noise) - 3 components [gyro, accel, bias]
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softplus()  # Ensure positivity
        )
        
        # R (measurement noise) - single value
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                    
    def forward(self, context, mset, mask=None, return_attention=False):
        """
        Args:
            context: Temporal context (IMU data) [Batch, Window, ContextDim]
            mset: Set features (visual measurements) [Batch, SetSize, SetDim]
            mask: Optional mask for variable-length sets
            return_attention: Whether to return attention weights
        """
        # 1. Encode temporal context
        ctx_out, (h, _) = self.ctx_enc(context)
        ctx_feat = self.ctx_proj(ctx_out.mean(dim=1))  # Pool over time
        
        # 2. Encode visual set using ISABs
        x = self.set_input_proj(mset)
        x = self.isab1(x, mask)
        x = self.isab2(x, mask)
        x = self.isab3(x, mask)
        
        # 3. Pool to fixed-size representation
        pooled = self.pma(x, mask)  # [B, num_seeds, H]
        set_feat = pooled.mean(dim=1)  # Average over seeds
        
        # 4. Adaptive fusion with gating
        fused = torch.cat([ctx_feat, set_feat], dim=-1)
        gate = self.fusion_gate(fused)
        
        # Gated fusion: blend context and set features
        combined = gate * ctx_feat + (1 - gate) * set_feat
        combined = self.fusion_norm(combined)
        
        # 5. Predict noise parameters
        q_scales = self.q_head(combined) + 1e-6  # [B, 3]
        r_scale = self.r_head(combined) + 1e-6   # [B, 1]
        
        if return_attention:
            return q_scales, r_scale, pooled
        return q_scales, r_scale
    
    def get_attention_weights(self, context, mset, mask=None):
        """Helper method to visualize attention"""
        self.eval()
        with torch.no_grad():
            _, _, attention = self.forward(context, mset, mask, return_attention=True)
        return attention
