import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int):
        return self.pe[:length]  # [L, dim]


class Struct2DispTransformer(nn.Module):
    """
    Input:
      x: [B, C_in, H]  (C_in=4: depth, Vp, Vs, rho)
    Output:
      mu, logvar: [B, 2, T]  (2: Rayleigh, Love)

    Notes:
      - T fixed to 59 by construction (pass T=59)
      - periods optional: [T] or [B,T]
    """
    def __init__(
        self,
        H: int,
        T: int = 59,          # <= 这里默认就是 59
        C_in: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 3,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        use_period_values: bool = True,
        period_minmax: tuple = (1.0, 100.0),  # 你的周期范围（用于归一化）
        logvar_clip: tuple = (-10.0, 3.0),    # 防止 logvar 发散
    ):
        super().__init__()
        self.H = H
        self.T = T
        self.C_in = C_in
        self.d_model = d_model
        self.use_period_values = use_period_values
        self.pmin, self.pmax = period_minmax
        self.logvar_clip = logvar_clip

        # ---- depth token embedding ----
        self.in_proj = nn.Sequential(
            nn.Linear(C_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ---- depth positional encoding ----
        self.depth_pos = SinusoidalPosEmb(d_model, max_len=max(4096, H))

        # ---- encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ---- T=59 period queries ----
        self.period_query = nn.Parameter(torch.randn(T, d_model) * 0.02)

        if use_period_values:
            self.period_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # ---- decoder (cross-attn) ----
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.final_norm = nn.LayerNorm(d_model)

        # ---- heads: per-period token -> 2 values (Rayleigh, Love) ----
        self.out_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.out_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor, periods: torch.Tensor = None):
        """
        x: [B, C_in, H]
        periods: optional [T] or [B, T] (seconds)
        """
        B, C, H = x.shape
        assert C == self.C_in and H == self.H, f"expect x=[B,{self.C_in},{self.H}] but got {x.shape}"
        assert self.T == 59 or True, "T is configurable, but you said T=59"

        # [B,H,C_in]
        xt = x.transpose(1, 2).contiguous()

        # token embed
        h = self.in_proj(xt)  # [B,H,d]

        # add depth pos
        h = h + self.depth_pos(H).unsqueeze(0)  # [B,H,d]

        # encoder memory
        memory = self.encoder(h)  # [B,H,d]

        # period query tokens: [B,T,d]
        q = self.period_query.unsqueeze(0).expand(B, -1, -1)

        if self.use_period_values:
            if periods is None:
                # 若你周期固定但没显式传入，就按 minmax 线性生成
                p = torch.linspace(self.pmin, self.pmax, self.T, device=x.device).unsqueeze(0).expand(B, -1)
            else:
                p = periods
                if p.ndim == 1:
                    assert p.numel() == self.T, f"periods length {p.numel()} != T={self.T}"
                    p = p.unsqueeze(0).expand(B, -1)
                else:
                    assert p.shape[1] == self.T, f"periods shape {p.shape} mismatch T={self.T}"

            # normalize to [-1,1]
            pn = (p - self.pmin) / max(1e-6, (self.pmax - self.pmin))
            pn = pn * 2.0 - 1.0
            q = q + self.period_mlp(pn.unsqueeze(-1))  # [B,T,d]

        # cross-attn decode
        y = self.decoder(tgt=q, memory=memory)  # [B,T,d]
        y = self.final_norm(y)

        mu = self.out_mu(y)         # [B,T,2]
        logvar = self.out_logvar(y) # [B,T,2]

        # reshape -> [B,2,T]
        mu = mu.permute(0, 2, 1).contiguous()
        logvar = logvar.permute(0, 2, 1).contiguous()

        # stabilize logvar
        lo, hi = self.logvar_clip
        logvar = logvar.clamp(min=lo, max=hi)

        return mu, logvar
