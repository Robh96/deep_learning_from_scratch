import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_size=768, image_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embedding_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.wg = nn.Linear(d_model, hidden)  # gate
        self.w2 = nn.Linear(hidden, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        a = self.act(self.w1(x))
        g = self.wg(x)
        y = a * g
        y = self.dropout(y)
        y = self.w2(y)
        y = self.dropout(y)
        return y

class TransformerEncoderLayerSwiGLU(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        # Choose hidden ≈ 8/3 * d_model to match ~GELU-4x params/FLOPs
        hidden = int(round((8.0 / 3.0) * d_model))
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUFeedForward(d_model, hidden, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # norm-first
        y = self.norm1(x)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + self.dropout(y)
        y = self.norm2(x)
        y = self.ffn(y)
        x = x + y
        return x


class ViT(nn.Module):
    def __init__(self, ch=3, img_size=224, patch_size=16, embedding_size=768,
                 n_layers=6, num_heads=8, num_classes=10, dropout=0.1, use_swiglu: bool = False):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=ch, patch_size=patch_size, embedding_size=embedding_size, image_size=img_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_size) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

        if use_swiglu:
            encoder_layer = TransformerEncoderLayerSwiGLU(d_model=embedding_size, nhead=num_heads, dropout=dropout)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_size, nhead=num_heads,
                dim_feedforward=embedding_size * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
        # Silence nested-tensor warning when norm_first=True (no behavior change)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        x = self.patch_embedding(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)