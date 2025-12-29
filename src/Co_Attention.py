import torch
import torch.nn as nn

class CoAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()

        self.img_to_txt = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.txt_to_img = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat, txt_feat):
        txt_feat=txt_feat.mean(dim=1)
        img = img_feat.unsqueeze(1) 
        txt = txt_feat.unsqueeze(1)
        t2i, _ = self.txt_to_img(txt, img, img)

        i2t, _ = self.img_to_txt(img, txt, txt)

        fused = (t2i + i2t) / 2
        fused = fused.squeeze(1)
        return self.norm(self.proj(fused) + img_feat)


