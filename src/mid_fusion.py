import torch
import torch.nn as nn
from vision_encoder import SwinEncoder
from text_encoder import TextEncoder
from Co_Attention import CoAttentionFusion

class BioFuse(nn.Module):
    def __init__(self, embed_dim=768, num_classes=14):
        super().__init__()
        self.image_encoder = SwinEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.fusion = CoAttentionFusion(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        fused_feat = self.fusion(img_feat, txt_feat)
        return self.classifier(fused_feat)



