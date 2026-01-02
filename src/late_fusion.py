import torch
import torch.nn as nn
from vision_encoder import SwinEncoder
from text_encoder import TextEncoder

class BioFuseLate(nn.Module):
    def __init__(self, embed_dim=768, num_classes=14):
        super(BioFuseLate, self).__init__()
        self.image_encoder = SwinEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.img_classifier = nn.Linear(embed_dim, num_classes)
        self.txt_classifier = nn.Linear(embed_dim, num_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        img_logits = self.img_classifier(img_feat)
        txt_logits = self.txt_classifier(txt_feat)
        alpha = self.sigmoid(self.alpha)
        fused_logits = alpha * img_logits + (1 - alpha) * txt_logits
        return fused_logits

