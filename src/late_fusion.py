class BioFuseLate(nn.Module):
    def __init__(self, embed_dim=768, num_classes=14):
        super(BioFuseLate, self).__init__()
        self.image_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        
        # Independent classifiers for each modality
        self.img_classifier = nn.Linear(embed_dim, num_classes)
        self.txt_classifier = nn.Linear(embed_dim, num_classes)
        
        # Learnable fusion weight α initialized at 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask):
        # Encode each modality
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)

        # Individual predictions
        img_logits = self.img_classifier(img_feat)
        txt_logits = self.txt_classifier(txt_feat)

        # Late fusion using learnable α ∈ (0,1)
        alpha = self.sigmoid(self.alpha)  # ensures it's between 0 and 1
        fused_logits = alpha * img_logits + (1 - alpha) * txt_logits

        return fused_logits

