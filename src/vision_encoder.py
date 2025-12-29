class SwinEncoder(nn.Module):
    """
    Vision encoder based on a pre-trained Swin Transformer.
    """
    def __init__(self, embed_dim=768, model_name='swin_b'):
        super(SwinEncoder, self).__init__()

        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)

        in_features = self.swin.head.in_features

        self.swin.head = nn.Identity()

        self.projector = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.swin(x)
        return self.projector(features)

