import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

class VIT(nn.Module):
    def __init__(self, pretrained=True, device='cpu'):
        """
        Feature extractor using Vision Transformer (ViT-B/16).
        Args:
            pretrained (bool): Whether to load pretrained weights.
            device (str): Device to load the model ('cpu' or 'cuda').
        """
        super(VIT, self).__init__()
        self.device = torch.device(device)
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT 
                                   if pretrained else None).to(self.device)

    def forward(self, x):
        """
        Extract features from the CLS token.
        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, Channels, Height, Width).
        Returns:
            torch.Tensor: Feature vector of shape (Batch, 768).
        """
        feats = self.vit._process_input(x)
        
        batch_class_token = self.vit.class_token.expand(feats.shape[0], -1, -1).to(self.device)
        feats = torch.cat([batch_class_token, feats], dim=1)
        
        feats = self.vit.encoder(feats)
        return feats[:, 0]