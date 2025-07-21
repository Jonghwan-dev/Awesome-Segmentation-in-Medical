# transUnet/transunet.py
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path

# --- FIX START: Address 'load_state_dict_from_url' not defined error ---
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from . import vit_seg_modeling
vit_seg_modeling.load_state_dict_from_url = load_state_dict_from_url
# --- FIX END ---

from .vit_seg_modeling import VisionTransformer as ViT_seg, CONFIGS as CONFIGS_ViT_seg

class TransUnet(nn.Module):
    def __init__(self, img_size=224, img_ch=1, output_ch=1):
        super(TransUnet, self).__init__()
        
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = output_ch
        config_vit.n_skip = 3
        
        if (img_size % 16) != 0:
            raise ValueError("img_size must be divisible by 16 for TransUnet.")
        config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))

        self.net = ViT_seg(config_vit, img_size=img_size, num_classes=output_ch)
        
        # --- FIX START: Correct the attribute path to the embedding layer ---
        # The embedding layer is inside the 'transformer' attribute of the ViT_seg model.
        if img_ch == 1 and self.net.transformer.embeddings.patch_embeddings.in_channels == 3:
            # Get original weights from the 3-channel convolution
            original_weights = self.net.transformer.embeddings.patch_embeddings.weight.data
            
            # Create a new convolution layer for 1-channel input
            new_first_conv = nn.Conv2d(1, config_vit.hidden_size, kernel_size=(16, 16), stride=(16, 16))
            
            # Average the weights across the channel dimension and assign to the new layer
            new_first_conv.weight.data = original_weights.mean(dim=1, keepdim=True)
            
            # Replace the original patch embedding layer
            self.net.transformer.embeddings.patch_embeddings = new_first_conv
        # --- FIX END ---

        # Add encoder attribute for freezing logic
        self.encoder = self.net.transformer

    def forward(self, x):
        return self.net(x)

    def load_from(self, weights_npz_path):
        """Loads weights from a local .npz file for the ViT backbone."""
        if weights_npz_path and Path(weights_npz_path).exists():
            print(f"Loading pre-trained weights for ViT backbone from .npz file: {weights_npz_path}")
            self.net.load_from(weights=np.load(weights_npz_path))
        else:
            print("Warning: No local .npz weights provided or path is invalid. Using default pre-trained weights if available.")
