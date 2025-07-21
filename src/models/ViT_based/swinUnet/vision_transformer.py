# swin_Unet/vision_transformer.py
import torch
import torch.nn as nn
import copy
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )
        
        self.encoder = self.swin_unet

    def forward(self, x):
        return self.swin_unet(x)

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print(f"Loading pretrained model from: {pretrained_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            if "model" in pretrained_dict:
                pretrained_dict = pretrained_dict['model']

            # --- FIX: Adapt the first convolutional layer for single-channel input ---
            # Get the weight of the first conv layer from the pretrained model
            pretrained_first_conv_weight = pretrained_dict.get('patch_embed.proj.weight')
            
            # Get the model's first conv layer's shape
            model_first_conv_shape = self.swin_unet.state_dict().get('patch_embed.proj.weight').shape

            # If shapes don't match (likely due to channel difference)
            if pretrained_first_conv_weight is not None and pretrained_first_conv_weight.shape != model_first_conv_shape:
                print(f"Adapting patch_embed.proj.weight for input channels: pretrained {pretrained_first_conv_weight.shape} -> model {model_first_conv_shape}")
                
                # If pretrained is 3 channels (RGB) and model is 1 channel (grayscale)
                if pretrained_first_conv_weight.shape[1] == 3 and model_first_conv_shape[1] == 1:
                    # Average the weights across the channel dimension
                    adapted_weight = pretrained_first_conv_weight.mean(dim=1, keepdim=True)
                    # Update the pretrained_dict with the adapted weight
                    pretrained_dict['patch_embed.proj.weight'] = adapted_weight
            # --- END FIX ---

            msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            
            print(f"Successfully loaded SwinUnet weights. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
        else:
            print("No pretrained weights specified for SwinUnet. Training from scratch.")
 