import argparse
import numpy as np
from pathlib import Path

# Import the refactored model classes from their respective files
from .transUnet.transunet import TransUnet
from .swinUnet.vision_transformer import SwinUnet
from .swinUnet.config import get_config as get_swin_config
from .medicalT.axialnet import MedT

def get_transformer_based_model(model_name: str, config: dict, num_classes: int = 1):
    """
    Factory function to create a transformer-based model instance.
    'config' is the main config dictionary from ConfigParser.
    """
    data_config = config.get('data', {})
    img_size = data_config.get('target_size')
    in_channels = 1 # Grayscale images

    if model_name == "MedT":
        model = MedT(img_size=img_size, imgchan=in_channels, num_classes=num_classes)
    
    elif model_name == "SwinUnet":
        swin_unet_config_section = config.get('swin_unet_args', {})
        
        relative_yaml_path = swin_unet_config_section.get('cfg')
        if not relative_yaml_path:
            raise ValueError("SwinUnet 'cfg' path is not specified in config.json")
        project_root = Path(__file__).parent.parent.parent.parent
        absolute_yaml_path = project_root / relative_yaml_path.lstrip('./')
        if not absolute_yaml_path.exists():
            raise FileNotFoundError(f"SwinUnet config YAML not found at resolved path: {absolute_yaml_path}")

        swin_args = argparse.Namespace(
            batch_size=data_config.get('batch_size'),
            zip=swin_unet_config_section.get('zip', False),
            cache_mode=swin_unet_config_section.get('cache_mode', 'part'),
            resume=swin_unet_config_section.get('resume'),
            accumulation_steps=swin_unet_config_section.get('accumulation-steps'),
            use_checkpoint=swin_unet_config_section.get('use-checkpoint'),
            amp_opt_level=None,
            tag='default',
            eval=False,
            throughput=False,
            opts=swin_unet_config_section.get('opts'),
            cfg=str(absolute_yaml_path)
        )
        
        swin_config_obj = get_swin_config(swin_args)
        
        swin_config_obj.defrost()
        swin_config_obj.MODEL.SWIN.IN_CHANS = in_channels
        
        # --- FIX: Read PRETRAIN_CKPT from config.json and set it in swin_config_obj ---
        pretrained_path_str = swin_unet_config_section.get('PRETRAIN_CKPT')
        if pretrained_path_str:
            absolute_pretrained_path = project_root / pretrained_path_str.lstrip('./')
            if absolute_pretrained_path.exists():
                swin_config_obj.MODEL.PRETRAIN_CKPT = str(absolute_pretrained_path)
            else:
                print(f"Warning: Pretrained model path specified but not found: {absolute_pretrained_path}")
        # --- END FIX ---
        
        swin_config_obj.freeze()

        model = SwinUnet(config=swin_config_obj, img_size=img_size, num_classes=num_classes)
        
        # The load_from method will now be called because PRETRAIN_CKPT is set
        if swin_config_obj.MODEL.PRETRAIN_CKPT:
            model.load_from(swin_config_obj)
            
    elif model_name == "TransUnet":
        model = TransUnet(img_size=img_size, img_ch=in_channels, output_ch=num_classes)
        
        vit_args = config.get('vit_seg_args', {})
        if vit_args.get('vit_patches_path'):
            model.load_from(weights_npz_path=vit_args['vit_patches_path'])

    else:
        raise ValueError(f"Transformer model '{model_name}' not recognized.")
        
    return model