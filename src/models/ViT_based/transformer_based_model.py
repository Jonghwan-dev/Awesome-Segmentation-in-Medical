from .transUnet.transunet import TransUnet
from .swinUnet.vision_transformer import SwinUnet
from .swinUnet.config import get_config
from .medicalT.axialnet import MedT

def get_transformer_based_model(model_name: str, config, num_classes: int):
    """
    Creates a transformer-based model instance based on a config object.
    The config object is the result of argparse from train.py.
    """
    if model_name == "MedT":
        # MedT expects img_size and in_ch from config
        model = MedT(img_size=config.target_size, imgchan=1, num_classes=num_classes)
    
    elif model_name == "SwinUnet":
        # get_config now uses the pre-parsed 'config' object
        swin_config = get_config(config)
        model = SwinUnet(config=swin_config, img_size=config.target_size, num_classes=num_classes)
        
    elif model_name == "TransUnet":
        # TransUnet expects img_ch and output_ch
        model = TransUnet(img_ch=1, output_ch=num_classes)
        
    else:
        raise ValueError(f"Transformer model '{model_name}' not recognized.")
        
    return model