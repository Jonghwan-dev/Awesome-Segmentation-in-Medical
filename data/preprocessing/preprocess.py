#data/preprocessing/preprocess.py
import albumentations as A
import cv2

def get_preprocessing_transform(target_size=512, padding_color=0):
    """
    target_size: int, desired output size (e.g., 224, 256, 448, 512)
    padding_color: int or (int, int, int), value for padding
    """
    def get_transform():
        return A.Compose([
            # 1. Pad to make the longest side equal to target_size (keep aspect ratio)
            A.LongestMaxSize(max_size=target_size, interpolation=cv2.INTER_LANCZOS4),
            # 2. Pad remaining area to reach target_size
            A.PadIfNeeded(
                min_height=target_size, min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT, position='center' 
            ),
        ], additional_targets={'mask': 'mask'})
    return get_transform() 