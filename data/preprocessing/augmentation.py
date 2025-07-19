import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

# Define a custom transform for Speckle Noise
class SpeckleNoise(A.ImageOnlyTransform):
    """
    Applies speckle noise, a common artifact in ultrasound images.
    It's a multiplicative noise: noisy_image = image + image * noise.
    """
    def __init__(self, mean=0, std_limit=(0.05, 0.15), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std_limit = std_limit

    def apply(self, img, **params):
        # Generate Gaussian noise with a random std deviation
        std = np.random.uniform(self.std_limit[0], self.std_limit[1])
        noise = np.random.normal(self.mean, std, img.shape)
        
        # Apply multiplicative noise and clip to preserve data range
        img_float = img.astype(np.float32)
        noisy_img = img_float + img_float * noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ('mean', 'std_limit')


def get_augmentation_transform():
    return A.Compose([
        # --- Geometric Augmentations ---
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.2),

        # --- Distortion Augmentations (apply one of the following) ---
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),
            A.OpticalDistortion(distort_limit=0.2, p=0.5)
        ], p=0.1),

        # --- Pixel-level Augmentations ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.CLAHE(clip_limit=1.2, tile_grid_size=(8, 8), p=0.2),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(p=0.2),

        # --- Noise/Filter Effects (apply one of the following) ---
        A.OneOf([
            SpeckleNoise(p=0.5), # Replaced GaussNoise with custom SpeckleNoise
            A.MotionBlur(blur_limit=3, p=0.5),
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=0.3),
        ], p=0.2),

    ], p=0.5, additional_targets={'mask': 'mask'})