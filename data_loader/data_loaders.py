# data_loader/data_loaders.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import Compose
# Assume these are in data.preprocessing, as in the original code
from data.preprocessing.preprocess import get_preprocessing_transform 
from data.preprocessing.augmentation import get_augmentation_transform

class BUSImageDataset(Dataset):
    def __init__(self, df, transform=None, label_map=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {'benign': 0, 'malignant': 1, 'normal': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        mask = None
        if pd.notnull(row.get('mask_path', None)):
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

            # --- FIX: Ensure mask and image have the same dimensions ---
            # This handles inconsistencies in the source dataset files.
            if mask is not None and mask.shape != image.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            # For masks that don't exist, pass a placeholder to albumentations
            augmented = self.transform(image=image, mask=mask if mask is not None else image)
            image = augmented['image']
            if mask is not None:
                mask = augmented['mask']
        
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            mask = (mask > 0.5).float() 

        label = self.label_map.get(str(row['label']).lower(), -1)
        
        return {
            "image": image,
            "mask": mask if mask is not None else torch.empty(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

class BUSDataLoader(DataLoader):
    """PyTorch DataLoader Factory for Breast US images."""
    def __init__(
        self, df, batch_size, split='1', is_test=False, num_workers=0, 
        label_map=None, target_size=512, padding_color=0, augment=True
    ):
        # Handle test mode automatically
        if is_test:
            shuffle = False
            augment = False
        else:
            shuffle = True

        # Simplified transform composition
        transforms_list = [get_preprocessing_transform(target_size, padding_color)]
        if augment:
            transforms_list.append(get_augmentation_transform())
        transform = Compose(transforms_list, additional_targets={'mask': 'mask'})
        
        # Filter dataframe for the specified split
        if not is_test:
            # In k-fold, the train split contains all data EXCEPT the current validation fold and the test set
            df_split = df[(df['split'] != str(split)) & (df['split'] != 'test')].copy()
        else: 
             # For validation or test loader, we just need the specified split
             df_split = df[df['split'] == str(split)].copy()
        
        dataset = BUSImageDataset(df_split, transform=transform, label_map=label_map)
        
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, pin_memory=True
        )


