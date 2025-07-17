import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class BUSImageDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Loads an image/mask from paths specified in a DataFrame row.
    """
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

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        label = self.label_map.get(str(row['label']).lower(), -1)
        
        return {
            "image": image,
            "mask": mask if mask is not None else torch.empty(0),
            "label": torch.tensor(label, dtype=torch.long),
            "image_path": row['image_path'],
            "idx": row.get("idx", idx),
            "metadata": row.get("metadata", {})
        }

class BUSDataLoader(DataLoader):
    """
    PyTorch DataLoader Factory. Inherits from the base DataLoader.
    It filters a dataframe by a specific split ('train', 'test', or a fold number) 
    and creates a DataLoader instance for it.
    """
    def __init__(self, df, batch_size, split='1', shuffle=True, num_workers=0, 
                 transform=None, label_map=None):
        
        # Filter dataframe for the specified split
        df_split = df[df['split'] == str(split)].copy()
        
        dataset = BUSImageDataset(df_split, transform=transform, label_map=label_map)
        
        # Initialize the parent DataLoader. Removed 'test_split' and 'validation_split=0.0' 
        # as they are not valid PyTorch DataLoader arguments.
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


