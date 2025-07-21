import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import shutil
import argparse
from tqdm import tqdm

# Assuming the project structure allows these imports
from data.prepare_datasets import PrepareDataset
from data.preprocessing.preprocess import get_preprocessing_transform

# --- IMPORTANT ---
# You must import your actual generator model architectures here.
# For example, if you have a file `models/generators.py` with a `GeneratorUNet` class:
# from src.models.generators import GeneratorUNet 
#
# As a placeholder, we'll define a dummy class. Replace this with your actual import.
class PlaceholderGenerator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        print("Warning: Using a placeholder generator model. Please replace with your actual model.")
        return x

class PrepareSyntheticDataset:
    """
    Generates synthetic images from a source dataset using a pre-trained model,
    and creates corresponding CSV files for the new dataset.
    """
    def __init__(self, project_root, model_path, model_name, datasets_to_process, target_size=256):
        self.project_root = Path(project_root).resolve()
        self.source_data_dir = self.project_root / "data"
        self.synthetic_data_root = self.project_root / "datasets/synthetic_BreastUS"
        self.model_path = model_path
        self.model_name = model_name
        self.datasets_to_process = datasets_to_process
        self.target_size = target_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self._load_model()
        
        # Preprocessing transform for generator input
        self.transform = get_preprocessing_transform(target_size=self.target_size)

    def _load_model(self):
        """Loads the pre-trained generator model."""
        print(f"Loading generator model '{self.model_name}' from: {self.model_path}")
        
        # --- Replace this with your model instantiation logic ---
        # This is where you would select the model based on `self.model_name`
        # For example:
        # if self.model_name == 'GeneratorUNet':
        #     model = GeneratorUNet(in_channels=1, out_channels=1)
        # else:
        #     raise ValueError(f"Unknown model name: {self.model_name}")
        model = PlaceholderGenerator() # Using placeholder for now
        # ---------------------------------------------------------

        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        print("Generator model loaded successfully and set to evaluation mode.")
        return model

    def _generate_and_save(self, source_img_path, source_mask_path, out_img_path, out_mask_path):
        """
        Generates a single synthetic image, saves it, and copies the mask.
        """
        # Ensure output directories exist
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Preprocess the source image for the generator
        image_bgr = cv2.imread(str(source_img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=image_rgb)
        image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).unsqueeze(0)
        
        # Take only the first channel if it's grayscale but loaded as 3-channel
        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor[:, 0, :, :].unsqueeze(1)

        image_tensor = image_tensor.float().to(self.device) / 255.0

        # 2. Generate synthetic image
        with torch.no_grad():
            synthetic_output = self.generator(image_tensor)

        # 3. Post-process and save the synthetic image
        # Assuming model output is in range [0, 1] or [-1, 1]
        synthetic_output = synthetic_output.squeeze().cpu()
        synthetic_output = (synthetic_output.clamp(0, 1) * 255).byte()
        synthetic_image = Image.fromarray(synthetic_output.numpy(), mode='L')
        synthetic_image.save(out_img_path)

        # 4. Copy the original mask to the new destination
        if source_mask_path and Path(source_mask_path).exists():
            shutil.copy(source_mask_path, out_mask_path)

    def run(self):
        """Main loop to process all specified datasets."""
        # Use the original PrepareDataset to easily access source dataframes
        source_preparer = PrepareDataset(project_root=self.project_root)

        for dataset_name in self.datasets_to_process:
            print(f"\n--- Processing source dataset: {dataset_name} ---")
            
            # Define paths for the new synthetic dataset
            syn_dataset_name = f"syn_{dataset_name}"
            syn_img_dir = self.synthetic_data_root / syn_dataset_name / "images"
            syn_mask_dir = self.synthetic_data_root / syn_dataset_name / "masks"
            
            # Get the original dataframe
            source_csv_path = self.source_data_dir / f"{dataset_name}.csv"
            if not source_csv_path.exists():
                print(f"Warning: Source CSV not found at {source_csv_path}. Skipping.")
                continue
            source_df = pd.read_csv(source_csv_path)

            new_data_rows = []
            
            # Process each image in the source dataset
            for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc=f"Generating {syn_dataset_name}"):
                source_img_path = Path(row['image_path'])
                source_mask_path = Path(row['mask_path']) if pd.notnull(row['mask_path']) else None

                # Define new paths for synthetic data
                new_img_path = syn_img_dir / source_img_path.name
                new_mask_path = syn_mask_dir / f"syn_{source_mask_path.name}" if source_mask_path else None
                
                self._generate_and_save(source_img_path, source_mask_path, new_img_path, new_mask_path)

                # Collect info for the new CSV
                new_data_rows.append({
                    "idx": idx,
                    "dataset": syn_dataset_name,
                    "label": row['label'],
                    "image_path": str(new_img_path),
                    "mask_path": str(new_mask_path) if new_mask_path else None
                })
            
            # Save the new CSV file
            if new_data_rows:
                output_df = pd.DataFrame(new_data_rows)
                output_csv_path = self.source_data_dir / f"{syn_dataset_name}.csv"
                output_df.to_csv(output_csv_path, index=False)
                print(f"-> Successfully created synthetic dataset CSV at: {output_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Synthetic Breast US Datasets")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the pre-trained generator .pth file.")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the generator model class to instantiate.")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of source dataset names to process (e.g., busi bus_uc).")
    parser.add_argument('--project-root', type=str, default=None, help="Root directory of the project. Defaults to current dir.")
    
    args = parser.parse_args()

    # Instantiate and run the synthetic dataset preparation
    synthetic_preparer = PrepareSyntheticDataset(
        project_root=args.project_root,
        model_path=args.model_path,
        model_name=args.model_name,
        datasets_to_process=args.datasets
    )
    synthetic_preparer.run()