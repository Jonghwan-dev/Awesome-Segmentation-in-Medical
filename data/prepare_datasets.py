# prepare_datasets.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import Image
import os
import json # For handling metadata serialization

class PrepareDataset:
    def __init__(self, project_root=None, random_state=42):
        # Use current working directory if project_root is not specified.
        if project_root is None:
            project_root = Path.cwd()
        
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data" 
        self.random_state = random_state
        
        # --- Centralized path management for all datasets ---
        self.dataset_paths = {
            "busi": {
                "args": (self.project_root / "datasets/BreastUS/BUSI",),
                "reader": self.read_data_BUSI,
                "out_prefix": "busi"
            },
            "yap": {
                "args": (
                    self.project_root / "datasets/BreastUS/Yap2018/original",
                    self.project_root / "datasets/BreastUS/Yap2018/GT",
                    self.project_root / "datasets/BreastUS/Yap2018/DatasetB.xlsx"
                ),
                "reader": self.read_data_Yap,
                "out_prefix": "yap2018"
            },
            "uc": {
                "args": (self.project_root / "datasets/BreastUS/BUS_UC",),
                "reader": self.read_data_UC,
                "out_prefix": "bus_uc"
            },
            "uclm": {
                "args": (
                    self.project_root / "datasets/BreastUS/BUS-UCLM/INFO.csv",
                    self.project_root / "datasets/BreastUS/BUS-UCLM/images",
                    self.project_root / "datasets/BreastUS/BUS-UCLM/masks",
                    self.project_root / "datasets/BreastUS/BUS-UCLM/Masks_white"
                ),
                "reader": self.read_data_UCLM,
                "out_prefix": "bus_uclm"
            },
            "busbra": {
                "args": (
                    self.project_root / "datasets/BreastUS/BUSBRA/bus_data.csv",
                    self.project_root / "datasets/BreastUS/BUSBRA/Images",
                    self.project_root / "datasets/BreastUS/BUSBRA/Masks"
                ),
                "reader": self.read_data_BUSBRA,
                "out_prefix": "busbra"
            }
        }
        
        self.merged_mask_dir = self.project_root / "datasets/BreastUS/BUSI/BUSI_mask_merged"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.merged_mask_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def save_mask_white(mask_path, save_path, color_map):
        # Converts a given color mask to a binary (black/white) mask.
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask_white = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color, value in color_map.items():
            mask_white[(mask[..., :3] == color).all(axis=-1)] = value
        Image.fromarray(mask_white).save(save_path)
    
    # --- Readers for each dataset ---

    def read_data_BUSI(self, dataset_name, root_dir):
        # Reads BUSI dataset, merging multiple masks into one.
        root, data = Path(root_dir), []
        for label_folder in ["benign", "malignant", "normal"]:
            for img_path in (root / label_folder).glob("*.png"):
                if "_mask" in img_path.stem: continue
                mask_paths = sorted((root / label_folder).glob(f"{img_path.stem}_mask*.png"))
                if not mask_paths: continue
                merged_mask_path = self.merged_mask_dir / f"{img_path.stem}_merged.png"
                if not merged_mask_path.exists():
                    base = np.array(Image.open(mask_paths[0]).convert("L"))
                    for p in mask_paths[1:]: base = np.maximum(base, np.array(Image.open(p).convert("L")))
                    Image.fromarray(base).save(merged_mask_path)
                with Image.open(img_path) as im: width, height = im.size
                data.append({
                    "dataset": dataset_name, "label": label_folder,
                    "width": width, "height": height, "image_path": str(img_path),
                    "mask_path": str(merged_mask_path), "metadata": {"merged_mask_count": len(mask_paths)}
                })
        return pd.DataFrame(data)

    def read_data_Yap(self, dataset_name, image_dir, mask_dir, excel_path):
        # Reads Yap2018 dataset from images and an Excel file.
        excel_df = pd.read_excel(excel_path)
        excel_df.columns = [col.strip().lower() for col in excel_df.columns]
        data = []
        for img_path in Path(image_dir).glob("*.png"):
            mask_path = Path(mask_dir) / img_path.name
            if not mask_path.exists(): continue
            match = excel_df[excel_df["image"].astype(str).str.zfill(6) == img_path.stem]
            label = match.iloc[0].get("type", "unknown").lower() if not match.empty else "unknown"
            meta = match.iloc[0].to_dict() if not match.empty else {}
            if "image" in meta: meta.pop("image")
            with Image.open(img_path) as im: width, height = im.size
            data.append({
                "dataset": dataset_name, "label": label,
                "width": width, "height": height, "image_path": str(img_path),
                "mask_path": str(mask_path), "metadata": meta
            })
        return pd.DataFrame(data)

    def read_data_UC(self, dataset_name, root_dir):
        # Reads BUS_UC dataset structured in subfolders.
        root, data = Path(root_dir), []
        for label_folder in ["Benign", "Malignant"]:
            for img_path in (root / label_folder / "images").glob("*.png"):
                mask_path = (root / label_folder / "masks") / img_path.name
                if not mask_path.exists(): continue
                with Image.open(img_path) as im: width, height = im.size
                data.append({
                    "dataset": dataset_name, "label": label_folder.lower(),
                    "width": width, "height": height, "image_path": str(img_path),
                    "mask_path": str(mask_path), "metadata": {}
                })
        return pd.DataFrame(data)

    def read_data_UCLM(self, dataset_name, info_csv_path, image_dir, mask_dir, out_mask_dir):
        # Reads BUS-UCLM, correctly constructs file paths.
        info_df = pd.read_csv(info_csv_path, delimiter=';')
        Path(out_mask_dir).mkdir(exist_ok=True, parents=True)
        color_map = {(0, 255, 0): 255, (255, 0, 0): 255}
        data = []
        
        for _, row in info_df.iterrows():
            image_name = row['Image']
            # --- FIX: Construct the full path from the image_dir ---
            full_image_path = Path(image_dir) / image_name
            
            raw_mask_path = Path(mask_dir) / image_name
            white_mask_path = Path(out_mask_dir) / image_name
            
            if not white_mask_path.exists() and raw_mask_path.exists():
                self.save_mask_white(raw_mask_path, white_mask_path, color_map)

            wh = str(row['Resolution']).split('x')
            width, height = (int(wh[0]), int(wh[1])) if len(wh) == 2 else (0, 0)
            
            metadata = {
                'Doppler': row['Doppler'], 'Marks': row['Marks'], 'Combined': row['Combined']
            }
            
            data.append({
                "dataset": dataset_name, "label": str(row['Label']).lower(),
                "width": width, "height": height,
                "image_path": str(full_image_path), # Use the corrected path
                "mask_path": str(white_mask_path) if white_mask_path.exists() else None,
                "metadata": metadata
            })
        return pd.DataFrame(data)

    def read_data_BUSBRA(self, dataset_name, info_csv_path, image_dir, mask_dir):
        info_df = pd.read_csv(info_csv_path)
        data = []
        
        for _, row in info_df.iterrows():
            image_id = row['ID'] # e.g., "bus_0001-l"
            
            # Construct image path
            image_name = f"{image_id}.png"
            full_image_path = Path(image_dir) / image_name

            # --- FIX: Correct mask filename logic ---
            # 'bus_0001-l' -> '0001-l'
            base_name = image_id.replace('bus_', '')
            # '0001-l' -> 'mask_0001-l.png'
            mask_name = f"mask_{base_name}.png"
            full_mask_path = Path(mask_dir) / mask_name
            # -----------------------------------------

            metadata_keys = ['Case', 'Histology', 'BIRADS', 'Device', 'Side', 'BBOX']
            metadata = {key: row[key] for key in metadata_keys if key in row}

            data.append({
                "dataset": dataset_name, "label": str(row['Pathology']).lower(),
                "width": row['Width'], "height": row['Height'],
                "image_path": str(full_image_path),
                "mask_path": str(full_mask_path) if full_mask_path.exists() else None,
                "metadata": metadata
            })
        return pd.DataFrame(data)

    def split_and_save(self, df, out_path, n_folds=5, test_size=0.2):
        # This method remains unchanged.
        df.reset_index(drop=True, inplace=True)
        df['idx'] = df.index
        df['metadata'] = df['metadata'].apply(json.dumps)
        train_idx, test_idx = train_test_split(df.index, test_size=test_size, stratify=df['label'], random_state=self.random_state)
        df.loc[train_idx, 'split'] = 'train'
        df.loc[test_idx, 'split'] = 'test'
        
        train_df_idx = df[df['split'] == 'train'].index
        train_labels = df.loc[train_df_idx, 'label']
        
        if len(np.unique(train_labels)) > 1:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_df_idx, train_labels), 1):
                df.loc[train_df_idx[val_fold_idx], 'split'] = str(fold)
        
        final_cols = ["idx", "dataset", "label", "split", "width", "height", "image_path", "mask_path", "metadata"]
        df_final = pd.DataFrame(columns=final_cols)
        df_final = pd.concat([df_final, df[final_cols]], ignore_index=True)
        
        df_final.to_csv(out_path, index=False)
        print(f"  -> Saved data to {out_path}")

    def run(self, dataset_list):
        """Processes and saves each dataset as a separate CSV file."""
        # --- FIX: Loop through datasets and save each one individually ---
        for name in dataset_list:
            if name in self.dataset_paths:
                print(f"Processing dataset: '{name}'...")
                config = self.dataset_paths[name]
                df = config["reader"](name, *config["args"])
                
                if not df.empty:
                    # Define output path based on dataset name (e.g., data/busi.csv)
                    output_path = self.data_dir / f"{name}.csv"
                    self.split_and_save(df, output_path, n_folds=5, test_size=0.2)
            else:
                print(f"Warning: Dataset '{name}' is not defined. Skipping.")

