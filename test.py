# test.py
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)

from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.utils.parse_config import ConfigParser
from src.utils.metrics import pixel_accuracy, dice_score, hd95_batch, iou_score
import src.models.cnn_based as cnn_models
import src.models.ViT_based as transformer_models

def main(config):
    """Main function to run the evaluation pipeline."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare Data
    data_config = config['data']
    preparer = PrepareDataset()
    all_dfs = []
    for name in data_config['datasets']:
        csv_path = preparer.data_dir / f"{name}.csv"
        if csv_path.exists():
            all_dfs.append(pd.read_csv(csv_path))
        else:
            raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.concat(all_dfs, ignore_index=True)
    
    loader_args = {
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'target_size': config['data']['target_size']
    }
    
    test_loader = BUSDataLoader(df, **loader_args, split='test', is_test=True)
    print(f"Test dataset loaded with {len(test_loader.dataset)} samples.")

    # --- FIX: Update model loading logic to match train.py ---
    model_type = config['arch']['type']
    if hasattr(cnn_models, model_type):
        model = config.init_obj('arch', cnn_models)
    elif model_type in ["TransUnet", "SwinUnet", "MedT"]:
        model = transformer_models.get_transformer_based_model(
            model_name=model_type,
            config=config.config,
            num_classes=1
        )
    else:
        raise ValueError(f"Model type '{model_type}' not found in cnn_based or ViT_based models.")
    # --- END FIX ---
    
    print(f"Loading checkpoint: {config.resume} ...")
    checkpoint = torch.load(config.resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Initialize Metric Accumulators
    total_metrics_sum = defaultdict(float)
    total_metrics_count = defaultdict(int)

    # Evaluation Loop
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            image, target = batch['image'].to(device), batch['mask'].to(device)
            output = model(image)
            pred_sigmoid = torch.sigmoid(output)
            
            dsc_sum, count = dice_score(pred_sigmoid, target)
            hd95_sum, hd95_n_valid = hd95_batch(pred_sigmoid, target)
            iou_sum, _ = iou_score(pred_sigmoid, target)
            pa_correct, pa_total_pixels = pixel_accuracy(pred_sigmoid, target)
            
            total_metrics_sum['DSC'] += dsc_sum
            total_metrics_count['DSC'] += count
            
            total_metrics_sum['HD95'] += hd95_sum
            total_metrics_count['HD95'] += hd95_n_valid

            total_metrics_sum['IoU'] += iou_sum
            total_metrics_count['IoU'] += count

            total_metrics_sum['PA'] += pa_correct
            total_metrics_count['PA'] += pa_total_pixels
    
    # Calculate Final Averages
    final_results = {}
    for key in ['DSC', 'HD95', 'IoU', 'PA']:
        if total_metrics_count[key] > 0:
            final_results[key] = total_metrics_sum[key] / total_metrics_count[key]
        else:
            final_results[key] = 0.0

    # Calculate FLOPs and Parameters
    dummy_input = torch.randn(1, 1, data_config['target_size'], data_config['target_size']).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Print Final Results
    print("\n--- Test Results ---")
    for key, value in final_results.items():
        print(f"{key}: {value:.4f}")
    print(f"GFLOPs: {flops / 1e9:.2f}")
    print(f"Params: {params / 1e6:.2f}M")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation Testing')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint to test')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    
    parsed_args, _ = args.parse_known_args()
    
    resume_path = Path(parsed_args.resume)
    exper_name = resume_path.name.split('_fold')[0]
    cfg_path = resume_path.parent / f"{exper_name}_config.json"
    
    args.add_argument('-c', '--config', default=str(cfg_path), type=str)

    config = ConfigParser.from_args(args)
    main(config)