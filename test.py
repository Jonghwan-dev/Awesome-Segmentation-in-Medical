# test.py
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile
from pathlib import Path
import collections

# --- Import project modules ---
from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.utils.parse_config import ConfigParser
from src.utils.metrics import pixel_accuracy, dice_score, hd95_batch, iou_score
from src.utils.util import MetricTracker
# Import all model modules
import src.models.cnn_based as cnn_models
import src.models.ViT_based as transformer_models

def main(config):
    """Main function to run the evaluation pipeline."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- Prepare Data ---
    data_config = config['data']
    preparer = PrepareDataset()
    all_dfs = []
    # Use the specific dataset list from the config for testing
    for name in data_config['datasets']:
        csv_path = preparer.data_dir / f"{name}.csv"
        if csv_path.exists(): all_dfs.append(pd.read_csv(csv_path))
        else: raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Create a config dict specifically for the DataLoader
    loader_args = {
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'target_size': config['data']['target_size']
    }
    
    test_loader = BUSDataLoader(df, **loader_args, split='test', is_test=True)
    print(f"Test dataset loaded with {len(test_loader.dataset)} samples.")

    # --- Initialize Model and Load Weights ---
    model_type = config['arch']['type']
    if hasattr(cnn_models, model_type):
        model = config.init_obj('arch', cnn_models)
    elif hasattr(transformer_models, model_type):
        model = transformer_models.get_transformer_based_model(
            model_name=model_type,
            config=argparse.Namespace(**{**config.config, **config['swin_unet_args']}),
            num_classes=1
        )
    else:
        raise ValueError(f"Model type '{model_type}' not found.")
    
    print("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # --- Initialize Metrics ---
    test_metrics = MetricTracker('PA', 'DSC', 'HD95', 'IoU')

    # --- Evaluation Loop ---
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            image, target = batch['image'].to(device), batch['mask'].to(device)
            output = model(image)
            pred_sigmoid = torch.sigmoid(output)
            
            test_metrics.update('PA', pixel_accuracy(pred_sigmoid, target), n=image.size(0))
            test_metrics.update('DSC', dice_score(pred_sigmoid, target), n=image.size(0))
            test_metrics.update('HD95', hd95_batch(pred_sigmoid, target), n=image.size(0))
            test_metrics.update('IoU', iou_score(pred_sigmoid, target), n=image.size(0))

    # --- Calculate FLOPs and Parameters ---
    dummy_input = torch.randn(1, 1, data_config['target_size'], data_config['target_size']).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # --- Print Final Results in a parsable format ---
    result = test_metrics.result()
    print("\n--- Test Results ---")
    for key, value in result.items():
        print(f"{key}:{value:.4f}")
    print(f"GFLOPs:{flops / 1e9:.2f}")
    print(f"Params:{params / 1e6:.2f}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation Testing')
    args.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint to test')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    
    # --- FIX: Parse known args first to get the resume path, then add the config path ---
    # This ensures the correct config path is available before ConfigParser is called.
    parsed_args, _ = args.parse_known_args()
    
    # Example: checkpoints/busi_UNet_fold1_best.pth -> busi_UNet
    resume_path = Path(parsed_args.resume)
    ckpt_filename = resume_path.name
    exper_name = ckpt_filename.split('_fold')[0]
    
    # Example: checkpoints/busi_UNet_config.json
    cfg_path = resume_path.parent / f"{exper_name}_config.json"
    
    args.add_argument('-c', '--config', default=str(cfg_path), type=str)

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target kwargs')
    options = [
        CustomArgs(['--bs'], type=int, target='data;batch_size', kwargs=None),
    ]

    config = ConfigParser.from_args(args, options)
    main(config)