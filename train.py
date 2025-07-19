# train.py
import argparse
import collections
import pandas as pd
import torch
import numpy as np
import random
import wandb

# --- Import project modules ---
from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.trainer.trainer import Trainer
import src.utils.losses as loss_module
import src.utils.metrics as metric_module 
from src.utils.parse_config import ConfigParser
# Import all model modules
import src.models.cnn_based as cnn_models
import src.models.ViT_based as transformer_models


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config):
    """Main function to run the training pipeline using a config object."""
    set_seed(config['system']['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- K-Fold Cross-Validation Loop ---
    trainer_config = config['trainer']
    data_config = config['data']
    
    for fold in range(1, trainer_config['k_folds'] + 1):
        print(f"\n{'='*20} FOLD {fold}/{trainer_config['k_folds']} {'='*20}")
        
        # --- Initialize a new wandb run for each fold ---
        run_name = f"{config['name']}_fold{fold}"
        if not config['wandb']['disable']:
            wandb.init(
                project=config['wandb']['project'], 
                name=run_name, 
                config=config.config,
                reinit=True # Allow re-initialization in the same process
            )
        
        # --- Prepare and Load Data for each fold ---
        preparer = PrepareDataset()
        if data_config['force_prepare']:
            print("Forcing dataset preparation...")
            preparer.run(dataset_list=data_config['datasets'])
        
        all_dfs = []
        for name in data_config['datasets']:
            csv_path = preparer.data_dir / f"{name}.csv"
            if csv_path.exists():
                all_dfs.append(pd.read_csv(csv_path))
            else:
                raise FileNotFoundError(f"{csv_path} not found. Please run with --force_prepare first.")
        df = pd.concat(all_dfs, ignore_index=True)

        loader_args = {
            'batch_size': data_config['batch_size'],
            'num_workers': data_config['num_workers'],
            'target_size': data_config['target_size']
        }

        train_loader = BUSDataLoader(df, **loader_args, split=str(fold), is_test=False, augment=True)
        df_val = df[df['split'] == str(fold)].copy()
        val_loader = BUSDataLoader(df_val, **loader_args, split=str(fold), is_test=True)

        # --- Initialize Components for the current fold ---
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
            raise ValueError(f"Model type '{model_type}' not found in any model module.")
            
        model = model.to(device)
        criterion = config.init_obj('loss', loss_module)
        optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        metrics = [getattr(metric_module, met) for met in config['metrics']]

        if not config['wandb']['disable']:
            wandb.watch(model, criterion, log="all", log_freq=100)

        # --- Setup and run Trainer ---
        trainer_run_config = config.config.copy()
        trainer_run_config['checkpoint_name'] = f"{run_name}_best.pth"
        
        for key, value in config['trainer'].items():
            trainer_run_config[key] = value
        
        trainer = Trainer(
            model=model, criterion=criterion, metrics=metrics, optimizer=optimizer,
            config=trainer_run_config, device=device, train_loader=train_loader,
            val_loader=val_loader, lr_scheduler=lr_scheduler
        )
        
        try:
            trainer.train()
        finally:
            if not config['wandb']['disable']:
                wandb.finish()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    
    # --- FIX: Modified CustomArgs to accept kwargs for additional argparse options ---
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target kwargs')
    options = [
        CustomArgs(['--lr'], type=float, target='optimizer;args;lr', kwargs=None),
        CustomArgs(['--bs'], type=int, target='data;batch_size', kwargs=None),
        CustomArgs(['--name'], type=str, target='name', kwargs=None),
        CustomArgs(['--datasets'], type=str, target='data;datasets', kwargs={'nargs': '+'}),
        CustomArgs(['--model'], type=str, target='arch;type', kwargs=None)
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
