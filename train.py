# train.py
import argparse
import collections
import pandas as pd
import torch
import numpy as np
import random
import wandb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)

from data.prepare_datasets import PrepareDataset
from data_loader.data_loaders import BUSDataLoader
from src.trainer.trainer import Trainer
import src.utils.losses as loss_module
import src.utils.metrics as metric_module 
from src.utils.parse_config import ConfigParser
from src.utils.util import count_params
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

def apply_freezing(model, freeze_mode):
    """Applies weight freezing to the model based on the specified mode."""
    if freeze_mode == 'none' or freeze_mode is None:
        print("No layers frozen. Training all parameters.")
        return model

    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == 'encoder':
        if hasattr(model, 'encoder'):
            print("Freezing ENCODER weights...")
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            print(f"Warning: freeze_mode is 'encoder' but model {type(model).__name__} has no 'encoder' attribute. Training all params.")
    
    return model

def main(config):
    set_seed(config['system']['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer_config = config['trainer']
    data_config = config['data']
    
    for fold in range(1, trainer_config['k_folds'] + 1):
        print(f"\n{'='*20} FOLD {fold}/{trainer_config['k_folds']} {'='*20}")
        
        run_name = f"{config['name']}_fold{fold}"
        
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

        if config.transfer_from and config.transfer_from.exists():
            print(f"\nLoading weights for transfer learning from: {config.transfer_from}")
            checkpoint = torch.load(config.transfer_from, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Weights loaded successfully.")

        freeze_mode = config['trainer'].get('freeze_mode', 'none')
        model = apply_freezing(model, freeze_mode)
        model = model.to(device)
        
        total_params = count_params(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

        if not config['wandb']['disable']:
            wandb.init(
                project=config['wandb']['project'], 
                name=run_name, 
                config=config.config,
                reinit=True
            )
            wandb.config.update({
                'trainable_params_M': round(trainable_params / 1e6, 2),
                'freeze_mode': freeze_mode
            }, allow_val_change=True)

        trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_model_params)
        
        criterion = config.init_obj('loss', loss_module)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        metrics = [getattr(metric_module, met) for met in config['metrics']]

        if not config['wandb']['disable']:
            wandb.watch(model, criterion, log="all", log_freq=100)

        # --- FIX: Restore logic to flatten trainer config for the Trainer class ---
        trainer_run_config = config.config.copy()
        trainer_run_config['checkpoint_name'] = f"{run_name}_best.pth"
        # This loop ensures keys like 'epochs' are at the top level of the config dict
        for key, value in config['trainer'].items():
            trainer_run_config[key] = value
        # --- END FIX ---
        
        trainer = Trainer(
            model=model, criterion=criterion, metrics=metrics, optimizer=optimizer,
            config=trainer_run_config, device=device, train_loader=train_loader,
            val_loader=val_loader, lr_scheduler=lr_scheduler
        )
        
        if config.resume:
            trainer.resume_checkpoint(config.resume)

        try:
            trainer.train()
        finally:
            if not config['wandb']['disable']:
                wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Breast Ultrasound Segmentation')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--transfer-from', default=None, type=str, help='path to checkpoint for transfer learning (loads weights only)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target kwargs')
    options = [
        CustomArgs(['--name'], str, 'name', {}),
        CustomArgs(['--model'], str, 'arch;type', {}),
        CustomArgs(['--datasets'], str, 'data;datasets', {'nargs': '+'}),
        CustomArgs(['--bs'], int, 'data;batch_size', {}),
        CustomArgs(['--size'], int, 'data;target_size', {}),
        CustomArgs(['--lr'], float, 'optimizer;args;lr', {}),
        CustomArgs(['--wd'], float, 'optimizer;args;weight_decay', {}),
        CustomArgs(['--loss'], str, 'loss;type', {}),
        CustomArgs(['--scheduler'], str, 'lr_scheduler;type', {}),
        CustomArgs(['--epochs'], int, 'trainer;epochs', {}),
        CustomArgs(['--patience'], int, 'trainer;early_stopping_patience', {}),
        CustomArgs(['--freeze-mode'], str, 'trainer;freeze_mode', {'choices': ['none', 'encoder'], 'default': 'none'}),
    ]
    
    config = ConfigParser.from_args(parser, options)
    main(config)
