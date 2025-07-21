# src/trainer/trainer.py
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wandb
from torchvision.utils import make_grid

# Assume these utils and metrics are in the correct path
from src.utils.metrics import pixel_accuracy, dice_score, hd95_batch, iou_score
from src.utils.util import MetricTracker

class Trainer:
    def __init__(self, model, criterion, metrics, optimizer, config, device,
                 train_loader, val_loader=None, lr_scheduler=None):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.metric_fns = metrics
        
        train_metric_keys = ['loss', 'iou_score'] 
        self.train_metrics = MetricTracker(*train_metric_keys)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns])

        self.start_epoch = 1
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_name = config.get('checkpoint_name', 'best_model.pth')

        self.early_stopping_patience = config.get('early_stopping_patience', 30)
        self.early_stopping_counter = 0

        self.metric_history = [] # Store dicts of metrics for each epoch
        self.best_epoch_info = {} # Store info of the best epoch

        if self.val_loader:
            self.fixed_val_batch = next(iter(self.val_loader))

    def train(self):
        """Full training logic with dynamic ranking and early stopping."""
        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            train_log = self._train_epoch(epoch)
            
            log_dict = {'epoch': epoch, **train_log}
            
            if self.val_loader:
                val_log = self._valid_epoch(epoch)
                log_dict.update({f'val_{k}': v for k, v in val_log.items()})
                
                self._log_validation_images(epoch)
                
                is_new_best = self._update_and_check_best(val_log, epoch)
                
                if is_new_best:
                    self.early_stopping_counter = 0
                    self._save_checkpoint()
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Validation performance did not improve for {self.early_stopping_patience} epochs. Early stopping.")
                    print(f"Best model was from epoch {self.best_epoch_info.get('epoch', 'N/A')} with rank score {self.best_epoch_info.get('avg_rank', 'N/A'):.4f}")
                    break
            
            if wandb.run: wandb.log(log_dict, step=epoch)
            
            formatted_log = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in log_dict.items()}
            print(f"Epoch {epoch} Summary: {formatted_log}")

    def _update_and_check_best(self, val_log, current_epoch):
        """
        Appends current metrics to history, re-calculates ranks for all epochs,
        and determines if the current epoch is the new best.
        """
        current_metrics = {'epoch': current_epoch, **val_log}
        self.metric_history.append(current_metrics)
        
        history_df = pd.DataFrame(self.metric_history)
        
        history_df['loss_rank'] = history_df['loss'].rank(ascending=True, method='dense')
        history_df['hd95_batch_rank'] = history_df['hd95_batch'].rank(ascending=True, method='dense')
        history_df['iou_score_rank'] = history_df['iou_score'].rank(ascending=False, method='dense')
        history_df['dice_score_rank'] = history_df['dice_score'].rank(ascending=False, method='dense')
        
        rank_cols = ['loss_rank', 'hd95_batch_rank', 'iou_score_rank', 'dice_score_rank']
        history_df['avg_rank'] = history_df[rank_cols].mean(axis=1)
        
        best_epoch_idx = history_df['avg_rank'].idxmin()
        self.best_epoch_info = history_df.loc[best_epoch_idx].to_dict()
        
        if self.best_epoch_info['epoch'] == current_epoch:
            print(f"New best model found at epoch {current_epoch}!")
            print(f"  - Avg Rank: {self.best_epoch_info['avg_rank']:.4f}")
            print(f"  - Metrics: Loss={val_log['loss']:.4f}, HD95={val_log['hd95_batch']:.4f}, IoU={val_log['iou_score']:.4f}, Dice={val_log['dice_score']:.4f}")
            return True
        return False

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, sampled_batch in enumerate(progress_bar):
            image, target = sampled_batch['image'].to(self.device), sampled_batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item(), n=image.size(0))
            
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(output.detach())
                # iou_score ->(sum, count)
                iou_sum, iou_count = iou_score(pred_sigmoid, target)
                if iou_count > 0:
                    batch_avg = iou_sum / iou_count
                    # MetricTracker send to (key, batch_avg, batch_count)
                    self.train_metrics.update('iou_score', batch_avg, n=iou_count)

            
            progress_bar.set_postfix(loss=loss.item(), iou=self.train_metrics.avg('iou_score'))
            
        if self.lr_scheduler: self.lr_scheduler.step()
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        progress_bar = tqdm(self.val_loader, desc=f"Valid Epoch {epoch}")
        with torch.no_grad():
            for batch_idx, sampled_batch in enumerate(progress_bar):
                image, target = sampled_batch['image'].to(self.device), sampled_batch['mask'].to(self.device)
                
                output = self.model(image)
                loss = self.criterion(output, target.float())
                pred_sigmoid = torch.sigmoid(output)
                
                self.valid_metrics.update('loss', loss.item(), n=image.size(0))

                for met in self.metric_fns:
                    # each metric func -> (sum, count)
                    metric_sum, metric_count = met(pred_sigmoid, target)
                    if metric_count > 0:
                        batch_avg = metric_sum / metric_count
                        # MetricTracker send to (key, batch_avg, batch_count)
                        self.valid_metrics.update(met.__name__, batch_avg, n=metric_count)

        return self.valid_metrics.result()

    def _save_checkpoint(self):
        """Saves model checkpoint locally and logs it as a wandb artifact."""
        state = {
            'arch': type(self.model).__name__,
            'state_dict': self.model.state_dict(),
            'best_epoch_info': self.best_epoch_info,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / self.checkpoint_name)
        torch.save(state, filename)
        
        if wandb.run:
            artifact = wandb.Artifact(name=f"{wandb.run.name}-best-model", type="model")
            artifact.add_file(filename)

    def _log_validation_images(self, epoch):
        """Logs a fixed batch of validation images, masks, and predictions to wandb."""
        if not wandb.run: return
        self.model.eval()
        images = self.fixed_val_batch['image'].to(self.device)
        gt_masks = self.fixed_val_batch['mask'].to(self.device)
        
        with torch.no_grad():
            pred_logits = self.model(images)
            pred_masks = torch.sigmoid(pred_logits)

        images_grid = make_grid(images, normalize=True)
        gt_masks_grid = make_grid(gt_masks, normalize=True)
        pred_masks_grid = make_grid(pred_masks, normalize=True)

        wandb.log({
            "validation_samples": [
                wandb.Image(images_grid, caption="Input Images"),
                wandb.Image(gt_masks_grid, caption="Ground Truth Masks"),
                wandb.Image(pred_masks_grid, caption=f"Predicted Masks (Epoch {epoch})")
            ]
        }, step=epoch)
