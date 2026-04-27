import os
import json
import torch


class TrainingLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }

    def log_epoch(self, train_loss, val_loss, lr):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rates'].append(lr)

    def save_metrics(self):
        path = os.path.join(self.log_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, model, optimizer, epoch, best_val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics': self.metrics,
        }
        path = os.path.join(self.log_dir, 'checkpoint.pth')
        torch.save(checkpoint, path)
        return path

    def load_model(self, path, model, optimizer=None):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))
