import torch
import shutil
from pathlib import Path

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, run_name: str):
    """Save model checkpoint, keeping track of the best one."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    last_ckpt_path = Path(checkpoint_dir) / f"{run_name}_last.pt"
    best_ckpt_path = Path(checkpoint_dir) / f"{run_name}_best.pt"
    
    torch.save(state, last_ckpt_path)
    
    if is_best:
        shutil.copyfile(last_ckpt_path, best_ckpt_path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None) -> dict:
    """Load model weights and optional optimizer/scheduler states."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    return checkpoint
