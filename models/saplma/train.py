#!/usr/bin/env python3
"""
Script to train SAPLMA (Self-Assessed Post-LLM Answer) models.
Implements a simple MLP architecture to classify correctness based on
response representations extracted from the response's final token.
Uses static configuration for direct training.
Supports training on final, middle, or upper-middle layer representations.
"""

import argparse
import json
import os
import sys
import numpy as np
import random
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
from tqdm import tqdm

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Static training configurations
TRAINING_CONFIGS = {
    'default': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'hidden_layers': (512, 128),
        'dropout': 0.2,
        'batch_size': 32,
        'max_epochs': 200,
        'patience': 20,
        'grad_clip': 1.0
    },
    'small': {
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'hidden_layers': (256, 64),
        'dropout': 0.1,
        'batch_size': 32,
        'max_epochs': 200,
        'patience': 20,
        'grad_clip': 1.0
    },
    'large': {
        'learning_rate': 2e-3,
        'weight_decay': 1e-3,
        'hidden_layers': (1024, 256, 64),
        'dropout': 0.3,
        'batch_size': 32,
        'max_epochs': 200,
        'patience': 20,
        'grad_clip': 1.0
    },
    'linear': {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'hidden_layers': (),
        'dropout': 0.0,
        'batch_size': 32,
        'max_epochs': 200,
        'patience': 20,
        'grad_clip': 1.0
    }
}

# Parse arguments early to set CUDA_VISIBLE_DEVICES before importing torch
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SAPLMA models with static configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--representations-dir", type=str, required=True,
                       help="Directory containing hidden state representations")
    parser.add_argument("--output-dir", type=str, default="../trained_models/SAPLMA",
                       help="Output directory for trained models")
    
    # Layer selection
    parser.add_argument("--layer-type", type=str, required=True,
                       choices=['final', 'middle', 'upper_middle'],
                       help="Which layer to train on: final, middle, or upper_middle")
    
    # Training configuration
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda/cpu)")
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                       help="CUDA_VISIBLE_DEVICES setting (e.g., '0,1,2,3')")
    
    # Dataset selection
    parser.add_argument("--model-id", type=str, default=None,
                       help="Specific model to train on (if not specified, trains on all found models)")
    
    # Configuration selection
    parser.add_argument("--config", type=str, default="default",
                       choices=list(TRAINING_CONFIGS.keys()),
                       help=f"Training configuration to use: {list(TRAINING_CONFIGS.keys())}")
    
    return parser.parse_args()

# Parse arguments immediately
args = parse_arguments()

# Set CUDA_VISIBLE_DEVICES early if specified
if args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

# Now import torch and other CUDA-dependent libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

# Add utils directory to path
sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything
from eval import calculate_all_metrics

RANDOM_SEED = 23
seed_everything(RANDOM_SEED)

# Get selected configuration
TRAIN_CONFIG = TRAINING_CONFIGS[args.config]
logger.info(f"Using training configuration: {args.config}")
logger.info(f"Configuration: {json.dumps(TRAIN_CONFIG, indent=2)}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_hidden_layers(layer_str: str) -> Tuple[int, ...]:
    """Parse hidden layer string into tuple of integers."""
    return tuple(int(x) for x in layer_str.split(","))

def calculate_composite_score(auroc: float, ece: float, auroc_weight: float = 0.5) -> float:
    """Calculate composite validation score balancing AUROC and ECE."""
    ece_weight = 1.0 - auroc_weight
    ece_score = 1.0 - ece
    composite_score = auroc_weight * auroc + ece_weight * ece_score
    return composite_score

def calculate_auroc_with_ece_constraint(auroc: float, ece: float) -> float:
    """Calculate AUROC with ECE penalty."""
    penalty_factor = 1.0 - ece
    return auroc * penalty_factor

def pick_tau_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick optimal threshold by maximizing Youden's J statistic."""
    taus = np.linspace(0.0, 1.0, 1001)
    best_j, best_tau = -1.0, 0.5
    
    for t in taus:
        yhat = (y_prob >= t).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        tn = np.sum((yhat == 0) & (y_true == 0))
        fp = np.sum((yhat == 1) & (y_true == 0))
        fn = np.sum((yhat == 0) & (y_true == 1))
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_tau = j, t
    
    return float(best_tau)

def sens_spec_at_tau(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Tuple[float, float]:
    """Calculate sensitivity and specificity at a given threshold."""
    yhat = (y_prob >= tau).astype(int)
    tp = np.sum((yhat == 1) & (y_true == 1))
    tn = np.sum((yhat == 0) & (y_true == 0))
    fp = np.sum((yhat == 1) & (y_true == 0))
    fn = np.sum((yhat == 0) & (y_true == 1))
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return float(sens), float(spec)

def calculate_composite_with_constraints(metrics: Dict[str, float], y_true: np.ndarray, y_prob: np.ndarray,
                                      alpha: float = 0.6, min_sens: float = 0.60, min_spec: float = 0.60) -> Tuple[float, bool, Dict[str, float]]:
    """Calculate composite score with sensitivity/specificity constraints."""
    auroc = metrics["auroc"]
    ece = metrics["ece"]
    score = alpha * auroc + (1 - alpha) * (1.0 - ece)
    
    tau = pick_tau_by_youden(y_true, y_prob)
    sens, spec = sens_spec_at_tau(y_true, y_prob, tau)
    feasible = (sens >= min_sens) and (spec >= min_spec)
    
    return score, feasible, {"tau": tau, "sens": sens, "spec": spec}

class SAPLMADataset(Dataset):
    """Dataset for SAPLMA training."""
    
    def __init__(self, hidden_states: np.ndarray, labels: np.ndarray):
        """Initialize dataset."""
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.labels = torch.FloatTensor(labels.astype(float))
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]


class SAPLMAModel(nn.Module):
    """SAPLMA MLP model for classifying response representations."""
    
    def __init__(self, input_dim: int, hidden_layers: Union[str, Tuple[int, ...], List[int]], dropout: float = 0.0):
        """Initialize SAPLMA model.
        
        Args:
            input_dim: Input dimension
            hidden_layers: Layer configuration as string ("1024,128") or sequence of dimensions
            dropout: Dropout probability
        """
        super(SAPLMAModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Parse hidden layers if string
        if isinstance(hidden_layers, str):
            self.hidden_layers = parse_hidden_layers(hidden_layers)
        else:
            self.hidden_layers = tuple(hidden_layers) if isinstance(hidden_layers, (list, tuple)) else hidden_layers
        
        if not self.hidden_layers:
            self.classifier = nn.Linear(input_dim, 1)
        else:
            # Build dynamic sequential model
            layers = []
            current_dim = input_dim
            
            for hidden_dim in self.hidden_layers:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
            
            # Add final output layer
            layers.append(nn.Linear(current_dim, 1))
            
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        logits = self.classifier(x)
        return logits.squeeze(-1)


class SAPLMATrainer:
    """SAPLMA model trainer."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize trainer."""
        if torch.cuda.is_available() and device.startswith('cuda'):
            if device == 'cuda':
                self.device = 'cuda:0'
            else:
                self.device = device
        else:
            self.device = 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if self.device.startswith('cuda'):
            gpu_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3:.2f}GB")
        
    
    def load_hidden_states_from_directory(self, data_dir: Path, layer_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all hidden states and labels from a directory of .npz files for a specific layer."""
        hidden_states_list = []
        labels_list = []
        
        # Load from the layer-specific subdirectory
        layer_dir = data_dir / layer_type
        if not layer_dir.exists():
            raise ValueError(f"Layer directory not found: {layer_dir}")
        
        npz_files = list(layer_dir.glob("*.npz"))
        logger.info(f"Loading data from {len(npz_files)} files in {layer_dir}")
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                hidden_state = data['hidden_state']
                
                # Load correctness (binary 0/1)
                if 'correctness' in data:
                    correctness = data['correctness'].item()
                elif 'overall_grade' in data:
                    overall_grade = data['overall_grade'].item()
                    correctness = int(bool(overall_grade))
                    logger.warning(f"{npz_file}: Using deprecated 'overall_grade', converting to correctness")
                else:
                    logger.warning(f"Skipping {npz_file}: no correctness or overall_grade found")
                    continue
                
                # Convert correctness to binary label (True/False)
                label = bool(correctness)
                
                hidden_states_list.append(hidden_state)
                labels_list.append(label)
                
            except Exception as e:
                logger.warning(f"Error loading {npz_file}: {e}")
                continue
        
        if not hidden_states_list:
            raise ValueError(f"No valid data found in {layer_dir}")
        
        # Stack all data
        all_hidden_states = np.stack(hidden_states_list, axis=0)
        all_labels = np.array(labels_list, dtype=bool)
        
        logger.info(f"Loaded {len(all_hidden_states)} samples from {layer_type} layer")
        logger.info(f"Hidden states shape: {all_hidden_states.shape}")
        logger.info(f"Labels shape: {all_labels.shape}")
        logger.info(f"Positive label ratio: {np.mean(all_labels):.3f}")
        
        return all_hidden_states, all_labels
    
    def calculate_pos_weight_ratio(self, labels: np.ndarray) -> float:
        """Calculate the ratio of negative to positive samples."""
        n_positive = np.sum(labels)
        n_negative = len(labels) - n_positive
        ratio = n_negative / n_positive if n_positive > 0 else 1.0
        logger.info(f"Positive samples: {n_positive}, Negative samples: {n_negative}")
        logger.info(f"Pos weight ratio: {ratio:.3f}")
        return ratio
    
    def split_train_val(self, hidden_states: np.ndarray, labels: np.ndarray, 
                       val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets with stratified sampling."""
        train_hidden_states, val_hidden_states, train_labels, val_labels = train_test_split(
            hidden_states, labels, 
            test_size=val_ratio, 
            random_state=RANDOM_SEED, 
            stratify=labels
        )
        
        logger.info(f"Train set: {len(train_hidden_states)} samples")
        logger.info(f"Validation set: {len(val_hidden_states)} samples")
        logger.info(f"Train positive ratio: {np.mean(train_labels):.3f}")
        logger.info(f"Val positive ratio: {np.mean(val_labels):.3f}")
        
        return train_hidden_states, train_labels, val_hidden_states, val_labels
    
    def train_model(self, model: SAPLMAModel, train_loader: DataLoader, val_loader: DataLoader,
                    pos_weight_ratio: float, learning_rate: float, weight_decay: float,
                    max_epochs: int = 200, patience: int = 20, grad_clip: float = 1.0) -> Tuple[SAPLMAModel, Dict[str, Any]]:
        """Train a SAPLMA model with early stopping."""
        logger.info(f"Moving model to device {self.device}...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"CUDA cache cleared")
        
        model = model.to(self.device)
        logger.info(f"Model moved to device successfully")
        
        # Initialize loss function and optimizer
        pos_weight = torch.tensor(pos_weight_ratio, device=self.device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Initialize AMP scaler
        device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'
        scaler = GradScaler(enabled=device_type == 'cuda')
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'best_epoch': None,
            'best_operating_point': None
        }
        
        best_val_score = float('-inf')
        best_model_state = None
        best_operating_point = None
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_hidden_states, batch_labels in train_loader:
                batch_hidden_states = batch_hidden_states.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast(device_type=device_type, enabled=scaler.is_enabled()):
                    predictions = model(batch_hidden_states)
                    loss = criterion(predictions, batch_labels)
                
                scaler.scale(loss).backward()
                
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * len(batch_hidden_states)
                train_samples += len(batch_hidden_states)
            
            avg_train_loss = train_loss / train_samples
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_samples = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch_hidden_states, batch_labels in val_loader:
                    batch_hidden_states = batch_hidden_states.to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)
                    
                    with autocast(device_type=device_type, enabled=scaler.is_enabled()):
                        logits = model(batch_hidden_states)
                        loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item() * len(batch_hidden_states)
                    val_samples += len(batch_hidden_states)
                    
                    probabilities = torch.sigmoid(logits)
                    
                    all_val_preds.extend(probabilities.cpu().numpy())
                    all_val_labels.extend(batch_labels.cpu().numpy())
            
            avg_val_loss = val_loss / val_samples
            all_val_preds = np.array(all_val_preds)
            all_val_labels = np.array(all_val_labels)
            
            val_metrics = calculate_all_metrics(all_val_labels, all_val_preds)
            val_metrics['loss'] = avg_val_loss
            
            # Always compute operating point metrics for monitoring
            tau = pick_tau_by_youden(all_val_labels, all_val_preds)
            sens, spec = sens_spec_at_tau(all_val_labels, all_val_preds, tau)
            operating_point = {"tau": tau, "sens": sens, "spec": spec}
            
            # Calculate validation score (using AUROC as primary metric)
            val_score = val_metrics['auroc']
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Early stopping based on validation AUROC
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                history['best_epoch'] = epoch
                best_operating_point = operating_point
                history['best_operating_point'] = operating_point
                
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val AUROC: {val_metrics['auroc']:.4f}, "
                           f"ECE: {val_metrics['ece']:.4f}, Brier: {val_metrics['brier']:.4f}")
                logger.info(f"Operating point (Youden's J) - threshold: {operating_point['tau']:.3f}, "
                          f"sensitivity: {operating_point['sens']:.3f}, specificity: {operating_point['spec']:.3f} "
                          f"(balanced accuracy: {(operating_point['sens'] + operating_point['spec'])/2:.3f})")
            else:
                patience_counter += 1
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                               f"Val AUROC: {val_metrics['auroc']:.4f}, "
                               f"Operating point - sens: {operating_point['sens']:.3f}, spec: {operating_point['spec']:.3f} "
                               f"(J: {operating_point['sens'] + operating_point['spec'] - 1:.3f})")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Training completed successfully")
        return model, history
    
    def evaluate_model(self, model: SAPLMAModel, data_loader: DataLoader, return_arrays: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Evaluate model on a dataset."""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_hidden_states, batch_labels in data_loader:
                batch_hidden_states = batch_hidden_states.to(self.device)
                logits = model(batch_hidden_states)
                
                probabilities = torch.sigmoid(logits)
                
                all_preds.extend(probabilities.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics = calculate_all_metrics(all_labels, all_preds)
        
        if return_arrays:
            return metrics, all_labels, all_preds
        return metrics
    
    def train_with_config(self, train_hidden_states: np.ndarray, train_labels: np.ndarray,
                         val_hidden_states: np.ndarray, val_labels: np.ndarray,
                         output_dir: Path, config: Dict[str, Any], layer_type: str) -> Tuple[Dict[str, Any], SAPLMAModel, Dict[str, Any]]:
        """Train a SAPLMA model with given configuration."""
        
        pos_weight_ratio = self.calculate_pos_weight_ratio(train_labels)
        
        train_dataset = SAPLMADataset(train_hidden_states, train_labels)
        val_dataset = SAPLMADataset(val_hidden_states, val_labels)
        
        self.input_dim = train_hidden_states.shape[1]
        
        # Create model with config
        model = SAPLMAModel(self.input_dim, config['hidden_layers'], config['dropout'])
        n_params = count_parameters(model)
        logger.info(f"Model has {n_params:,} parameters")
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                generator=torch.Generator().manual_seed(RANDOM_SEED),
                                pin_memory=self.device.startswith('cuda'),
                                num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                              pin_memory=self.device.startswith('cuda'),
                              num_workers=0)
        
        # Train model
        trained_model, history = self.train_model(
            model, train_loader, val_loader,
            pos_weight_ratio,
            config['learning_rate'], config['weight_decay'],
            max_epochs=config['max_epochs'],
            patience=config['patience'],
            grad_clip=config['grad_clip']
        )
        
        # Evaluate model
        val_metrics, val_labels, val_preds = self.evaluate_model(trained_model, val_loader, return_arrays=True)
        
        logger.info(f"\nFinal validation metrics for {layer_type} layer:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save model
        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(trained_model.state_dict(), best_dir / "model.pth")
        
        model_info = {
            'input_dim': trained_model.input_dim,
            'hidden_layers': ','.join(str(x) for x in trained_model.hidden_layers),
            'layer_type': layer_type,
            'config': config,
            'metrics': val_metrics,
            'best_epoch': history.get('best_epoch'),
            'best_operating_point': history.get('best_operating_point'),
            'parameter_info': {
                'n_params': n_params
            }
        }
        
        with open(best_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model saved to: {best_dir}")
        
        return config, trained_model, val_metrics
    
    def train_dataset(self, model_dir: Path, output_dir: Path, model_name: str, 
                     config: Dict[str, Any], layer_type: str):
        """Train SAPLMA model for a specific model and layer type."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training SAPLMA model for model: {model_name}")
        logger.info(f"Layer type: {layer_type}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Configuration: {config}")
        
        model_output_dir = output_dir / model_name / layer_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_dir = model_dir / "train"
        if not train_dir.exists():
            logger.error(f"Training directory not found: {train_dir}")
            return
        
        hidden_states, labels = self.load_hidden_states_from_directory(train_dir, layer_type)
        
        # Shuffle the data before splitting
        indices = np.arange(len(hidden_states))
        np.random.shuffle(indices)
        hidden_states = hidden_states[indices]
        labels = labels[indices]
        
        train_hidden_states, train_labels, val_hidden_states, val_labels = \
            self.split_train_val(hidden_states, labels)
        
        # Train with config
        best_config, best_model, best_metrics = self.train_with_config(
            train_hidden_states, train_labels, val_hidden_states, val_labels,
            model_output_dir, config, layer_type
        )
        
        logger.info(f"Training completed for model: {model_name}, layer: {layer_type}")


def main():
    
    # Log CUDA setup
    logger.info(f"SAPLMA Training Script with Static Configuration")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info(f"CUDA available: False")
    
    # Initialize trainer
    trainer = SAPLMATrainer(device=args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Representations directory: {args.representations_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Layer type: {args.layer_type}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Configuration: {args.config}")
    
    # Find all model directories in representations dir
    representations_base = Path(args.representations_dir)
    
    if not representations_base.exists():
        logger.error(f"Representations directory not found: {representations_base}")
        return
    
    # Find all model directories (subdirectories with train/test folders)
    model_dirs = []
    for potential_model_dir in representations_base.iterdir():
        if potential_model_dir.is_dir():
            train_dir = potential_model_dir / "train" / args.layer_type
            if train_dir.exists() and any(train_dir.glob("*.npz")):
                model_dirs.append(potential_model_dir)
    
    if not model_dirs:
        logger.error(f"No valid model directories found in {representations_base} for layer type {args.layer_type}")
        return
    
    logger.info(f"Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        logger.info(f"  - {model_dir.name}")
    
    # Train models for each model
    for model_dir in model_dirs:
        model_name = model_dir.name
        
        # Skip if specific model requested and this isn't it
        if args.model_id is not None:
            requested_model_dir_name = args.model_id.replace('/', '-')
            if model_name != requested_model_dir_name:
                logger.info(f"Skipping {model_name} (not requested model)")
                continue
        
        try:
            trainer.train_dataset(model_dir, output_dir, model_name, TRAIN_CONFIG, args.layer_type)
        except Exception as e:
            logger.error(f"Error training model for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("Training completed for all models!")


if __name__ == "__main__":
   main()

# Example Usage:
# python SAPLMA_train.py \
#   --representations-dir ../representations/SAPLMA \
#   --output-dir        ../trained_models/SAPLMA \
#   --model-id          "Qwen/Qwen2.5-0.5B-Instruct" \
#   --layer-type        final \
#   --device            cuda \
#   --cuda-visible-devices "0" \
#   --config            default
