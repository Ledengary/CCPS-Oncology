#!/usr/bin/env python3
"""
Comprehensive PIK evaluation script.
Loads trained PIK models and evaluates them on test data for question understanding classification.
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm

# Add utils directory to path for eval functions
sys.path.append(str(Path(__file__).parent / "../utils"))
from eval import evaluate_by_groups, save_evaluation_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PIKModel(nn.Module):
    """PIK MLP model for classifying question understanding representations."""
    
    def __init__(self, input_dim: int, hidden_layers: Union[str, Tuple[int, ...], List[int]], dropout: float = 0.0):
        """Initialize PIK model.
        
        Args:
            input_dim: Input dimension
            hidden_layers: Layer configuration as string ("1024,128") or sequence of dimensions
            dropout: Dropout probability
        """
        super(PIKModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Parse hidden layers if string
        if isinstance(hidden_layers, str):
            self.hidden_layers = tuple(int(x) for x in hidden_layers.split(","))
        else:
            self.hidden_layers = tuple(hidden_layers) if isinstance(hidden_layers, (list, tuple)) else hidden_layers
        
        if not self.hidden_layers:  # Linear probe
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
        return logits.squeeze()


class PIKEvaluator:
    def __init__(self, model_id: str, trained_models_dir: str, target_model_version: str = "best"):
        """Initialize the PIK evaluator."""
        self.model_id = model_id
        self.trained_models_dir = trained_models_dir
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_trained_model(target_model_version)
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"PIK model loaded and moved to device: {self.device}")
    
    def _load_trained_model(self, target_model_version) -> Optional[PIKModel]:
        """Load the best trained PIK model."""
        model_dir_name = self.model_id.replace('/', '-')
        if target_model_version != "best":
            target_model_version = Path("models") / target_model_version
        model_path = Path(self.trained_models_dir) / model_dir_name / target_model_version
        
        if not model_path.exists():
            logger.error(f"Trained model directory not found: {model_path}")
            return None
        
        # Load model info
        model_info_path = model_path / "model_info.json"
        if not model_info_path.exists():
            logger.error(f"Model info file not found: {model_info_path}")
            return None
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create model with correct architecture
        input_dim = model_info['input_dim']
        hidden_layers = model_info.get('hidden_layers', None)
        
        # Handle different formats: string ("512,128"), tuple/list, or legacy hidden_dim
        if hidden_layers is None and 'hidden_dim' in model_info:
            # Handle legacy format
            hidden_layers = [model_info['hidden_dim']]
        elif isinstance(hidden_layers, str):
            # Already in string format (comma-separated), PIKModel will parse it
            pass
        elif isinstance(hidden_layers, (list, tuple)):
            # Convert to string format for consistency
            hidden_layers = ','.join(str(x) for x in hidden_layers)
        
        dropout = model_info.get('config', {}).get('dropout', 0.0)
        
        logger.info(f"Loading PIK model with input_dim={input_dim}, hidden_layers={hidden_layers}, dropout={dropout}")
        model = PIKModel(input_dim, hidden_layers, dropout)
        
        # Load state dict
        state_dict_path = model_path / "model.pth"
        if not state_dict_path.exists():
            logger.error(f"Model state dict not found: {state_dict_path}")
            return None
        
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        
        logger.info(f"Model config: {model_info.get('config', {})}")
        logger.info(f"Model metrics: {model_info.get('metrics', {})}")
        
        # Store model_info for later use
        self.model_info = model_info
        
        return model
    
    
    def load_test_representations(self, representations_dir: Path, split_name: str = "test") -> List[Dict[str, Any]]:
        """Load test representations from directory."""
        model_dir_name = self.model_id.replace('/', '-')
        split_dir = representations_dir / model_dir_name / split_name
        
        if not split_dir.exists():
            logger.error(f"{split_name} representations directory not found: {split_dir}")
            return []
        
        records = []
        npz_files = list(split_dir.glob("*.npz"))
        logger.info(f"Loading {split_name} data from {len(npz_files)} files in {split_dir}")
        
        hidden_dims: List[int] = []
        
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                hidden_state = data['hidden_state']  # Shape: (hidden_dim,) - single representation
                
                # Extract record_id (row number from CSV, not sidx to avoid duplicates)
                record_id = data['record_id'].item() if hasattr(data['record_id'], 'item') else str(data['record_id'])
                # Ensure record_id is string (row number)
                record_id = str(record_id)
                
                # Load correctness (binary 0/1)
                if 'correctness' in data:
                    correctness = data['correctness'].item() if hasattr(data['correctness'], 'item') else int(data['correctness'])
                elif 'overall_grade' in data:
                    # Backward compatibility: convert overall_grade to correctness
                    overall_grade = data['overall_grade'].item() if hasattr(data['overall_grade'], 'item') else data['overall_grade']
                    correctness = int(bool(overall_grade))
                    logger.warning(f"{npz_file}: Using deprecated 'overall_grade', converting to correctness")
                else:
                    logger.warning(f"Skipping {npz_file}: no correctness or overall_grade found")
                    continue
                
                hidden_dim = hidden_state.shape[0]
                hidden_dims.append(hidden_dim)
                
                records.append({
                    'record_id': record_id,  # Row number from CSV (not sidx to avoid duplicates)
                    'hidden_state': hidden_state,
                    'correctness': correctness
                })
                
            except Exception as e:
                logger.warning(f"Error loading {npz_file}: {e}")
                continue
        
        num_records = len(records)
        if num_records == 0:
            logger.warning("No valid test records loaded")
            return records
        
        # Check hidden-dim consistency
        unique_hidden_dims = set(hidden_dims)
        if len(unique_hidden_dims) != 1:
            logger.error(f"Inconsistent hidden_dim sizes found across records: {unique_hidden_dims}")
        hidden_dim_value = unique_hidden_dims.pop() if unique_hidden_dims else None
        
        # Report summary
        logger.info(f"Loaded {num_records} records with hidden_dim: {hidden_dim_value}")
        
        return records
    
    def apply_pik_classification(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply PIK classification to a single record."""
        if not self.model:
            logger.error("PIK model not loaded")
            return None

        hidden_state = record['hidden_state']  # Shape: (hidden_dim,)
        
        # Convert to tensor and add batch dimension
        hidden_state_tensor = torch.FloatTensor(hidden_state).unsqueeze(0).to(self.device)  # Shape: (1, hidden_dim)

        with torch.no_grad():
            logits = self.model(hidden_state_tensor)  # Output logits
            
            # Apply sigmoid to get probabilities
            confidence = torch.sigmoid(logits).item()

        return {
            'record_id': record['record_id'],  # Row number from CSV
            'confidence_score': confidence,
            'correctness': record['correctness']
        }
    
    def extract_ground_truth_label(self, processed_result: Dict[str, Any]) -> Optional[int]:
        """Extract ground truth label for the record."""
        correctness = processed_result.get('correctness')
        if correctness is None:
            return None
        return int(correctness)
    
    def extract_dataset_info(self, processed_result: Dict[str, Any]) -> Tuple[str, str]:
        """Extract dataset information."""
        dataset = processed_result.get('dataset', 'unknown')
        # Use dummy category for MCQA (matching PTRUE.py pattern)
        category = 'all'
        return dataset, category
    
    def process_test_data(self, representations_dir: Path, split_name: str = "test") -> Tuple[List[Dict[str, Any]], int]:
        """Process all test records and return classification results."""
        logger.info(f"Processing {split_name} data with PIK classification...")
        
        # Determine dataset name based on split_name (matching PTRUE.py pattern)
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        
        # Load test representations
        test_records = self.load_test_representations(representations_dir, split_name)
        if not test_records:
            return [], 0
        
        total_records = len(test_records)
        processed_results = []
        
        # Process each record
        for record in tqdm(test_records, desc="Applying PIK classification"):
            try:
                result = self.apply_pik_classification(record)
                if result:
                    # Ensure dataset name is correct based on split_name
                    result['dataset'] = dataset_name
                    processed_results.append(result)
            except Exception as e:
                logger.error(f"Error processing record {record['record_id']}: {e}")
                continue
        
        logger.info(f"\nPIK Classification Summary:")
        logger.info(f"  Total records processed: {len(processed_results)}")
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Average confidence: {np.mean([r['confidence_score'] for r in processed_results]):.3f}")
        
        return processed_results, total_records
    
    def prepare_evaluation_records(self, processed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare records for evaluation in the standard format."""
        evaluation_records = []
        
        for result in processed_results:
            # Extract ground truth
            ground_truth_label = self.extract_ground_truth_label(result)
            if ground_truth_label is None:
                continue
            
            evaluation_record = {
                'record_id': result['record_id'],
                'dataset': result.get('dataset', 'unknown'),
                'category': 'all',  # Dummy category for MCQA (matching PTRUE.py pattern)
                'ground_truth_correctness': ground_truth_label,
                'confidence_score': result['confidence_score'],
                'original_result': result
            }
            
            evaluation_records.append(evaluation_record)
        
        return evaluation_records
    
    def save_processed_data(self, processed_records: List[Dict[str, Any]], output_path: Path) -> None:
        """Save processed records to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_records, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {output_path}")
    
    def calculate_and_save_results(self, evaluation_records: List[Dict[str, Any]], 
                                 processed_results: List[Dict[str, Any]],
                                 output_path: Path, split_name: str, model_id: str,
                                 n_total_samples: int) -> None:
        """Calculate comprehensive evaluation results and save them."""
        if not evaluation_records:
            logger.warning("No evaluation records to process")
            return
        
        # Extract arrays for evaluation
        ground_truth_labels = np.array([r['ground_truth_correctness'] for r in evaluation_records])
        confidence_scores = np.array([r['confidence_score'] for r in evaluation_records])
        datasets = np.array([r['dataset'] for r in evaluation_records])
        # Use dummy categories for MCQA (matching PTRUE.py pattern)
        categories = np.array(['all'] * len(evaluation_records))
        
        # Calculate comprehensive evaluation results
        evaluation_results = evaluate_by_groups(ground_truth_labels, confidence_scores, datasets, categories)    
        
        # Inject total count
        evaluation_results['overall']['n_total_samples'] = n_total_samples
        
        # Calculate PIK-specific statistics
        avg_confidence = np.mean(confidence_scores)
        
        # Determine dataset name based on split_name (matching PTRUE.py pattern)
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        
        # Add PIK-specific metadata
        evaluation_results['metadata'] = {
            'model_id': model_id,
            'split_name': split_name,
            'dataset': dataset_name,
            'total_records': len(evaluation_records),
            'unique_datasets': list(set(r['dataset'] for r in evaluation_records)),
            'evaluation_timestamp': str(np.datetime64('now')),
            'pik_statistics': {
                'avg_confidence': float(avg_confidence),
                'confidence_std': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores))
            },
            'pik_model_info': {
                'trained_models_dir': self.trained_models_dir,
                'input_dim': self.model.input_dim if self.model else None,
                'hidden_layers': self.model.hidden_layers if self.model else None,
                'dropout': getattr(self, 'model_info', {}).get('config', {}).get('dropout', None) if self.model else None
            }
        }
        
        # Save evaluation results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_results(evaluation_results, output_path)
        
        # Print summary
        self.print_results_summary(evaluation_results, split_name)
    
    def print_results_summary(self, results: Dict[str, Any], split_name: str) -> None:
        """Print a summary of evaluation results."""
        print(f"\n{split_name.upper()} PIK EVALUATION SUMMARY:")
        print("=" * 60)
        
        # Overall results
        if 'overall' in results:
            overall = results['overall']
            print(f"Overall Performance:")
            print(f"  Samples: {overall['n_samples']} / {overall['n_total_samples']}")
            print(f"  Accuracy: {overall['accuracy']:.4f}")
            print(f"  Precision: {overall['precision']:.4f}")
            print(f"  Recall: {overall['recall']:.4f}")
            print(f"  F1-Score: {overall['f1']:.4f}")
            print(f"  Sensitivity: {overall['sensitivity']:.4f}")
            print(f"  Specificity: {overall['specificity']:.4f}")
            print(f"  ECE: {overall['ece']:.4f}")
            print(f"  Brier Score: {overall['brier']:.4f}")
            print(f"  AUROC: {overall['auroc']:.4f}")
            print(f"  AUCPR: {overall['aucpr']:.4f}")
        
        # PIK statistics
        if 'metadata' in results and 'pik_statistics' in results['metadata']:
            stats = results['metadata']['pik_statistics']
            print(f"\nPIK Classification Statistics:")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Confidence std: {stats['confidence_std']:.3f}")
            print(f"  Confidence range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
        
        # Dataset breakdown
        if 'by_dataset' in results:
            print(f"\nDataset Breakdown:")
            for dataset, metrics in results['by_dataset'].items():
                print(f"  {dataset}: {metrics['n_samples']} samples, "
                      f"Acc: {metrics['accuracy']:.3f}, ECE: {metrics['ece']:.3f}")
        
        print("=" * 60)
    
    def process_split(self, representations_dir: Path, output_dir: Path, split_name: str, model_id: str) -> None:
        """Process a single split (test only for evaluation)."""
        logger.info(f"Processing {split_name} data for model: {model_id}")
        
        # Determine dataset name based on split_name (matching PTRUE.py pattern)
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        logger.info(f"Dataset: {dataset_name}")
        
        # Process the test data
        processed_results, n_total_samples = self.process_test_data(representations_dir, split_name)
        if not processed_results:
            logger.error(f"No valid records processed from representations")
            return
        
        # Prepare evaluation records
        evaluation_records = self.prepare_evaluation_records(processed_results)
        
        # Save processed data
        labels_path = output_dir / f"{split_name}_labels.json"
        self.save_processed_data(evaluation_records, labels_path)
        
        # Calculate and save results
        results_path = output_dir / f"{split_name}_results.json"
        self.calculate_and_save_results(
            evaluation_records, processed_results, results_path, 
            split_name, model_id, n_total_samples
        )


def create_output_path(output_dir: str, model_id: str) -> Path:
    """Create output directory path."""
    model_dir_name = model_id.replace('/', '-')
    return Path(output_dir) / model_dir_name


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PIK models for question understanding classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input configuration
    parser.add_argument("--representations-dir", type=str, required=True,
                       help="Directory containing test representations (e.g., ../representations/PIK)")
    parser.add_argument("--trained-models-dir", type=str, required=True,
                       help="Directory containing trained PIK models (e.g., ../trained_models/PIK)")
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model ID to evaluate (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')")
    parser.add_argument("--target-model-version", type=str, default="best",
                       help="Target model version (best, or a config dir)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save evaluation results")
    
    # Processing options
    parser.add_argument("--test-only", action="store_true",
                       help="Process only test data (default behavior)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Initialize evaluator
    evaluator = PIKEvaluator(
        model_id=args.model_id,
        trained_models_dir=args.trained_models_dir,
        target_model_version=args.target_model_version
    )
    
    # Create output path
    output_path = create_output_path(args.output_dir, args.model_id)
    representations_dir = Path(args.representations_dir)
    
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Representations directory: {representations_dir}")
    logger.info(f"Trained models directory: {args.trained_models_dir}")
    logger.info(f"Output directory: {output_path}")
    
    # Process test data
    evaluator.process_split(representations_dir, output_path, 'test', args.model_id)
    
    print(f"\nEvaluation complete! Results saved to: {output_path}")
    print("\nGenerated files:")
    print("- test_labels.json: Structured records with ground truth and confidence scores")
    print("- test_results.json: Comprehensive evaluation metrics")
    print("  - Overall classification metrics")
    print("  - PIK classification statistics")
    print("  - Metrics by dataset and category combinations")


if __name__ == "__main__":
    main()

# microsoft/Phi-4-mini-flash-reasoning - 6038
# Qwen/Qwen3-8B - 6081
# Qwen/Qwen3-14B - 6148
# mistralai/Magistral-Small-2506 - 6236
# Qwen/QwQ-32B - 6328
# LGAI-EXAONE/EXAONE-Deep-32B - 6320

# Example usage:
# python PIK_evaluation.py \
#     --representations-dir ../representations/PIK \
#     --trained-models-dir ../trained_models/PIK \
#     --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
#     --output-dir ../results/PIK \
#     --target-model-version best