import sys
import argparse
import os
import json
import copy
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Perform uncertainty estimation using pretrained CCPS model on test data")

# Add arguments with both hyphen and underscore versions for backward compatibility
parser.add_argument('--cuda-devices', '--visible_cudas', dest='cuda_devices', type=str, required=True, help='CUDA devices to use (e.g., "0,1,2,3")')
parser.add_argument('--model-id', '--llm_id', dest='model_id', type=str, required=True, help='Model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct)')
parser.add_argument('--test-dataset-name', '--test_dataset_name', dest='test_dataset_name', type=str, required=True, help='Test dataset name (e.g., test)')
parser.add_argument('--train-dataset-name', '--pretrained_dataset_name', dest='train_dataset_name', type=str, required=True, help='Train dataset name used for training (e.g., train)')
parser.add_argument('--feature-dir', '--feature_dir', dest='feature_dir', type=str, required=True, help='Directory containing feature files (e.g., ../features/OrigPert)')
parser.add_argument('--contrastive-model-path', '--contrastive_model_path', dest='contrastive_model_path', type=str, required=True, help='Path to the trained contrastive model directory')
parser.add_argument('--classifier-model-path', '--classifier_model_path', dest='classifier_model_path', type=str, required=True, help='Path to the trained classifier model directory')
parser.add_argument('--output-dir', '--results_dir', dest='output_dir', type=str, required=True, help='Directory to save uncertainty estimation results')

# Optional arguments
parser.add_argument('--test-data-dir', '--test_data_dir', dest='test_data_dir', type=str, default=None, help='Test data directory (root of tasks) - for old format compatibility')
parser.add_argument('--data-subdir', '--data_subdir', dest='data_subdir', type=str, default=None, help='Subdirectory under each task where assessed outputs are saved - for old format compatibility')
parser.add_argument('--tasks', type=str, required=False, nargs='*', help='List of Tasks to process', default=None)
parser.add_argument('--seed', type=int, required=False, help="Seed for reproducibility", default=23)
parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, required=False, help="Batch size for evaluation", default=64)
parser.add_argument('--dtype', type=str, required=False, help="Data type to use", default="bfloat16")
parser.add_argument('--device', type=str, required=False, help="Device to use", default="cuda")
parser.add_argument('--isoreg-path', '--isoreg_path', dest='isoreg_path', type=str, default=None, help='Path to the Isotonic Regression model directory')

args = parser.parse_args()

print('=' * 50)
print('args:', args)
print('=' * 50)

# Set the GPU number before further imports
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Import from the utils directory
original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    set_visible_cudas, 
    seed_everything,
    get_dtype,
)
from eval import evaluate_by_groups, save_evaluation_results
sys.path = original_sys_path

# Set seed and visible CUDA devices
seed_everything(args.seed)
set_visible_cudas(args.cuda_devices)

original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from ccps import (
    EmbeddingNet,
    ClassifierWithEmbedding
)
sys.path = original_sys_path


def remove_eot_token_ids(data, model_id):
    """Remove end-of-text tokens from the data"""
    def return_eot_token_ids(model_id):
        if "llama" in model_id.lower():
            return [128009]
        elif "mistral" in model_id.lower():
            return [2]
        elif "qwen" in model_id.lower():
            return [151645, 198]
        else:
            logger.warning(f"Unknown model ID: {model_id}, skipping EOT token removal")
            return []
    
    if 'token_id' not in data.columns:
        logger.warning("token_id column not found, skipping EOT token removal")
        return data
    
    eot_token_ids = return_eot_token_ids(model_id)
    if eot_token_ids:
        data = data[~data['token_id'].isin(eot_token_ids)]
    return data

def load_answered_task_data(task, data_subdir, model_id, test_data_dir):
    """
    Loads the previously saved answered and assessed task data.
    Assumes the file is saved as 'answered_data.pkl' under:
      {test_data_dir}/{task}/answered/{data_subdir}/{model_id_replaced}/
    """
    model_id_dir = model_id.replace('/', '-')
    data_path = os.path.join(test_data_dir, task, "answered", data_subdir, model_id_dir, "task_data_answered.pkl")
    print("Loading assessed task data from:", data_path)
    with open(data_path, "rb") as f:
        task_data = pickle.load(f)
    print('Loaded', len(task_data), 'records')
    unique_hash_ids = list(set([item['hash_id'] for item in task_data]))
    print(f"Source - Count of unique hash_ids: {len(unique_hash_ids)}")
    # print(f"Source - hash_ids: {unique_hash_ids}")
    return task_data

def load_model_config(model_path):
    """Load model configuration from a JSON file"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_contrastive_model(contrastive_model_path, device, dtype):
    """Load the pretrained contrastive model"""
    print("Loading contrastive model from:", contrastive_model_path)
    config = load_model_config(contrastive_model_path)
    model_config = config["model"]
    
    # Create model with the same architecture
    input_dim = model_config["input_dim"]
    hidden_dims = model_config["hidden_dims"]
    embed_dim = model_config["embed_dim"]
    activation = config.get("activation", "relu")
    dropout = config.get("dropout", 0.1)
    
    model = EmbeddingNet(input_dim, embed_dim, hidden_dims, activation, dropout)
    
    # Load model weights
    model_path = os.path.join(contrastive_model_path, "contrastive_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device, dtype=dtype)
    model.eval()
    
    print(f"Contrastive model loaded successfully.")
    
    # Load scaler
    scaler_path = os.path.join(contrastive_model_path, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    columns_path = os.path.join(contrastive_model_path, "feature_columns.json")
    with open(columns_path, "r") as f:
        feature_columns = json.load(f)
    
    return model, scaler, feature_columns, embed_dim

def load_classifier_model(classifier_model_path, contrastive_model, embed_dim, device, dtype):
    """Load the pretrained classifier model"""
    print("Loading classifier model from:", classifier_model_path)
    config = load_model_config(classifier_model_path)
    
    # Parse classifier hidden dimensions if provided
    classifier_hidden_dims = None
    classifier_hidden_dims_str = config.get("model", {}).get("classifier_hidden_dims") or config.get("classifier_hidden_dims")
    if classifier_hidden_dims_str:
        if isinstance(classifier_hidden_dims_str, str):
            classifier_hidden_dims = [int(dim) for dim in classifier_hidden_dims_str.split(',')]
        elif isinstance(classifier_hidden_dims_str, list):
            classifier_hidden_dims = classifier_hidden_dims_str
    
    # Get activation and dropout from config
    activation = config.get("model", {}).get("activation") or config.get("activation", "relu")
    dropout = config.get("model", {}).get("dropout") or config.get("dropout", 0.1)
    use_dropout = config.get("model", {}).get("use_dropout") or config.get("use_dropout", False)
    
    # Create classifier model
    model = ClassifierWithEmbedding(
        embedding_model=contrastive_model,
        embed_dim=embed_dim,
        hidden_dims=classifier_hidden_dims,
        num_classes=2,
        activation=activation,
        dropout=dropout,
        use_dropout=use_dropout
    )
    
    # Load model weights
    model_path = os.path.join(classifier_model_path, "classifier_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device, dtype=dtype)
    model.eval()
    
    print(f"Classifier model loaded successfully.")
    return model

def load_test_features(feature_dir, model_id, test_dataset_name):
    """Load the test features - supports both new and old formats"""
    model_dir_name = model_id.replace('/', '-')
    test_features = None
    
    # Try new format first: {feature_dir}/{model_id}/{test_dataset_name}_features.csv or .pkl
    feature_file = os.path.join(feature_dir, model_dir_name, f"{test_dataset_name}_features.csv")
    if os.path.exists(feature_file):
        print(f"Loading test features from CSV (new format): {feature_file}")
        test_features = pd.read_csv(feature_file)
    else:
        pickle_file = os.path.join(feature_dir, model_dir_name, f"{test_dataset_name}_features.pkl")
        if os.path.exists(pickle_file):
            print(f"Loading test features from pickle (new format): {pickle_file}")
            with open(pickle_file, "rb") as f:
                test_features = pickle.load(f)
    
    # Fallback to old format: {feature_dir}/{test_dataset_name}/{model_id}/{task}/llm_output_features.pkl
    # Note: This would require task parameter, so we'll skip it for now and require new format
    if test_features is None:
        raise FileNotFoundError(
            f"Test features not found. Expected:\n"
            f"  New format: {os.path.join(feature_dir, model_dir_name, f'{test_dataset_name}_features.csv')}\n"
            f"  Or: {os.path.join(feature_dir, model_dir_name, f'{test_dataset_name}_features.pkl')}"
        )
    
    # Remove EOT tokens if token_id column exists
    if 'token_id' in test_features.columns:
        test_features = remove_eot_token_ids(test_features, model_id)
    
    print(f"Loaded test features with shape {test_features.shape}")
    
    # Check for hash_id or record_id column
    if 'hash_id' in test_features.columns:
        unique_hash_ids = list(set(test_features['hash_id'].values))
        print(f"Feature - Count of unique hash_ids: {len(unique_hash_ids)}")
    elif 'record_id' in test_features.columns:
        unique_record_ids = list(set(test_features['record_id'].values))
        print(f"Feature - Count of unique record_ids: {len(unique_record_ids)}")
    
    return test_features

def process_features(test_features, exclude_cols, scaler, feature_columns):
    """Process and scale features"""
    # Extract features
    print("Processing features...")
    print("Original features shape:", test_features.shape)
    
    # Only exclude columns that exist
    exclude_cols = [col for col in exclude_cols if col in test_features.columns]
    feat_df = test_features.drop(columns=exclude_cols)
    
    # Handle inf values
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Handle NaN values - fill with 0 for now
    feat_df.fillna(0, inplace=True)
    
    mask = feat_df.notna().all(axis=1)
    print("Valid features shape:", feat_df[mask].shape)
    feat_df = feat_df[mask]
    
    # Check if feat_df has all the feature columns
    missing_cols = [col for col in feature_columns if col not in feat_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test features: {missing_cols}")
    
    # Reorder columns to match the trained model
    feat_df = feat_df[feature_columns]
    
    # Transform using the same scaler
    features_scaled = scaler.transform(feat_df)
    
    # Get hash_ids or record_ids corresponding to valid features
    if 'hash_id' in test_features.columns:
        hash_ids = test_features.loc[mask, 'hash_id'].values
    elif 'record_id' in test_features.columns:
        hash_ids = test_features.loc[mask, 'record_id'].values
    else:
        # Fallback to index
        hash_ids = test_features.loc[mask].index.values
    
    # Get original indices for mapping back
    if isinstance(test_features.index, pd.RangeIndex):
        indices = np.where(mask)[0].tolist()
    else:
        indices = mask.index[mask].tolist()
    
    return features_scaled, hash_ids, indices, mask

def evaluate_test_data(classifier_model, test_features, batch_size, device, dtype):
    """Evaluate test data with the classifier model"""
    model = classifier_model
    model.eval()
    
    all_logits = []
    all_probs = []
    
    dataset_size = len(test_features)
    
    # Process data in batches
    with torch.no_grad():
        for i in range(0, dataset_size, batch_size):
            batch_end = min(i + batch_size, dataset_size)
            batch_features = torch.tensor(test_features[i:batch_end], dtype=dtype).to(device)
            
            logits, _ = model(batch_features)
            probs = F.softmax(logits, dim=1)
            
            all_logits.append(logits.cpu().float().numpy())
            all_probs.append(probs.cpu().float().numpy())
    
    # Concatenate results
    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    
    return all_logits, all_probs

def match_features_to_task_data(task_data, hash_ids, all_logits, all_probs):
    """Match features to task data and add predictions"""
    print(f"Matching features to task data for {len(hash_ids)} records")
    task_data_with_predictions = copy.deepcopy(task_data)
    
    # Create a map from hash_id to predictions for fast lookup
    hash_id_to_preds = {}
    for i, hash_id in enumerate(hash_ids):
        hash_id_to_preds[hash_id] = {
            'logits': all_logits[i].tolist(),
            'p_false': float(all_probs[i, 0]),
            'p_true': float(all_probs[i, 1])
        }
    
    # Add predictions to task data
    matched_count = 0
    for item in task_data_with_predictions:
        hash_id = item['hash_id']
        if hash_id in hash_id_to_preds:
            preds = hash_id_to_preds[hash_id]
            item['logits'] = preds['logits']
            item['p_false'] = preds['p_false']
            item['p_true'] = preds['p_true']
            matched_count += 1
    
    print(f"Matched {matched_count} records out of {len(task_data_with_predictions)}")
    
    # Check if all records were matched
    if matched_count < len(task_data_with_predictions):
        print(f"Warning: {len(task_data_with_predictions) - matched_count} records were not matched with features")
    
    return task_data_with_predictions

def setup_path():
    """Setup paths for models and results - matching PIK/PTRUE structure"""
    model_dir_name = args.model_id.replace('/', '-')
    
    # Results directory: {output_dir}/{model_id}/ (matching PIK structure)
    results_dir = os.path.join(args.output_dir, model_dir_name)
    
    # Model paths: {model_path}/{train_dataset_name}_{test_dataset_name}/{model_id}/
    contrastive_model_path = os.path.join(args.contrastive_model_path, f"{args.train_dataset_name}_{args.test_dataset_name}", model_dir_name)
    classifier_model_path = os.path.join(args.classifier_model_path, f"{args.train_dataset_name}_{args.test_dataset_name}", model_dir_name)
    
    # Feature directory: {feature_dir}/{model_id}/ (features are stored per model, not per dataset)
    feature_dir = args.feature_dir
    
    return results_dir, contrastive_model_path, classifier_model_path, feature_dir

def print_results_summary(results: Dict[str, Any], split_name: str) -> None:
    """Print a summary of evaluation results."""
    print(f"\n{split_name.upper()} CCPS EVALUATION SUMMARY:")
    print("=" * 60)
    
    # Overall results
    if 'overall' in results:
        overall = results['overall']
        print(f"Overall Performance:")
        print(f"  Samples: {overall['n_samples']}")
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
    
    # CCPS statistics
    if 'metadata' in results and 'ccps_statistics' in results['metadata']:
        stats = results['metadata']['ccps_statistics']
        print(f"\nCCPS Classification Statistics:")
        print(f"  Average confidence (p_true): {stats['avg_confidence']:.3f}")
        print(f"  Average p_false: {stats['avg_p_false']:.3f}")
        print(f"  Confidence std: {stats['confidence_std']:.3f}")
        print(f"  Confidence range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
    
    # Dataset breakdown
    if 'by_dataset' in results:
        print(f"\nDataset Breakdown:")
        for dataset, metrics in results['by_dataset'].items():
            print(f"  {dataset}: {metrics['n_samples']} samples, "
                  f"Acc: {metrics['accuracy']:.3f}, ECE: {metrics['ece']:.3f}")
    
    print("=" * 60)


def main():
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    results_dir, contrastive_model_path, classifier_model_path, feature_dir = setup_path()
    # Load models
    contrastive_model, scaler, feature_columns, embed_dim = load_contrastive_model(
        contrastive_model_path, device, dtype
    )
    
    classifier_model = load_classifier_model(
        classifier_model_path, contrastive_model, embed_dim, device, dtype
    )
    
    # Define columns to exclude for feature processing
    exclude_cols = [
        'record_id', 'dataset_name', 'correctness', 'sidx',
        'hash_id', 'task_name', 'sample_idx_in_task', 'token_idx_in_response',
        'token_str', 'token_id', 'query_label_sample', 'answer_type',
        'pei_curve_token', 'wrong_answer_idx',
    ]
    
    # Load test features (new format: single file per model)
    print('=' * 20, f"Loading test features for {args.test_dataset_name}", '=' * 20)
    test_features = load_test_features(feature_dir, args.model_id, args.test_dataset_name)
    
    if test_features is None or test_features.empty:
        print(f"Error: No test features loaded. Exiting.")
        return
    
    # Process and scale features
    features_scaled, record_ids, indices, mask = process_features(test_features, exclude_cols, scaler, feature_columns)
    
    # Evaluate test data
    print('=' * 20, "Evaluating test data", '=' * 20)
    all_logits, all_probs = evaluate_test_data(classifier_model, features_scaled, args.batch_size, device, dtype)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine dataset name based on split_name (matching PIK/PTRUE pattern)
    split_name = args.test_dataset_name
    dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
    
    # Prepare evaluation records in PIK/PTRUE format
    print('=' * 20, "Preparing evaluation records", '=' * 20)
    evaluation_records = []
    
    if 'correctness' in test_features.columns:
        ground_truth_labels = test_features.loc[mask, 'correctness'].values.astype(int)
        confidence_scores = all_probs[:, 1]  # Use p_true as confidence score
        
        for i, record_id in enumerate(record_ids):
            evaluation_record = {
                'record_id': str(record_id),
                'dataset': dataset_name,
                'category': 'all',  # Dummy category for MCQA (matching PIK/PTRUE pattern)
                'ground_truth_correctness': int(ground_truth_labels[i]),
                'confidence_score': float(confidence_scores[i]),
                'original_result': {
                    'record_id': str(record_id),
                    'logits': all_logits[i].tolist(),
                    'p_false': float(all_probs[i, 0]),
                    'p_true': float(all_probs[i, 1])
                }
            }
            evaluation_records.append(evaluation_record)
        
        # Save test_labels.json (matching PIK format)
        labels_file = os.path.join(results_dir, f"{split_name}_labels.json")
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
        print(f"Labels saved to: {labels_file}")
        
        # Compute and save evaluation metrics
        print('=' * 20, "Computing evaluation metrics", '=' * 20)
        ground_truth_array = np.array([r['ground_truth_correctness'] for r in evaluation_records])
        confidence_array = np.array([r['confidence_score'] for r in evaluation_records])
        datasets_array = np.array([r['dataset'] for r in evaluation_records])
        categories_array = np.array([r['category'] for r in evaluation_records])
        
        # Calculate comprehensive evaluation results
        evaluation_results = evaluate_by_groups(ground_truth_array, confidence_array, datasets_array, categories_array)
        
        # Add total samples count
        evaluation_results['overall']['n_total_samples'] = len(test_features)
        
        # Add metadata (matching PIK format)
        evaluation_results['metadata'] = {
            'model_id': args.model_id,
            'split_name': split_name,
            'dataset': dataset_name,
            'total_records': len(evaluation_records),
            'unique_datasets': list(set(r['dataset'] for r in evaluation_records)),
            'evaluation_timestamp': str(np.datetime64('now')),
            'ccps_statistics': {
                'avg_confidence': float(np.mean(confidence_scores)),
                'confidence_std': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores)),
                'avg_p_false': float(np.mean(all_probs[:, 0])),
                'avg_p_true': float(np.mean(all_probs[:, 1]))
            },
            'ccps_model_info': {
                'contrastive_model_path': contrastive_model_path,
                'classifier_model_path': classifier_model_path,
                'train_dataset_name': args.train_dataset_name,
                'test_dataset_name': args.test_dataset_name
            }
        }
        
        # Save test_results.json (matching PIK format)
        results_file = os.path.join(results_dir, f"{split_name}_results.json")
        save_evaluation_results(evaluation_results, results_file)
        print(f"Results saved to: {results_file}")
        
        # Print summary
        print_results_summary(evaluation_results, split_name)
    else:
        print("Warning: No correctness labels found. Skipping metric computation.")
        # Still save labels without ground truth if needed
        evaluation_records = []
        for i, record_id in enumerate(record_ids):
            evaluation_record = {
                'record_id': str(record_id),
                'dataset': dataset_name,
                'category': 'all',
                'confidence_score': float(all_probs[i, 1]),
                'original_result': {
                    'record_id': str(record_id),
                    'logits': all_logits[i].tolist(),
                    'p_false': float(all_probs[i, 0]),
                    'p_true': float(all_probs[i, 1])
                }
            }
            evaluation_records.append(evaluation_record)
        
        labels_file = os.path.join(results_dir, f"{split_name}_labels.json")
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
        print(f"Labels saved to: {labels_file}")
    
    print('=' * 20, "Evaluation complete", '=' * 20)

if __name__ == "__main__":
    main()

# Qwen/Qwen2.5-0.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# Qwen/Qwen2.5-3B-Instruct

# Example Usage:
# python ccps_evaluation.py \
#     --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
#     --cuda-devices "0" \
#     --test-dataset-name test \
#     --train-dataset-name train \
#     --feature-dir ../features/OrigPert \
#     --contrastive-model-path ../trained_models/CCPS/contrastive \
#     --classifier-model-path ../trained_models/CCPS/classifier \
#     --output-dir ../results/CCPS \
#     --batch-size 64 \
#     --dtype bfloat16