#!/usr/bin/env python3
"""
Script to extract comprehensive features from OrigPert hidden state representations.
This processes saved .npz files and extracts features for each token.
"""

import gc
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from general import set_visible_cudas

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_extraction.log")
    ]
)
logger = logging.getLogger(__name__)


class OrigPertFeatureExtractor:
    def __init__(self, model_id: str, cuda_devices: str, dtype_str: str = 'float16',
                 magistral_system_prompt_path: str = "../utils/MAGISTRAL_SYSTEM_PROMPT_SHORT.txt"):
        """
        Initialize the OrigPert feature extractor.
        
        Args:
            model_id: Model ID to use for feature extraction
            cuda_devices: CUDA devices to use (e.g., "0,1,2,3")
            dtype_str: Model dtype ('float16', 'bfloat16', or 'float32')
            magistral_system_prompt_path: Path to the Magistral system prompt file
        """
        self.model_id = model_id
        self.magistral_system_prompt_path = magistral_system_prompt_path
        
        # Set CUDA devices before loading model
        set_visible_cudas(cuda_devices)
        
        # Convert dtype string to torch dtype
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"CUDA devices: {cuda_devices}")
        logger.info(f"Model dtype: {dtype_str}")
        
        # Load tokenizer
        if model_id == "mistralai/Magistral-Small-2506":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "unsloth/magistral-small-2506-unsloth-bnb-4bit", 
                trust_remote_code=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with auto device mapping
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
    
    def load_record_data(self, npz_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single record's data from .npz file"""
        if not npz_path.exists():
            return None
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Extract arrays
            original_hidden_states = data['original_hidden_states']
            original_logits = data['original_logits']
            jacobian_vectors = data['jacobian_vectors']
            perturbed_hidden_states = data['perturbed_hidden_states']
            perturbed_logits = data['perturbed_logits']
            
            # Debug: log shapes
            logger.debug(f"Loaded shapes - hidden_states: {original_hidden_states.shape}, "
                        f"logits: {original_logits.shape}, "
                        f"jacobian: {jacobian_vectors.shape}, "
                        f"perturbed_hidden: {perturbed_hidden_states.shape}")
            
            # Extract metadata
            metadata_json = data['metadata_json'].item() if isinstance(data['metadata_json'].item(), str) else str(data['metadata_json'].item())
            metadata = json.loads(metadata_json)
            
            # Extract other fields
            record_id = str(data['record_id'].item()) if 'record_id' in data else None
            dataset_name = str(data['dataset_name'].item()) if 'dataset_name' in data else None
            correctness = int(data['correctness'].item()) if 'correctness' in data else -1
            sidx = str(data['sidx'].item()) if 'sidx' in data else None
            
            return {
                'original_hidden_states': original_hidden_states,
                'original_logits': original_logits,
                'jacobian_vectors': jacobian_vectors,
                'perturbed_hidden_states': perturbed_hidden_states,
                'perturbed_logits': perturbed_logits,
                'metadata': metadata,
                'record_id': record_id,
                'dataset_name': dataset_name,
                'correctness': correctness,
                'sidx': sidx
            }
        except Exception as e:
            logger.error(f"Error loading record from {npz_path}: {e}")
            return None
    
    def generate_perturbed_logits_batch(self, perturbed_hidden_states: np.ndarray) -> torch.Tensor:
        """Generate logits for all perturbed hidden states in one batch"""
        with torch.no_grad():
            # Convert numpy array to tensor and move to device
            if isinstance(perturbed_hidden_states, np.ndarray):
                perturbed_hidden_states = torch.from_numpy(perturbed_hidden_states).to(self.model.device)
            
            # Batch process through lm_head
            # Shape: [num_perturbations, hidden_size]
            if perturbed_hidden_states.dim() == 2:
                perturbed_hidden_states = perturbed_hidden_states.unsqueeze(1)  # [batch, 1, hidden_size]
            
            # Process in batch
            lm_head_device = self.model.lm_head.weight.device
            logits = self.model.lm_head(perturbed_hidden_states.to(lm_head_device)).squeeze(1)  # [batch, vocab_size]
        
        return logits
    
    def extract_comprehensive_features(self, record_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all comprehensive features for all tokens in a record"""
        all_features = []
        
        original_hidden_states = record_data['original_hidden_states']
        original_logits = record_data['original_logits']
        jacobian_vectors = record_data['jacobian_vectors']
        perturbed_hidden_states = record_data['perturbed_hidden_states']
        perturbed_logits = record_data['perturbed_logits']
        metadata = record_data['metadata']
        
        # Process each token
        for token_meta in metadata:
            token_idx = token_meta['token_idx']
            token_id = token_meta['token_id']
            token_str = token_meta['token_str']
            
            # Get indices
            orig_hidden_idx = token_meta['original_hidden_state_idx']
            orig_logits_idx = token_meta['original_logits_idx']
            jacobian_idx = token_meta['jacobian_idx']
            perturb_start = token_meta['perturbed_start_idx']
            perturb_end = token_meta['perturbed_end_idx']
            
            # Validate indices
            if orig_hidden_idx >= len(original_hidden_states):
                logger.error(f"Hidden state index {orig_hidden_idx} out of bounds (size: {len(original_hidden_states)})")
                continue
            if orig_logits_idx >= len(original_logits):
                logger.error(f"Logits index {orig_logits_idx} out of bounds (size: {len(original_logits)}, shape: {original_logits.shape})")
                continue
            if jacobian_idx >= len(jacobian_vectors):
                logger.error(f"Jacobian index {jacobian_idx} out of bounds (size: {len(jacobian_vectors)})")
                continue
            
            # Get data for this token
            original_hidden = original_hidden_states[orig_hidden_idx]
            
            # Handle logits shape: could be (num_tokens, vocab_size) or (num_tokens, 1, vocab_size)
            # Check the shape of original_logits array
            logits_slice = original_logits[orig_logits_idx]
            if logits_slice.ndim == 2 and logits_slice.shape[0] == 1:
                # Shape is (1, vocab_size), squeeze the first dimension
                original_logits_token = logits_slice.squeeze(0)
            elif logits_slice.ndim == 1:
                # Shape is (vocab_size,)
                original_logits_token = logits_slice
            else:
                # Try to flatten to 1D
                original_logits_token = logits_slice.flatten()
                logger.warning(f"Unexpected logits slice shape: {logits_slice.shape}, flattened to {original_logits_token.shape}")
            
            jacobian = jacobian_vectors[jacobian_idx]
            
            # Get perturbed hidden states for this token
            token_perturbed_hidden = perturbed_hidden_states[perturb_start:perturb_end+1]
            
            # Get perturbation metadata
            perturbation_metadata = token_meta['perturbation_metadata']
            
            # Extract features
            features = self._extract_token_features(
                original_hidden, original_logits_token, jacobian,
                token_perturbed_hidden, token_id,
                perturbation_metadata
            )
            
            # Add metadata
            features.update({
                'record_id': record_data['record_id'],
                'dataset_name': record_data['dataset_name'],
                'correctness': record_data['correctness'],
                'sidx': record_data.get('sidx'),
                'token_idx_in_response': token_idx,
                'token_str': token_str,
                'token_id': token_id
            })
            
            all_features.append(features)
        
        return all_features
    
    def _extract_token_features(self, original_hidden_state: np.ndarray, original_logits: np.ndarray,
                               jacobian_vector: np.ndarray, perturbed_hidden_states: np.ndarray,
                               actual_token_id: int, perturbation_metadata: List[Dict]) -> Dict[str, Any]:
        """Extract all comprehensive features for a single token"""
        features = {}
        
        # Convert to torch tensors
        H_0 = torch.from_numpy(original_hidden_state).to(self.model.device)
        # Ensure logits is 1D: (vocab_size,)
        if original_logits.ndim > 1:
            original_logits = original_logits.squeeze()
        L_0 = torch.from_numpy(original_logits).to(self.model.device).unsqueeze(0)  # Add batch dim: (1, vocab_size)
        J_T = torch.from_numpy(jacobian_vector).to(self.model.device)
        
        # Generate all perturbed logits at once (if not already computed)
        # Note: perturbed_logits might already be in the data, but we'll regenerate for consistency
        perturbed_logits_tensor = self.generate_perturbed_logits_batch(perturbed_hidden_states)
        
        # I. Features from Original State
        probs_0 = torch.softmax(L_0, dim=-1).squeeze()
        log_probs_0 = torch.log_softmax(L_0, dim=-1).squeeze()
        
        # Get top-2 indices
        top2_values, top2_indices = torch.topk(L_0.squeeze(), 2)
        argmax_0 = top2_indices[0].item()
        second_best_0 = top2_indices[1].item() if len(top2_indices) > 1 else argmax_0
        
        features['original_log_prob_actual'] = log_probs_0[actual_token_id].item()
        features['original_prob_actual'] = probs_0[actual_token_id].item()
        features['original_logit_actual'] = L_0[0, actual_token_id].item()
        features['original_prob_argmax'] = probs_0[argmax_0].item()
        features['original_logit_argmax'] = L_0[0, argmax_0].item()
        
        # Entropy
        features['original_entropy'] = -torch.sum(probs_0 * torch.log(probs_0 + 1e-6)).item()
        
        # Margins
        features['original_margin_logit_top1_top2'] = L_0[0, argmax_0].item() - L_0[0, second_best_0].item()
        features['original_margin_prob_top1_top2'] = probs_0[argmax_0].item() - probs_0[second_best_0].item()
        
        # Norms
        features['original_norm_logits_L2'] = torch.norm(L_0).item()
        features['original_std_logits'] = torch.std(L_0).item()
        features['original_norm_hidden_state_L2'] = torch.norm(H_0).item()
        
        # Boolean features
        features['is_actual_token_original_argmax'] = int(actual_token_id == argmax_0)
        
        # II. Overall Perturbation Metrics
        features['jacobian_norm_token'] = torch.norm(J_T).item()
        
        # Calculate epsilon-to-flip using the pre-generated logits
        original_argmax = torch.argmax(L_0).item()
        epsilon_to_flip = float('inf')
        for i, metadata in enumerate(perturbation_metadata):
            if torch.argmax(perturbed_logits_tensor[i]).item() != original_argmax:
                epsilon_to_flip = metadata['perturbation_radius']
                break
        features['epsilon_to_flip_token'] = epsilon_to_flip
        
        # Calculate PEI using pre-generated logits
        log_probs_original = torch.log_softmax(L_0, dim=-1)
        log_p_original = log_probs_original[0, actual_token_id].item()
        f_values = [0.0]
        
        perturbed_log_probs = torch.log_softmax(perturbed_logits_tensor, dim=-1)
        for i in range(len(perturbed_hidden_states)):
            log_p_perturbed = perturbed_log_probs[i, actual_token_id].item()
            f_k = log_p_original - log_p_perturbed
            f_values.append(max(0.0, f_k))
        
        pei_steps = len(perturbed_hidden_states)
        pei_value = 0.0
        for k in range(pei_steps):
            pei_value += (f_values[k] + f_values[k+1]) / 2.0
        pei_value = pei_value / pei_steps if pei_steps > 0 else 0.0
        
        features['pei_value_token'] = pei_value
        
        # III. Features from Perturbed States
        perturbed_features = {
            'perturbed_log_prob_actual': [],
            'perturbed_prob_actual': [],
            'perturbed_logit_actual': [],
            'delta_log_prob_actual_from_original': [],
            'perturbed_prob_argmax': [],
            'perturbed_logit_argmax': [],
            'did_argmax_change_from_original': [],
            'perturbed_entropy': [],
            'perturbed_margin_logit_top1_top2': [],
            'perturbed_norm_logits_L2': [],
            'kl_div_perturbed_from_original': [],
            'js_div_perturbed_from_original': [],
            'cosine_sim_logits_perturbed_to_original': [],
            'cosine_sim_hidden_perturbed_to_original': [],
            'l2_dist_hidden_perturbed_from_original': []
        }
        
        # Process all perturbations using pre-generated logits
        perturbed_probs = torch.softmax(perturbed_logits_tensor, dim=-1)
        
        for i in range(len(perturbed_hidden_states)):
            H_p = torch.from_numpy(perturbed_hidden_states[i]).to(self.model.device)
            L_p = perturbed_logits_tensor[i].to(self.model.device)
            probs_p = perturbed_probs[i].to(self.model.device)
            log_probs_p = perturbed_log_probs[i].to(self.model.device)
            
            # Basic metrics
            perturbed_features['perturbed_log_prob_actual'].append(log_probs_p[actual_token_id].item())
            perturbed_features['perturbed_prob_actual'].append(probs_p[actual_token_id].item())
            perturbed_features['perturbed_logit_actual'].append(L_p[actual_token_id].item())
            
            # Delta from original
            delta_log_prob = log_probs_0[actual_token_id].item() - log_probs_p[actual_token_id].item()
            perturbed_features['delta_log_prob_actual_from_original'].append(delta_log_prob)
            
            # Argmax features
            argmax_p = torch.argmax(L_p).item()
            perturbed_features['perturbed_prob_argmax'].append(probs_p[argmax_p].item())
            perturbed_features['perturbed_logit_argmax'].append(L_p[argmax_p].item())
            perturbed_features['did_argmax_change_from_original'].append(int(argmax_p != argmax_0))
            
            # Entropy
            entropy_p = -torch.sum(probs_p * torch.log(probs_p + 1e-6)).item()
            perturbed_features['perturbed_entropy'].append(entropy_p)
            
            # Margin
            top2_p = torch.topk(L_p, 2).indices
            if len(top2_p) >= 2:
                margin = L_p[top2_p[0]].item() - L_p[top2_p[1]].item()
                perturbed_features['perturbed_margin_logit_top1_top2'].append(margin)
            else:
                perturbed_features['perturbed_margin_logit_top1_top2'].append(0.0)
            
            # Norms
            perturbed_features['perturbed_norm_logits_L2'].append(torch.norm(L_p).item())
            
            # Divergences
            kl_div = torch.nn.functional.kl_div(log_probs_p, probs_0, reduction='sum').item()
            perturbed_features['kl_div_perturbed_from_original'].append(kl_div)
            
            # JS divergence
            m_probs = 0.5 * (probs_0 + probs_p)
            js_div = 0.5 * torch.nn.functional.kl_div(log_probs_0, m_probs, reduction='sum') + \
                     0.5 * torch.nn.functional.kl_div(log_probs_p, m_probs, reduction='sum')
            perturbed_features['js_div_perturbed_from_original'].append(js_div.item())
            
            # Similarities
            cos_sim_logits = torch.nn.functional.cosine_similarity(L_0.squeeze(), L_p.squeeze(), dim=0).item()
            perturbed_features['cosine_sim_logits_perturbed_to_original'].append(cos_sim_logits)
            
            cos_sim_hidden = torch.nn.functional.cosine_similarity(H_0.squeeze(), H_p.squeeze(), dim=0).item()
            perturbed_features['cosine_sim_hidden_perturbed_to_original'].append(cos_sim_hidden)
            
            # L2 distance
            l2_dist = torch.norm(H_p - H_0).item()
            perturbed_features['l2_dist_hidden_perturbed_from_original'].append(l2_dist)
        
        # Add summary statistics
        for metric_name, values in perturbed_features.items():
            if values:
                features[f'{metric_name}_min'] = min(values)
                features[f'{metric_name}_max'] = max(values)
                features[f'{metric_name}_mean'] = np.mean(values)
                features[f'{metric_name}_std'] = np.std(values) if len(values) > 1 else 0.0
        
        return features
    
    def process_split(self, representations_dir: Path, split_name: str, 
                     output_dir: Path) -> Optional[pd.DataFrame]:
        """Process all records in a split and extract features"""
        split_dir = representations_dir / split_name
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return None
        
        # Find all .npz files
        npz_files = sorted(split_dir.glob("*.npz"))
        
        if not npz_files:
            logger.warning(f"No .npz files found in {split_dir}")
            return None
        
        logger.info(f"Found {len(npz_files)} records in {split_name}")
        
        all_features = []
        
        for npz_file in tqdm(npz_files, desc=f"Processing {split_name}"):
            # Load record data
            record_data = self.load_record_data(npz_file)
            
            if record_data is None:
                logger.warning(f"Failed to load record from {npz_file}")
                continue
            
            # Extract features for all tokens in this record
            token_features = self.extract_comprehensive_features(record_data)
            all_features.extend(token_features)
            
            # Clear GPU cache periodically
            if len(all_features) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        if not all_features:
            logger.warning(f"No features extracted from {split_name}")
            return None
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        logger.info(f"Extracted {len(features_df)} token features from {split_name}")
        
        return features_df
    
    def save_features(self, df: pd.DataFrame, output_dir: Path, filename: str) -> None:
        """Save features to CSV and pickle"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved features to {csv_path}")
        
        # Save as pickle
        pkl_path = output_dir / f"{filename}.pkl"
        df.to_pickle(pkl_path)
        logger.info(f"Saved features to {pkl_path}")


def create_input_address(input_dir: str, model_name: str) -> Path:
    """Create the input directory path."""
    model_name_for_path = model_name.split("/")[-1]
    return Path(input_dir) / model_name_for_path


def create_output_directory(output_dir: str, model_id: str) -> Path:
    """Create the output directory structure."""
    model_dir_name = model_id.replace('/', '-')
    features_dir = Path(output_dir) / model_dir_name
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract comprehensive features from OrigPert hidden state representations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model ID to use for feature extraction")
    parser.add_argument("--cuda-devices", type=str, default="0,1,2,3",
                       help="CUDA devices to use (e.g., '0,1,2,3')")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=['bfloat16', 'float16', 'float32'],
                       help="Model dtype")
    
    # Data configuration
    parser.add_argument("--representations-dir", type=str, required=True,
                       help="Directory containing the saved representations (../representations/OrigPert/{model_id}/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (for directory path, e.g., 'Qwen/Qwen2.5-0.5B-Instruct')")
    parser.add_argument("--output-dir", type=str, default="../features/OrigPert",
                       help="Base output directory for extracted features")
    
    # File-specific options
    parser.add_argument("--train-only", action="store_true",
                       help="Process only training data")
    parser.add_argument("--test-only", action="store_true",
                       help="Process only test data")
    
    # Magistral system prompt path
    parser.add_argument("--magistral-system-prompt-path", type=str, default="../utils/MAGISTRAL_SYSTEM_PROMPT_SHORT.txt",
                       help="Path to the Magistral system prompt file (for Magistral model)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate arguments
    if args.train_only and args.test_only:
        raise ValueError("Cannot specify both --train-only and --test-only")
    
    # Initialize the feature extractor
    extractor = OrigPertFeatureExtractor(
        model_id=args.model_id,
        cuda_devices=args.cuda_devices,
        dtype_str=args.dtype,
        magistral_system_prompt_path=args.magistral_system_prompt_path
    )
    
    # Create input and output directories
    model_dir_name = args.model_id.replace('/', '-')
    representations_dir = Path(args.representations_dir) / model_dir_name
    output_dir = create_output_directory(args.output_dir, extractor.model_id)
    
    logger.info(f"Representations directory: {representations_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process files based on arguments
    if not args.test_only:
        train_features = extractor.process_split(representations_dir, "train", output_dir)
        if train_features is not None and not train_features.empty:
            extractor.save_features(train_features, output_dir, "train_features")
    
    if not args.train_only:
        test_features = extractor.process_split(representations_dir, "test", output_dir)
        if test_features is not None and not test_features.empty:
            extractor.save_features(test_features, output_dir, "test_features")
    
    logger.info(f"\nFeature extraction complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

# Example Usage:
# python origpert_feature_extraction.py \
#     --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
#     --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
#     --cuda-devices "1" \
#     --representations-dir ../representations/OrigPert \
#     --output-dir ../features/OrigPert \
#     --dtype float16