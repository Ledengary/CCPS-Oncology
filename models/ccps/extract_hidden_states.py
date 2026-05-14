#!/usr/bin/env python3
"""
Script to extract original and perturbed hidden states, logits, and Jacobians
for OrigPert (Original and Perturbed) method.
This extracts hidden states and logits for each token in the model's response,
along with perturbed versions based on Jacobian-based perturbations.
"""

import gc
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils.data_io import read_table  # noqa: E402
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from general import set_visible_cudas

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extract_hidden_states.log")
    ]
)
logger = logging.getLogger(__name__)


class OrigPertExtractor:
    def __init__(self, model_id: str, cuda_devices: str, dtype_str: str = 'float16',
                 magistral_system_prompt_path: str = "../utils/MAGISTRAL_SYSTEM_PROMPT_SHORT.txt",
                 debug: bool = False):
        """
        Initialize the OrigPert hidden state extractor.
        
        Args:
            model_id: Model ID to use for hidden state extraction
            cuda_devices: CUDA devices to use (e.g., "0,1,2,3")
            dtype_str: Model dtype ('float16', 'bfloat16', or 'float32')
            magistral_system_prompt_path: Path to the Magistral system prompt file
            debug: Enable debug logging
        """
        self.debug = debug
        self.model_id = model_id
        self.magistral_system_prompt_path = magistral_system_prompt_path
        
        if self.debug:
            logger.info("🐛 DEBUG MODE: Verbose logging enabled")
        
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
    
    def convert_to_chat_template(self, conversation: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        """Convert conversation to chat template format."""
        try:
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            # Fallback formatting
            text = ""
            for item in conversation:
                text += f"{item['role']}: {item['content']}\n"
            if add_generation_prompt and conversation and conversation[-1]['role'] == 'user':
                text += "assistant:"
            return text
    
    def get_jacobian_for_token(self, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
        """Calculate Jacobian vector for a specific token"""
        self.model.zero_grad()
        
        # Forward pass without gradients
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
            hidden_state = outputs.hidden_states[-1][:, -1, :].clone().detach()
        
        # Make slice require grad for lm_head backprop
        hidden_state_grad = hidden_state.requires_grad_(True)
        
        # Pass through lm_head
        lm_head_device = self.model.lm_head.weight.device
        logits = self.model.lm_head(hidden_state_grad.to(lm_head_device).unsqueeze(1))
        logits_for_loss = logits.squeeze(1)
        
        # Calculate loss and backward
        log_probs = torch.log_softmax(logits_for_loss, dim=-1)
        loss = -log_probs[0, token_id]
        
        loss.backward()
        
        if hidden_state_grad.grad is not None:
            jacobian_vector = hidden_state_grad.grad.clone().detach()
        else:
            jacobian_vector = torch.zeros_like(hidden_state_grad)
        
        self.model.zero_grad()
        return jacobian_vector
    
    def extract_hidden_states_and_logits(self, llm_input: str, system_prompt: Optional[str], 
                                       answer_text: str, pei_radius: float, pei_steps: int,
                                       record_id: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract original and perturbed hidden states and logits for a sample
        
        Args:
            llm_input: The input prompt/question
            system_prompt: Optional system prompt
            answer_text: The answer text to extract hidden states for
            pei_radius: Maximum perturbation radius
            pei_steps: Number of perturbation steps
            record_id: Record identifier
            dataset_name: Dataset name
            
        Returns:
            Dictionary with hidden states, logits, and metadata
        """
        if self.debug:
            logger.info(f"\n{'='*80}")
            logger.info(f"EXTRACTING HIDDEN STATES FOR RECORD {record_id}")
            logger.info(f"{'='*80}")
            logger.info(f"📝 LLM INPUT (first 200 chars): {llm_input[:200]}{'...' if len(llm_input) > 200 else ''}")
            logger.info(f"🤖 ANSWER TEXT: {answer_text}")
            logger.info(f"📏 Answer length: {len(answer_text)} characters")
        
        # Build conversation
        conversation = []
        if system_prompt:
            conversation.append({'role': 'system', 'content': system_prompt})
            if self.debug:
                logger.info(f"💬 System prompt present: {len(system_prompt)} chars")
        conversation.append({'role': 'user', 'content': llm_input})
        
        # Create conversation with the answer appended
        conversation_with_response = conversation.copy()
        if answer_text:
            conversation_with_response.append({'role': 'assistant', 'content': answer_text})
        
        prompt_only_str = self.convert_to_chat_template(conversation, add_generation_prompt=True)
        full_prompt_str = self.convert_to_chat_template(conversation_with_response, add_generation_prompt=False)
        
        if self.debug:
            logger.info(f"📊 Prompt-only length: {len(prompt_only_str)} chars")
            logger.info(f"📊 Full prompt length: {len(full_prompt_str)} chars")
        
        # Tokenize
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt_only_str, return_tensors="pt").to(device)
        full_ids = self.tokenizer.encode(full_prompt_str, return_tensors="pt").to(device)
        
        if self.debug:
            logger.info(f"🔢 Input token count: {input_ids.shape[1]}")
            logger.info(f"🔢 Full sequence token count: {full_ids.shape[1]}")
        
        response_start_idx = input_ids.shape[1]
        if full_ids.shape[1] < response_start_idx:
            response_ids = []
        else:
            response_ids = full_ids[0, response_start_idx:].tolist()
        
        if not response_ids:
            logger.warning(f"No response IDs found for record {record_id}. Skipping.")
            return None
        
        if self.debug:
            logger.info(f"✅ Found {len(response_ids)} answer tokens (starting at position {response_start_idx})")
            logger.info(f"🎯 ANSWER TOKENS BEING EXTRACTED:")
            # Show first 10 and last 10 tokens
            for i, token_id in enumerate(response_ids[:10]):
                token_str = self.tokenizer.decode([token_id])
                logger.info(f"   [{i:3d}] Token ID: {token_id:6d} -> {repr(token_str)}")
            if len(response_ids) > 10:
                logger.info(f"   ... ({len(response_ids) - 20} more tokens) ...")
                for i, token_id in enumerate(response_ids[-10:], start=len(response_ids)-10):
                    token_str = self.tokenizer.decode([token_id])
                    logger.info(f"   [{i:3d}] Token ID: {token_id:6d} -> {repr(token_str)}")
            
            # Verify by decoding the full answer
            decoded_answer = self.tokenizer.decode(response_ids)
            logger.info(f"🔍 Decoded answer tokens: {repr(decoded_answer)}")
            logger.info(f"🔍 Original answer text:  {repr(answer_text)}")
            logger.info(f"✅ Match: {decoded_answer.strip() == answer_text.strip()}")
        
        # Store results for all tokens
        original_hidden_states_list = []
        original_logits_list = []
        jacobian_vectors_list = []
        perturbed_hidden_states_list = []
        perturbed_logits_list = []
        metadata_list = []
        
        current_input = input_ids
        
        # Process each token in the response
        for token_idx, token_id in enumerate(response_ids):
            if self.debug and token_idx < 3:  # Show first 3 tokens in detail
                logger.info(f"\n🔄 Processing token {token_idx}/{len(response_ids)-1}")
                token_str = self.tokenizer.decode([token_id])
                logger.info(f"   Token ID: {token_id} -> {repr(token_str)}")
                logger.info(f"   Current input sequence length: {current_input.shape[1]}")
            
            # Get original hidden state and logits
            with torch.no_grad():
                outputs = self.model(current_input, output_hidden_states=True, return_dict=True)
                hidden_state = outputs.hidden_states[-1][:, -1, :].clone()
                lm_head_device = self.model.lm_head.weight.device
                logits = self.model.lm_head(hidden_state.to(lm_head_device).unsqueeze(1))[:, -1, :]
            
            if self.debug and token_idx < 3:
                # Verify the predicted token matches what we expect
                predicted_token_id = torch.argmax(logits).item()
                predicted_token_str = self.tokenizer.decode([predicted_token_id])
                expected_token_id = token_id
                matches = (predicted_token_id == expected_token_id)
                logger.info(f"   Expected token: {expected_token_id} ({repr(self.tokenizer.decode([expected_token_id]))})")
                logger.info(f"   Predicted token: {predicted_token_id} ({repr(predicted_token_str)})")
                logger.info(f"   ✅ Match: {matches}")
            
            # Store original states
            original_hidden_states_list.append(hidden_state.cpu().float().numpy())
            original_logits_list.append(logits.cpu().float().numpy())
            
            # Calculate Jacobian for perturbation
            jacobian_vector = self.get_jacobian_for_token(current_input, token_id)
            jacobian_norm = torch.norm(jacobian_vector)
            
            # Store Jacobian vector
            jacobian_vectors_list.append(jacobian_vector.cpu().float().numpy())
            
            # Initialize lists for this token's perturbations
            token_perturbed_hidden_states = []
            token_perturbed_logits = []
            perturbed_metadata = []
            
            # Generate perturbations
            if jacobian_norm.item() > 1e-9:
                jacobian_direction = jacobian_vector / jacobian_norm
                delta_r = pei_radius / pei_steps
                
                for k in range(1, pei_steps + 1):
                    r_k = k * delta_r
                    
                    # Perturb the hidden state
                    perturbed_hidden = hidden_state + r_k * jacobian_direction
                    
                    # Get logits for perturbed state
                    with torch.no_grad():
                        perturbed_logits = self.model.lm_head(
                            perturbed_hidden.to(self.model.lm_head.weight.device).unsqueeze(1)
                        )[:, -1, :]
                    
                    token_perturbed_hidden_states.append(perturbed_hidden.cpu().float().numpy())
                    token_perturbed_logits.append(perturbed_logits.cpu().float().numpy())
                    
                    perturbed_metadata.append({
                        'perturbation_step': k,
                        'perturbation_radius': r_k,
                        'jacobian_norm': jacobian_norm.item()
                    })
            else:
                # For zero Jacobian, just repeat the original state
                for k in range(1, pei_steps + 1):
                    token_perturbed_hidden_states.append(hidden_state.cpu().float().numpy())
                    token_perturbed_logits.append(logits.cpu().float().numpy())
                    
                    perturbed_metadata.append({
                        'perturbation_step': k,
                        'perturbation_radius': 0.0,
                        'jacobian_norm': 0.0
                    })
            
            perturbed_hidden_states_list.extend(token_perturbed_hidden_states)
            perturbed_logits_list.extend(token_perturbed_logits)
            
            # Store metadata for this token
            token_metadata = {
                'token_idx': token_idx,
                'token_id': token_id,
                'token_str': self.tokenizer.convert_ids_to_tokens(token_id),
                'original_hidden_state_idx': len(original_hidden_states_list) - 1,
                'original_logits_idx': len(original_logits_list) - 1,
                'jacobian_idx': len(jacobian_vectors_list) - 1,
                'perturbed_start_idx': len(perturbed_hidden_states_list) - pei_steps,
                'perturbed_end_idx': len(perturbed_hidden_states_list) - 1,
                'perturbation_metadata': perturbed_metadata,
                'dataset_name': dataset_name
            }
            metadata_list.append(token_metadata)
            
            # Update input for next token
            if token_idx < len(response_ids) - 1:
                next_token_tensor = torch.tensor([[token_id]], device=device)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)
        
        if self.debug:
            logger.info(f"\n✅ EXTRACTION COMPLETE FOR RECORD {record_id}")
            logger.info(f"   Total tokens processed: {len(response_ids)}")
            logger.info(f"   Original hidden states: {len(original_hidden_states_list)}")
            logger.info(f"   Jacobian vectors: {len(jacobian_vectors_list)}")
            logger.info(f"   Perturbed hidden states: {len(perturbed_hidden_states_list)}")
            logger.info(f"{'='*80}\n")
        
        return {
            'original_hidden_states': np.array(original_hidden_states_list),
            'original_logits': np.array(original_logits_list),
            'jacobian_vectors': np.array(jacobian_vectors_list),
            'perturbed_hidden_states': np.array(perturbed_hidden_states_list),
            'perturbed_logits': np.array(perturbed_logits_list),
            'metadata': metadata_list,
            'record_id': record_id,
            'dataset_name': dataset_name
        }
    
    def save_sample_data(self, sample_data: Dict[str, Any], output_file: Path) -> None:
        """Save the extracted data for a single sample"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata to JSON string for npz compatibility
        metadata_json = json.dumps(sample_data['metadata'])
        
        # Prepare save dictionary
        save_dict = {
            'original_hidden_states': sample_data['original_hidden_states'].astype(np.float16),
            'original_logits': sample_data['original_logits'].astype(np.float16),
            'jacobian_vectors': sample_data['jacobian_vectors'].astype(np.float16),
            'perturbed_hidden_states': sample_data['perturbed_hidden_states'].astype(np.float16),
            'perturbed_logits': sample_data['perturbed_logits'].astype(np.float16),
            'metadata_json': metadata_json,  # Save as JSON string
            'record_id': sample_data['record_id'],
            'dataset_name': sample_data['dataset_name'],
            'correctness': np.int32(sample_data.get('correctness', -1))
        }
        
        # Add sidx if available
        if sample_data.get('sidx') is not None:
            save_dict['sidx'] = str(sample_data['sidx'])
        
        # Save as compressed npz
        np.savez_compressed(output_file, **save_dict)
        if self.debug:
            logger.info(f"💾 Saved hidden states and logits to: {output_file}")
    
    def process_dataset(self, csv_path: Path, representations_dir: Path, split_name: str,
                      pei_radius: float, pei_steps: int, limit: Optional[int] = None) -> None:
        """Process an entire CSV file for hidden state extraction."""
        logger.info(f"Processing {csv_path} for OrigPert hidden state extraction")
        
        if not csv_path.exists():
            logger.error(f"Input file does not exist: {csv_path}")
            return
        
        # Determine dataset name based on split_name (matching PIK.py pattern)
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        
        # Create output directory
        output_dir = representations_dir / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read CSV
        df = read_table(csv_path)
        
        # Validate required columns
        required_columns = ['llm_input', 'correctness']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return
        
        # Check for llm_output column (required for extraction)
        if 'llm_output' not in df.columns:
            logger.error(f"Missing 'llm_output' column. Cannot extract hidden states without model responses.")
            return
        
        # Check for sidx column (will be stored but not used as record_id to avoid duplicates)
        has_sidx = 'sidx' in df.columns
        if not has_sidx:
            logger.info(f"No 'sidx' column found in {csv_path}. Using row number as record_id.")
        
        # Get system_prompt if available
        has_system_prompt = 'system_prompt' in df.columns
        
        # Handle Magistral system prompt
        magistral_system_prompt = None
        if self.model_id == "mistralai/Magistral-Small-2506":
            try:
                prompt_path = Path(self.magistral_system_prompt_path)
                if prompt_path.exists():
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        magistral_system_prompt = f.read()
                    logger.info("Loaded Magistral system prompt.")
                else:
                    logger.warning(f"Magistral system prompt file not found: {prompt_path}")
            except Exception as e:
                logger.warning(f"Could not load Magistral system prompt: {e}")
        
        # Apply limit if specified
        if limit is not None:
            df = df.head(limit)
        
        total_records = len(df)
        logger.info(f"Loaded {total_records} records from {csv_path}")
        
        # Check which records are already processed (using row number as record_id)
        processed_count = 0
        for i, row in df.iterrows():
            record_id = str(i)  # Always use row number
            output_file = output_dir / f"{record_id}.npz"
            if output_file.exists():
                processed_count += 1
        
        logger.info(f"Found {processed_count}/{total_records} records already processed")
        logger.info(f"Need to process {total_records - processed_count} remaining records")
        
        # Process each record (using row number as record_id to ensure uniqueness)
        records_to_process = []
        for i, row in df.iterrows():
            records_to_process.append((i, row))
        
        total_records_to_process = len(records_to_process)
        logger.info(f"Processing {total_records_to_process} records")
        
        with tqdm(total=total_records_to_process, desc=f"Processing {split_name}") as pbar:
            for record_idx, (i, row) in enumerate(records_to_process, 1):
                try:
                    # Use row number as record_id to ensure uniqueness
                    record_id = str(i)
                    output_file = output_dir / f"{record_id}.npz"
                    
                    # Skip if already processed
                    if output_file.exists():
                        logger.debug(f"Skipping already processed record: {record_id}")
                        pbar.update(1)
                        continue
                    
                    if self.debug:
                        logger.info(f"\nProcessing record {record_idx}/{total_records_to_process}: {record_id}")
                    else:
                        # Only show progress every 10 records if not in debug mode
                        if record_idx % 10 == 0:
                            logger.info(f"Processing record {record_idx}/{total_records_to_process}")
                    
                    # Extract prompt from llm_input
                    llm_input = str(row['llm_input'])
                    
                    # Get llm_output (required)
                    llm_output = str(row['llm_output']).strip()
                    if not llm_output:
                        logger.warning(f"Empty llm_output for record {record_id}. Skipping.")
                        pbar.update(1)
                        continue
                    
                    # Get correctness (binary 0/1)
                    correctness = int(row['correctness'])
                    
                    # Get system_prompt if available
                    system_prompt = None
                    if has_system_prompt:
                        system_prompt = str(row['system_prompt'])
                        if system_prompt.strip() == '':
                            system_prompt = None
                    
                    # Use Magistral system prompt if model is Magistral and no system prompt provided
                    if not system_prompt and self.model_id == "mistralai/Magistral-Small-2506":
                        system_prompt = magistral_system_prompt
                    
                    # Extract hidden states and logits
                    sample_data = self.extract_hidden_states_and_logits(
                        llm_input=llm_input,
                        system_prompt=system_prompt,
                        answer_text=llm_output,
                        pei_radius=pei_radius,
                        pei_steps=pei_steps,
                        record_id=record_id,
                        dataset_name=dataset_name
                    )
                    
                    if sample_data is not None:
                        # Add additional metadata
                        sample_data['correctness'] = correctness
                        sample_data['sidx'] = str(row['sidx']) if has_sidx else None
                        
                        # Save hidden state
                        self.save_sample_data(sample_data, output_file)
                        if self.debug:
                            logger.info(f"✅ Successfully processed record: {record_id}")
                    else:
                        logger.warning(f"Failed to extract hidden states for record: {record_id}")
                    
                    pbar.update(1)
                    
                    # Clear GPU cache periodically
                    if record_idx % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing record {i}: {e}")
                    pbar.update(1)
                    continue
        
        logger.info(f"Completed processing {split_name} (dataset: {dataset_name})")


def create_input_address(input_dir: str, model_name: str, train_file: str, test_file: str) -> Tuple[Path, Path]:
    """Create the input file paths."""
    # Extract just the model name (last part after /) for directory paths
    model_name_for_path = model_name.split("/")[-1]
    input_dir_path = Path(input_dir) / model_name_for_path
    train_path = input_dir_path / train_file
    test_path = input_dir_path / test_file
    return train_path, test_path


def create_output_directory(representations_dir: str, model_id: str) -> Path:
    """Create the output directory structure."""
    model_dir_name = model_id.replace('/', '-')
    origpert_dir = Path(representations_dir) / model_dir_name
    origpert_dir.mkdir(parents=True, exist_ok=True)
    return origpert_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract original and perturbed hidden states, logits, and Jacobians for OrigPert method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model ID to use for hidden state extraction")
    parser.add_argument("--cuda-devices", type=str, default="0,1,2,3",
                       help="CUDA devices to use (e.g., '0,1,2,3')")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=['bfloat16', 'float16', 'float32'],
                       help="Model dtype")
    
    # Data configuration
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing the input CSV files (labeled_v2/{model_name}/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (directory name in input_dir, e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    parser.add_argument("--representations-dir", type=str, default="../representations/OrigPert",
                       help="Base output directory for hidden state representations")
    
    # Input file patterns
    parser.add_argument("--train-file", type=str, default="ehrnoteqa_train_mcqa_lbl.csv",
                       help="Training CSV file name (default: ehrnoteqa_train_mcqa_lbl.csv)")
    parser.add_argument("--test-file", type=str, default="CORTEX_contextual_labeled.jsonl",
                       help="Test CSV file name (default: CORTEX_contextual_labeled.jsonl)")
    
    # Processing configuration
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of records to process per file (for testing)")
    
    # PEI configuration
    parser.add_argument("--pei-radius", type=float, default=10.0,
                       help="Maximum perturbation radius for PEI calculation")
    parser.add_argument("--pei-steps", type=int, default=5,
                       help="Number of steps for PEI integration")
    
    # File-specific options
    parser.add_argument("--train-only", action="store_true",
                       help="Process only training data")
    parser.add_argument("--test-only", action="store_true",
                       help="Process only test data")
    
    # Magistral system prompt path
    parser.add_argument("--magistral-system-prompt-path", type=str, default="../utils/MAGISTRAL_SYSTEM_PROMPT_SHORT.txt",
                       help="Path to the Magistral system prompt file (for Magistral model)")
    
    # Debug option
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode for verbose logging")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate arguments
    if args.train_only and args.test_only:
        raise ValueError("Cannot specify both --train-only and --test-only")
    
    # Initialize the hidden state extractor
    extractor = OrigPertExtractor(
        model_id=args.model_id,
        cuda_devices=args.cuda_devices,
        dtype_str=args.dtype,
        magistral_system_prompt_path=args.magistral_system_prompt_path,
        debug=args.debug
    )
    
    # Log debug mode status
    if args.debug:
        logger.info("🐛 DEBUG MODE ENABLED - Verbose logging active")
    else:
        logger.info("INFO: Use --debug flag for detailed token extraction information")
    
    # Create output directory structure
    origpert_dir = create_output_directory(args.representations_dir, extractor.model_id)
    
    logger.info(f"OrigPert directory: {origpert_dir}")
    
    # Define input files
    train_input, test_input = create_input_address(
        args.input_dir, args.model_name, args.train_file, args.test_file
    )
    
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Model ID: {extractor.model_id}")
    logger.info(f"Train input: {train_input}")
    logger.info(f"Test input: {test_input}")
    logger.info(f"PEI radius: {args.pei_radius}")
    logger.info(f"PEI steps: {args.pei_steps}")
    logger.info(f"Debug mode: {args.debug} (extractor.debug: {extractor.debug})")
    
    # Process files based on arguments
    if not args.test_only:
        if train_input.exists():
            logger.info(f"Processing training data from: {train_input}")
            extractor.process_dataset(
                train_input, origpert_dir, "train",
                pei_radius=args.pei_radius,
                pei_steps=args.pei_steps,
                limit=args.limit
            )
        else:
            logger.warning(f"Training file not found: {train_input}")
    
    if not args.train_only:
        if test_input.exists():
            logger.info(f"Processing test data from: {test_input}")
            extractor.process_dataset(
                test_input, origpert_dir, "test",
                pei_radius=args.pei_radius,
                pei_steps=args.pei_steps,
                limit=args.limit
            )
        else:
            logger.warning(f"Test file not found: {test_input}")
    
    logger.info("OrigPert hidden state extraction complete!")
    
    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Model name: {args.model_name}")
    logger.info(f"  Model ID: {extractor.model_id}")
    logger.info(f"  CUDA devices: {args.cuda_devices}")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Train input: {train_input}")
    logger.info(f"  Test input: {test_input}")
    logger.info(f"  Representations directory: {origpert_dir}")
    logger.info(f"  PEI radius: {args.pei_radius}")
    logger.info(f"  PEI steps: {args.pei_steps}")


if __name__ == "__main__":
    main()

# Qwen/Qwen2.5-0.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# Qwen/Qwen2.5-3B-Instruct

# Example Usage:
# python origpert_hidden_state_logit_extraction.py \
#     --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
#     --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
#     --cuda-devices "0" \
#     --input-dir ../data/labeled_v2 \
#     --representations-dir ../representations/OrigPert \
#     --train-file ehrnoteqa_train_mcqa_lbl.csv \
#     --test-file CORTEX_contextual_labeled.jsonl \
#     --pei-radius 20.0 \
#     --pei-steps 5 \
#     --dtype float16
#     --debug