#!/usr/bin/env python3
"""
Script to extract final hidden states right after seeing the full input question query 
and before answering for PIK (Reasoning During Understanding) method.
This extracts the model's internal representation of the question understanding.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob
import re
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils.data_io import read_table  # noqa: E402

# Add utils directory to path
sys.path.append(str(Path(__file__).parent / "../utils"))
from general import set_visible_cudas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PIKHiddenStateExtractor:
    def __init__(self, model_id: str, cuda_devices: str, magistral_system_prompt_path: str = "../utils/MAGISTRAL_SYSTEM_PROMPT_SHORT.txt"):
        """
        Initialize the PIK hidden state extractor.
        
        Args:
            model_id: Model ID to use for hidden state extraction
            cuda_devices: CUDA devices to use (e.g., "0,1,2,3")
            magistral_system_prompt_path: Path to the Magistral system prompt file
        """
        self.model_id = model_id
        self.magistral_system_prompt_path = magistral_system_prompt_path
        
        # Set CUDA devices before loading model
        set_visible_cudas(cuda_devices)
        
        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"CUDA devices: {cuda_devices}")
                
        # Load model with auto device mapping
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # Load tokenizer
        if model_id == "mistralai/Magistral-Small-2506":
            self.tokenizer = AutoTokenizer.from_pretrained("unsloth/magistral-small-2506-unsloth-bnb-4bit", trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
    
    def _construct_question_only_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Construct messages containing only the user question/prompt.
        This represents the state right after understanding the question
        but before generating any response.
        
        Args:
            prompt: Original user prompt/question
            system_prompt: Optional system prompt from CSV (if available)
            
        Returns:
            List of messages containing only the user input
        """
        messages = []
        
        # Add system prompt if provided (from CSV or model-specific)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif self.model_id == "mistralai/Magistral-Small-2506":
            # Fallback to model-specific system prompt for Magistral if not in CSV
            prompt_path = self.magistral_system_prompt_path
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    magistral_system_prompt = f.read()
                messages.append({"role": "system", "content": magistral_system_prompt})
            except FileNotFoundError:
                logger.warning(f"Magistral system prompt file not found: {prompt_path}")
                exit()
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _tokenize_messages_with_generation_prompt(self, messages: List[Dict[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize messages with generation prompt to get the exact state
        right before the model starts generating.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Apply chat template with generation prompt to get the exact state
        # right before generation starts
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True  # This adds the assistant prompt
        )
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        return encoding.input_ids, encoding.attention_mask
        
    def extract_hidden_state_for_record(self, record: Dict[str, Any]) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Extract the final hidden state right after processing the question
        and before generating the answer, along with metadata.
        
        Args:
            record: Record with 'prompt', 'system_prompt' (optional), and other metadata
            
        Returns:
            Tuple of (hidden_state, metadata) or None if extraction fails
        """
        try:
            # Get the original prompt and metadata
            prompt = record['prompt']
            system_prompt = record.get('system_prompt', None)
            record_id = record.get('record_id', 'unknown')
            
            # Extract metadata
            metadata = {
                'record_id': record_id,
                'dataset': record.get('dataset', 'unknown'),
                'correctness': record.get('correctness', None),
                'sidx': record.get('sidx', None)  # Store sidx if available
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing record: {record_id}")
            logger.info(f"Dataset: {metadata['dataset']}")
            logger.info(f"Correctness: {metadata['correctness']}")
            logger.info(f"Prompt preview: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            # Construct messages containing only the user question
            messages = self._construct_question_only_messages(prompt, system_prompt)
            
            # Tokenize with generation prompt to get the exact state before generation
            input_ids, attention_mask = self._tokenize_messages_with_generation_prompt(messages)
            
            logger.info(f"Input sequence length: {input_ids.shape[1]} tokens")
            
            # Move to device and convert to float16
            input_ids = input_ids.to(next(self.model.parameters()).device)
            attention_mask = attention_mask.to(next(self.model.parameters()).device)
            attention_mask = attention_mask.to(torch.float16)  # Convert attention mask to float16
            
            # Get hidden states
            try:
                with torch.no_grad():
                    # Ensure computation is done in float16
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
            except torch.cuda.OutOfMemoryError as oom:
                logger.error(f"CUDA OOM during record {record_id}: {oom}")
                return None
            except RuntimeError as e:
                # some versions throw a RuntimeError instead of OutOfMemoryError
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM (RuntimeError) during record {record_id}: {e}")
                    return None
                else:
                    raise
        
            # Extract final layer hidden state for the last token
            # This represents the model's understanding of the question
            # right before it starts generating the answer
            final_hidden_states = outputs.hidden_states[-1].to(torch.float16)  # Ensure float16
            last_token_hidden_state = final_hidden_states[0, -1, :].to(torch.float16)  # Ensure float16
            
            # Convert to float16 numpy array
            hidden_state_np = last_token_hidden_state.cpu().numpy().astype(np.float16)
            
            # Decode the last few tokens for debugging
            last_few_tokens = input_ids[0, -5:].cpu().numpy()
            decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in last_few_tokens]
            
            # Get the actual token we extracted the hidden state from
            last_token_id = input_ids[0, -1].cpu().item()
            extracted_token = self.tokenizer.decode([last_token_id])
            
            logger.info(f"Last 5 tokens: {decoded_tokens}")
            logger.info(f"EXTRACTED HIDDEN STATE FROM TOKEN: '{extracted_token}' (token_id: {last_token_id})")
            logger.info(f"Extracted hidden state shape: {hidden_state_np.shape}")
            logger.info(f"Hidden state extracted from token position {input_ids.shape[1] - 1}")
            logger.info(f"This represents the model's question understanding before generating response")
            
            # Clear GPU memory
            del outputs, final_hidden_states, last_token_hidden_state
            torch.cuda.empty_cache()
            
            return hidden_state_np, metadata
            
        except Exception as e:
            logger.error(f"Error extracting hidden state for record {record.get('record_id', 'unknown')}: {e}")
            return None
    
    def save_hidden_state(self, hidden_state: np.ndarray, metadata: Dict[str, Any], output_file: Path) -> None:
        """Save hidden state and metadata to .npz file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving hidden_state shape: {hidden_state.shape}")
        logger.info(f"Saving metadata: {metadata}")
        
        # Prepare save dictionary
        save_dict = {
            'hidden_state': hidden_state,
            'record_id': metadata['record_id'],  # Row number
            'dataset': metadata['dataset'],
            'correctness': metadata['correctness']
        }
        
        # Add sidx if available (for reference)
        if 'sidx' in metadata and metadata['sidx'] is not None:
            save_dict['sidx'] = metadata['sidx']
        
        # Save both hidden state and metadata
        np.savez_compressed(output_file, **save_dict)
        logger.info(f"Saved hidden state and metadata to: {output_file}")
    
    def process_dataset(self, csv_path: Path, representations_dir: Path, 
                       split_name: str, limit: Optional[int] = None) -> None:
        """Process an entire CSV file for hidden state extraction."""
        logger.info(f"Processing {csv_path} for PIK hidden state extraction")
        
        if not csv_path.exists():
            logger.error(f"Input file does not exist: {csv_path}")
            return
        
        # Determine dataset name based on split_name (matching PTRUE.py pattern)
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
        
        # Check for sidx column (will be stored but not used as record_id to avoid duplicates)
        has_sidx = 'sidx' in df.columns
        if not has_sidx:
            logger.info(f"No 'sidx' column found in {csv_path}. Using row number as record_id.")
        
        # Get system_prompt if available
        has_system_prompt = 'system_prompt' in df.columns
        
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
                    
                    logger.info(f"\nProcessing record {record_idx}/{total_records_to_process}: {record_id}")
                    
                    # Extract prompt from llm_input
                    prompt = str(row['llm_input'])
                    
                    # Get correctness (binary 0/1)
                    correctness = int(row['correctness'])
                    
                    # Get system_prompt if available
                    system_prompt = str(row['system_prompt']) if has_system_prompt else None
                    if system_prompt and system_prompt.strip() == '':
                        system_prompt = None
                    
                    # Build record dictionary
                    record = {
                        'prompt': prompt,
                        'system_prompt': system_prompt,
                        'record_id': record_id,  # Row number for uniqueness
                        'dataset': dataset_name,  # EHRNoteQA or CORAL-MCQA
                        'correctness': correctness,
                        'sidx': str(row['sidx']) if has_sidx else None  # Store sidx for reference but don't use as ID
                    }
                    
                    # Extract hidden state
                    hidden_state, metadata = self.extract_hidden_state_for_record(record)
                    
                    if hidden_state is not None:
                        # Save hidden state
                        self.save_hidden_state(hidden_state, metadata, output_file)
                        logger.info(f"Successfully processed record: {record_id}")
                    else:
                        logger.warning(f"Failed to extract hidden state for record: {record_id}")
                    
                    pbar.update(1)
                    
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
    pik_dir = Path(representations_dir) / model_dir_name
    pik_dir.mkdir(parents=True, exist_ok=True)
    return pik_dir

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract hidden states right after question understanding for PIK method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model ID to use for hidden state extraction")
    parser.add_argument("--cuda-devices", type=str, default="0,1,2,3",
                       help="CUDA devices to use (e.g., '0,1,2,3')")
    
    # Data configuration
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing the input CSV files (labeled_v2/{model_name}/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (directory name in input_dir, e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    parser.add_argument("--representations-dir", type=str, default="../representations/PIK",
                       help="Base output directory for hidden state representations")
    
    # Input file patterns
    parser.add_argument("--train-file", type=str, default="ehrnoteqa_train_mcqa_lbl.csv",
                       help="Training CSV file name (default: ehrnoteqa_train_mcqa_lbl.csv)")
    parser.add_argument("--test-file", type=str, default="CORTEX_contextual_labeled.jsonl",
                       help="Test CSV file name (default: CORTEX_contextual_labeled.jsonl)")
    
    # Processing configuration
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of records to process per file (for testing)")
    
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
    
    # Initialize the hidden state extractor
    extractor = PIKHiddenStateExtractor(
        model_id=args.model_id,
        cuda_devices=args.cuda_devices,
        magistral_system_prompt_path=args.magistral_system_prompt_path
    )
    
    # Create output directory structure
    pik_dir = create_output_directory(args.representations_dir, extractor.model_id)
    
    logger.info(f"PIK directory: {pik_dir}")
    
    # Define input files
    train_input, test_input = create_input_address(
        args.input_dir, args.model_name, args.train_file, args.test_file
    )
    
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Model ID: {extractor.model_id}")
    logger.info(f"Train input: {train_input}")
    logger.info(f"Test input: {test_input}")
    
    # Process files based on arguments
    if not args.test_only:
        if train_input.exists():
            logger.info(f"Processing training data from: {train_input}")
            extractor.process_dataset(
                train_input, pik_dir, "train",
                limit=args.limit
            )
        else:
            logger.warning(f"Training file not found: {train_input}")
    
    if not args.train_only:
        if test_input.exists():
            logger.info(f"Processing test data from: {test_input}")
            extractor.process_dataset(
                test_input, pik_dir, "test",
                limit=args.limit
            )
        else:
            logger.warning(f"Test file not found: {test_input}")
    
    logger.info("PIK hidden state extraction complete!")
    
    # Log configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  Model name: {args.model_name}")
    logger.info(f"  Model ID: {extractor.model_id}")
    logger.info(f"  CUDA devices: {args.cuda_devices}")
    logger.info(f"  Input directory: {args.input_dir}")
    logger.info(f"  Train input: {train_input}")
    logger.info(f"  Test input: {test_input}")
    logger.info(f"  Representations directory: {pik_dir}")


if __name__ == "__main__":
    main()

# Qwen/Qwen2.5-0.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# Qwen/Qwen2.5-3B-Instruct

# Example Usage:
# python PIK_hidden_states.py \
#     --model-id "Qwen/Qwen2.5-0.5B-Instruct" \
#     --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
#     --cuda-devices "0" \
#     --input-dir ../data/labeled_v2 \
#     --representations-dir ../representations/PIK \
#     --train-file ehrnoteqa_train_mcqa_lbl.csv \
#     --test-file CORTEX_contextual_labeled.jsonl