#!/usr/bin/env python3
"""
P(True) evaluation script for MCQA datasets.
Reads CSV files with llm_input, queries models for answers and P(True) confidence,
extracts confidence probabilities from model logits, and evaluates against correctness labels.
"""

import argparse
import json
import sys
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils.data_io import read_table  # noqa: E402
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from eval import evaluate_by_groups, save_evaluation_results
from general import get_uncertainty_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PTrueProcessor:
    def __init__(self, model_id: str, use_local_model: bool = True, debug: bool = False):
        """Initialize the P(True) processor."""
        self.model_id = model_id
        self.use_local_model = use_local_model
        self.debug = debug
        
        # Initialize local model and tokenizer if needed
        if use_local_model and model_id:
            logger.info(f"Loading local model: {model_id}")
            try:
                if model_id == "mistralai/Magistral-Small-2506":
                    self.tokenizer = AutoTokenizer.from_pretrained("unsloth/magistral-small-2506-unsloth-bnb-4bit", trust_remote_code=True)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.model.eval()
                logger.info(f"Model loaded successfully on device map auto")
                
                # Get token IDs for choice options
                self.token_a_id = self.tokenizer.encode('a', add_special_tokens=False)[0]
                self.token_b_id = self.tokenizer.encode('b', add_special_tokens=False)[0]
                self.token_a_space_id = self.tokenizer.encode(' a', add_special_tokens=False)[0]
                self.token_b_space_id = self.tokenizer.encode(' b', add_special_tokens=False)[0]
                
                logger.info(f"Token IDs - 'a': {self.token_a_id}, 'b': {self.token_b_id}")
                logger.info(f"Token IDs with space - ' a': {self.token_a_space_id}, ' b': {self.token_b_space_id}")
                
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None
        
        self.system_prompt = None
        if model_id == "mistralai/Magistral-Small-2506":
            prompt_path = Path(__file__).parent.parent / "utils" / "MAGISTRAL_SYSTEM_PROMPT.txt"
            try:
                self.system_prompt = prompt_path.read_text(encoding="utf-8")
                logger.info("Loaded Magistral system prompt.")
            except FileNotFoundError:
                logger.warning(f"Could not find Magistral system prompt at {prompt_path}")
        
        # Track failed extractions
        self.failed_extractions = []
        self.failed_confidence_extraction_ids = []
    
    def extract_after_thinking(self, text: str) -> str:
        """
        Extract text after </think> or </thought> tags.
        """
        if not text:
            return ""
        
        if self.debug:
            print(f"\n🧠 EXTRACTING POST-THINKING TEXT:")
            print(f"   📝 Input text length: {len(text)}")
        
        # Split by thinking tags and take the last part
        for delimiter in ['</think>', '</thought>']:
            if delimiter in text:
                if self.debug:
                    print(f"   ✓ Found delimiter: {delimiter}")
                parts = text.split(delimiter)
                if self.debug:
                    print(f"   📊 Split into {len(parts)} parts")
                result = parts[-1].strip()
                if self.debug:
                    print(f"   📤 Extracted part: {repr(result)}")
                return result
        
        # If no thinking tags found, return the original text
        if self.debug:
            print(f"   ⚠️ No thinking tags found, returning original text")
        return text.strip()
    
    def find_choice_in_ptrue_response(self, ptrue_response: str) -> Optional[str]:
        """
        Find which choice (a or b) the model made in the ptrue response.
        Returns 'a', 'b', or None.
        """
        if not ptrue_response:
            return None
        
        # Extract post-thinking text first
        post_thinking = self.extract_after_thinking(ptrue_response)
        
        if self.debug:
            print(f"🔍 FINDING CHOICE IN RESPONSE:")
            print(f"   📝 Post-thinking text: {repr(post_thinking)}")
        
        # Look for (a) and (b) patterns - case insensitive
        pattern_a = r'\(a\)'
        pattern_b = r'\(b\)'
        
        matches_a = list(re.finditer(pattern_a, post_thinking, re.IGNORECASE))
        matches_b = list(re.finditer(pattern_b, post_thinking, re.IGNORECASE))
        
        if self.debug:
            print(f"   🔎 Found (a) patterns: {len(matches_a)}")
            print(f"   🔎 Found (b) patterns: {len(matches_b)}")
        
        # Take the last occurrence if any exist
        if matches_a and not matches_b:
            if self.debug:
                print(f"   ✓ Choice detected: 'a'")
            return 'a'
        elif matches_b and not matches_a:
            if self.debug:
                print(f"   ✓ Choice detected: 'b'")
            return 'b'
        elif matches_a and matches_b:
            # Take the last one that appears
            last_a_pos = matches_a[-1].start()
            last_b_pos = matches_b[-1].start()
            if last_a_pos > last_b_pos:
                if self.debug:
                    print(f"   ✓ Both found, last is 'a' at position {last_a_pos}")
                return 'a'
            else:
                if self.debug:
                    print(f"   ✓ Both found, last is 'b' at position {last_b_pos}")
                return 'b'
        else:
            if self.debug:
                print(f"   ❌ No parenthesized choice found, checking for single character...")
            # Try single character fallback
            single_char_choice = self.find_single_character_choice(post_thinking)
            if single_char_choice:
                return single_char_choice
            
            if self.debug:
                print(f"   ❌ No clear choice found")
            return None
    
    def find_single_character_choice(self, post_thinking_text: str) -> Optional[str]:
        """
        Check if the post-thinking response is just a single character 'a' or 'b'.
        """
        if self.debug:
            print(f"\n🔤 CHECKING FOR SINGLE CHARACTER CHOICE:")
            print(f"   📝 Post-thinking text: {repr(post_thinking_text)}")
        
        stripped_text = post_thinking_text.strip()
        if self.debug:
            print(f"   📏 Stripped text length: {len(stripped_text)}")
            print(f"   📄 Stripped text: {repr(stripped_text)}")
        
        # Check if it's exactly one character and is 'a' or 'b' (case insensitive)
        if len(stripped_text) == 1 and stripped_text.lower() in ['a', 'b']:
            choice = stripped_text.lower()
            if self.debug:
                print(f"   ✅ Found single character choice: '{choice}'")
            return choice
        
        if self.debug:
            print(f"   ❌ Not a single character choice")
        return None
    
    def build_conversation_for_logit_extraction(self, llm_input: str, system_prompt: str, 
                                                model_response: str, ptrue_response: str) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[int], Optional[List[int]], Optional[int]]:
        """
        Build the exact conversation that was fed to the model during generation.
        Returns (input_ids, chosen_option, absolute_logit_position, ptrue_token_ids, target_token_position)
        """
        if not self.tokenizer:
            return None, None, None, None, None

        if self.debug:
            print(f"\n" + "="*100)
            print(f"🔧 CONVERSATION RECONSTRUCTION FOR LOGIT EXTRACTION")
            print(f"="*100)

        # Get uncertainty query
        uncertainty_query = get_uncertainty_query()

        if self.debug:
            print(f"📋 CONVERSATION COMPONENTS:")
            print(f"   📝 Original llm_input length: {len(llm_input)}")
            print(f"   🤖 Model response length: {len(model_response)}")
            print(f"   ❓ Uncertainty query length: {len(uncertainty_query)}")
            print(f"   🎯 P(True) response length: {len(ptrue_response)}")

        # Find the choice in the ptrue_response
        chosen_option = self.find_choice_in_ptrue_response(ptrue_response)
        if self.debug:
            if chosen_option:
                print(f"   ✅ Detected choice: '{chosen_option}'")
            else:
                print(f"   ⚠️ Could not detect choice in response, will use first token for logit extraction")
        
        # If we can't find the choice, we'll still try to extract logits from the first token
        # This is more robust than failing completely
        if chosen_option is None:
            if self.debug:
                print(f"   ℹ️ No choice detected, but continuing with first-token logit extraction")
            # We'll use 'a' as a placeholder - the actual logit extraction will work regardless
            chosen_option = 'a'

        # Build the conversation up to the uncertainty query
        conversation = [
            {"role": "user", "content": llm_input},
            {"role": "assistant", "content": model_response},
            {"role": "user", "content": uncertainty_query}
        ]
        if system_prompt:
            conversation.insert(0, {"role": "system", "content": system_prompt})

        if self.debug:
            print(f"\n📞 BUILDING CONVERSATION STRUCTURE:")
            for i, msg in enumerate(conversation):
                display_text = self.truncate_text_display(msg["content"], 100)
                print(f"   {i+1}. {msg['role']}: {len(msg['content'])} chars - {display_text}")

        # Apply chat template to get the input up to generation prompt
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )

        if self.debug:
            print(f"   🔢 Chat template input_ids shape: {input_ids.shape}")
            print(f"   📄 Last 10 tokens of chat template:")
            for i, token_id in enumerate(input_ids[0][-10:].tolist()):
                token_str = self.tokenizer.decode([token_id])
                print(f"      [{len(input_ids[0])-10+i:3d}] {token_id:6d} -> {repr(token_str)}")

        # Now tokenize the stored P(True) response
        ptrue_token_ids = self.tokenizer.encode(ptrue_response, add_special_tokens=False)
        if self.debug:
            print(f"\n🎯 P(TRUE) RESPONSE TOKENIZATION:")
            print(f"   🔢 P(True) response tokens: {len(ptrue_token_ids)}")
            print(f"   📄 First 10 P(True) tokens:")
            for i, token_id in enumerate(ptrue_token_ids[:10]):
                token_str = self.tokenizer.decode([token_id])
                print(f"      [{i:3d}] {token_id:6d} -> {repr(token_str)}")

        # Find the position in P(True) sequence where we should extract logits
        ptrue_logit_position = self.find_choice_token_position(ptrue_token_ids, chosen_option, ptrue_response)
        if ptrue_logit_position is None:
            if self.debug:
                print(f"   ❌ Could not find choice token position in P(True) response")
            return None, None, None, None, None

        if self.debug:
            print(f"   ✅ Found logit extraction position {ptrue_logit_position} in P(True) response")
        
        # CRITICAL: The absolute position where we extract logits in the FULL sequence
        absolute_logit_position = len(input_ids[0]) + ptrue_logit_position

        if self.debug:
            print(f"\n🎯 LOGIT EXTRACTION SETUP:")
            print(f"   📍 P(True) logit position: {ptrue_logit_position}")
            print(f"   📍 Chat template length: {len(input_ids[0])}")
            print(f"   📍 Absolute logit position: {absolute_logit_position}")

        return input_ids, chosen_option, absolute_logit_position, ptrue_token_ids, ptrue_logit_position

    def truncate_text_display(self, text: str, max_length: int = 100) -> str:
        """
        Truncate text for display, showing first and last parts if too long.
        """
        if len(text) <= max_length:
            return repr(text)
        
        half = max_length // 2 - 3
        return f"{repr(text[:half])}...{repr(text[-half:])}"

    def find_choice_token_position(self, token_ids: List[int], chosen_option: str, ptrue_response: str) -> Optional[int]:
        """
        Find the position where we should extract logits to predict the choice token.
        Returns the position in the P(True) token sequence where logits should predict the choice.
        """
        if self.debug:
            print(f"\n🔍 FINDING CHOICE TOKEN POSITION:")
            print(f"   🎯 Looking for choice: '{chosen_option}'")
            print(f"   📝 Total tokens to search: {len(token_ids)}")
        
        # Extract post-thinking text to check if it's a single character response
        post_thinking = self.extract_after_thinking(ptrue_response)
        is_single_char = (len(post_thinking.strip()) == 1 and 
                        post_thinking.strip().lower() in ['a', 'b'])
        
        if self.debug:
            print(f"   📊 Is single character response: {is_single_char}")

        # Find the choice pattern in the tokenized response
        if is_single_char:
            target_pattern = chosen_option
        else:
            target_pattern = f"({chosen_option})"
        
        target_tokens = self.tokenizer.encode(target_pattern, add_special_tokens=False)
        
        if self.debug:
            print(f"\n🔤 ANALYZING TOKENIZATION OF '{target_pattern}':")
            print(f"   🔤 '{target_pattern}' -> {len(target_tokens)} tokens: {target_tokens}")
            for i, token_id in enumerate(target_tokens):
                token_str = self.tokenizer.decode([token_id])
                print(f"      [{i}] {token_id} -> {repr(token_str)}")
        
        # Find this exact pattern in our token sequence
        pattern_length = len(target_tokens)
        found_positions = []
        
        for i in range(len(token_ids) - pattern_length + 1):
            if token_ids[i:i+pattern_length] == target_tokens:
                found_positions.append(i)
                if self.debug:
                    print(f"   ✅ Found pattern '{target_pattern}' at positions {i}-{i+pattern_length-1}")
        
        if not found_positions:
            if self.debug:
                print(f"   ❌ Target pattern '{target_pattern}' not found in token sequence!")
            return self._fallback_choice_search(token_ids, chosen_option)
        
        # Take the last occurrence (most recent choice)
        target_start_pos = found_positions[-1]
        if self.debug:
            print(f"   🎯 Using FINAL occurrence at positions {target_start_pos}-{target_start_pos + pattern_length - 1}")
        
        # Determine logit extraction position based on tokenization pattern
        if pattern_length == 2:  # "(a)" -> ["(a", ")"]
            logit_extraction_pos = target_start_pos - 1 if target_start_pos > 0 else 0
        elif pattern_length == 3:  # "(a)" -> ["(", "a", ")"]
            logit_extraction_pos = target_start_pos
        else:
            logit_extraction_pos = target_start_pos - 1 if target_start_pos > 0 else 0
        
        if self.debug:
            print(f"   🎯 LOGIT EXTRACTION POSITION: {logit_extraction_pos}")
        
        if logit_extraction_pos < 0 or logit_extraction_pos >= len(token_ids):
            if self.debug:
                print(f"   ❌ Invalid logit position: {logit_extraction_pos}")
            return None
        
        return logit_extraction_pos

    def _fallback_choice_search(self, token_ids: List[int], chosen_option: str) -> Optional[int]:
        """
        Fallback method: search for any tokens containing the choice letter.
        """
        if self.debug:
            print(f"\n🔄 FALLBACK: Searching for choice letter '{chosen_option}'")
        
        # Get all possible token IDs for the choice
        choice_variants = [
            chosen_option,
            f' {chosen_option}',
            chosen_option.upper(),
            f' {chosen_option.upper()}',
            f'({chosen_option}',
            f'({chosen_option.upper()}',
            f'{chosen_option})',
            f'{chosen_option.upper()})',
        ]
        
        choice_token_ids = set()
        for variant in choice_variants:
            try:
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                choice_token_ids.update(tokens)
            except:
                pass
        
        if self.debug:
            print(f"   🔤 Choice token ID candidates: {sorted(choice_token_ids)}")
        
        # Find all positions where choice tokens appear
        choice_positions = []
        for i, token_id in enumerate(token_ids):
            if token_id in choice_token_ids:
                token_str = self.tokenizer.decode([token_id])
                choice_positions.append((i, token_id, token_str))
                if self.debug:
                    print(f"   📍 Found choice-related token at position {i}: {token_id} -> {repr(token_str)}")
        
        if not choice_positions:
            if self.debug:
                print(f"   ⚠️ No choice tokens found in fallback search!")
                print(f"   ℹ️ Using position 0 as last resort")
            # Last resort: use position 0 (first token)
            # This ensures we always return something rather than None
            return 0
        
        # Return the position that predicts the last choice token
        last_pos, last_token_id, last_token_str = choice_positions[-1]
        logit_pos = last_pos - 1 if last_pos > 0 else 0
        
        if self.debug:
            print(f"   ✅ Fallback: Using position {logit_pos} to predict token at {last_pos}")
        return logit_pos
    
    def extract_ptrue_from_prompt_only(self, llm_input: str, system_prompt: str, 
                                       model_response: str) -> Optional[Tuple[float, bool]]:
        """
        Extract P(True) directly from the prompt by getting logits for the next token
        after the uncertainty query. This is a robust fallback that doesn't depend on
        parsing the model's response.
        
        Returns tuple of (ptrue_probability, argmax_consistent) or None if extraction fails.
        """
        if not self.tokenizer or not self.model:
            return None
        
        if self.debug:
            print(f"\n" + "🔬" + "="*90)
            print(f"🔬 EXTRACTING P(TRUE) FROM PROMPT ONLY (ROBUST FALLBACK)")
            print(f"🔬" + "="*90)
        
        try:
            # Get uncertainty query
            uncertainty_query = get_uncertainty_query()
            
            # Build the conversation up to the uncertainty query
            conversation = [
                {"role": "user", "content": llm_input},
                {"role": "assistant", "content": model_response},
                {"role": "user", "content": uncertainty_query}
            ]
            if system_prompt:
                conversation.insert(0, {"role": "system", "content": system_prompt})
            
            # Apply chat template to get the input up to generation prompt
            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            )
            
            if self.debug:
                print(f"   🔢 Input sequence length: {input_ids.shape[1]}")
            
            # Get logits from the model for the next token
            input_ids = input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]  # Logits for the next token prediction
            
            if self.debug:
                print(f"   ✅ Successfully extracted logits shape: {logits.shape}")
            
            # Extract logits for 'a' and 'b' tokens
            logit_a = self.get_best_choice_logit(logits, 'a')
            logit_b = self.get_best_choice_logit(logits, 'b')
            
            if logit_a is None or logit_b is None:
                if self.debug:
                    print(f"   ⚠️ Could not extract logits for a/b choice tokens")
                    print(f"   ℹ️ Using fallback: assigning very low logits to missing tokens")
                # Fallback: if we can't find one of the tokens, use a very low logit value
                if logit_a is None:
                    logit_a = -100.0
                if logit_b is None:
                    logit_b = -100.0
            
            if self.debug:
                print(f"\n🎲 CHOICE TOKEN LOGITS:")
                print(f"   📊 Best 'a' choice logit: {logit_a:.4f}")
                print(f"   📊 Best 'b' choice logit: {logit_b:.4f}")
            
            # Convert to probabilities using softmax
            logits_tensor = torch.tensor([logit_a, logit_b], dtype=torch.float32)
            probs = F.softmax(logits_tensor, dim=0)
            
            # P(True) interpretation: (a) = "no" = False, (b) = "yes" = True
            # So P(True) = P(b)
            ptrue = float(probs[1])  # P(b) = P(True)
            
            if self.debug:
                print(f"\n🎯 FINAL RESULTS:")
                print(f"   📊 P(a) = P(False): {probs[0]:.4f}")
                print(f"   📊 P(b) = P(True):  {probs[1]:.4f}")
                print(f"   🎯 P(True): {ptrue:.4f}")
            
            # Since we're not using the actual response, we can't check consistency
            return ptrue, False
            
        except Exception as e:
            if self.debug:
                print(f"   ❌ Error during prompt-only logit extraction: {e}")
            logger.error(f"Error in extract_ptrue_from_prompt_only: {e}")
            return None
    
    def extract_ptrue_from_logits(self, llm_input: str, system_prompt: str, 
                                   model_response: str, ptrue_response: str) -> Optional[Tuple[float, bool]]:
        """
        Extract P(True) from model logits.
        Returns tuple of (ptrue_probability, argmax_consistent) or None if extraction fails.
        """
        if self.debug:
            print(f"\n" + "🔬" + "="*90)
            print(f"🔬 EXTRACTING P(TRUE) FROM LOGITS")
            print(f"🔬" + "="*90)

        # Build the conversation for logit extraction
        result = self.build_conversation_for_logit_extraction(llm_input, system_prompt, model_response, ptrue_response)
        if result[0] is None:
            if self.debug:
                print(f"❌ Failed to build conversation for logit extraction")
            return None
        
        input_ids, chosen_option, absolute_logit_position, ptrue_token_ids, ptrue_logit_position = result

        if self.debug:
            print(f"\n🧮 LOGIT COMPUTATION:")
            print(f"   📊 Will extract logits at absolute position {absolute_logit_position}")

        # CRITICAL: We need to include tokens up to AND INCLUDING the logit position
        tokens_needed = ptrue_logit_position + 1
        
        if self.debug:
            print(f"   🔧 CRITICAL CALCULATION:")
            print(f"      P(True) logit position: {ptrue_logit_position}")
            print(f"      Tokens needed from P(True) response: {tokens_needed}")
        
        if tokens_needed < 0:
            if self.debug:
                print(f"   ❌ Invalid token count: {tokens_needed}")
            return None
        
        if tokens_needed > len(ptrue_token_ids):
            if self.debug:
                print(f"   ❌ Not enough P(True) tokens: need {tokens_needed}, have {len(ptrue_token_ids)}")
            return None
        
        prefix_ptrue_tokens = torch.tensor(ptrue_token_ids[:tokens_needed], device=input_ids.device).unsqueeze(0)
        full_input_sequence = torch.cat([input_ids, prefix_ptrue_tokens], dim=1)
        attention_mask = torch.ones_like(full_input_sequence)

        if self.debug:
            print(f"   🔗 Combined sequence length: {full_input_sequence.shape[1]}")

        # Get logits from the model
        try:
            full_input_sequence = full_input_sequence.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=full_input_sequence, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]  # Logits for the next token prediction

            if self.debug:
                print(f"   ✅ Successfully extracted logits shape: {logits.shape}")

            # Get the actual predicted token and verify consistency
            predicted_token_id = torch.argmax(logits).item()
            predicted_token_str = self.tokenizer.decode([predicted_token_id])

            # Check if prediction matches what was actually generated
            argmax_consistent = False
            expected_next_token_id = ptrue_token_ids[tokens_needed] if tokens_needed < len(ptrue_token_ids) else None
            if expected_next_token_id:
                argmax_consistent = (predicted_token_id == expected_next_token_id)
                actual_token_str = self.tokenizer.decode([expected_next_token_id])
                if self.debug:
                    print(f"\n🎯 PREDICTION VERIFICATION:")
                    print(f"   🤖 Model predicted: {predicted_token_id} -> {repr(predicted_token_str)}")
                    print(f"   ✅ Actually generated: {expected_next_token_id} -> {repr(actual_token_str)}")
                    print(f"   🔍 Argmax consistent: {argmax_consistent}")

            # Extract logits for 'a' and 'b' tokens
            logit_a = self.get_best_choice_logit(logits, 'a')
            logit_b = self.get_best_choice_logit(logits, 'b')

            if logit_a is None or logit_b is None:
                if self.debug:
                    print(f"   ⚠️ Could not extract logits for a/b choice tokens")
                    print(f"   ℹ️ Using fallback: assigning very low logits to missing tokens")
                # Fallback: if we can't find one of the tokens, use a very low logit value
                # This ensures we always have a probability distribution
                if logit_a is None:
                    logit_a = -100.0  # Very low probability
                if logit_b is None:
                    logit_b = -100.0  # Very low probability

            if self.debug:
                print(f"\n🎲 CHOICE TOKEN LOGITS:")
                print(f"   📊 Best 'a' choice logit: {logit_a:.4f}")
                print(f"   📊 Best 'b' choice logit: {logit_b:.4f}")

            # Convert to probabilities using softmax
            logits_tensor = torch.tensor([logit_a, logit_b], dtype=torch.float32)
            probs = F.softmax(logits_tensor, dim=0)

            # P(True) interpretation: (a) = "no" = False, (b) = "yes" = True
            # So P(True) = P(b)
            ptrue = float(probs[1])  # P(b) = P(True)

            if self.debug:
                print(f"\n🎯 FINAL RESULTS:")
                print(f"   📊 P(a) = P(False): {probs[0]:.4f}")
                print(f"   📊 P(b) = P(True):  {probs[1]:.4f}")
                print(f"   🎯 P(True): {ptrue:.4f}")
                print(f"   ✅ Consistency: {argmax_consistent}")

            return ptrue, argmax_consistent

        except Exception as e:
            if self.debug:
                print(f"   ❌ Error during logit computation: {e}")
            return None
    
    def get_best_choice_logit(self, logits: torch.Tensor, choice: str) -> Optional[float]:
        """Get the highest logit value among all choice-related tokens for a given choice."""
        # Get all possible token IDs for this choice, including parenthesized versions
        choice_variants = [
            choice,
            f' {choice}',
            choice.upper(),
            f' {choice.upper()}',
            f'({choice}',
            f'({choice.upper()}',
            f'{choice})',
            f'{choice.upper()})',
            f'({choice})',
            f'({choice.upper()})',
        ]
        
        choice_token_ids = set()
        for variant in choice_variants:
            try:
                variant_tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                choice_token_ids.update(variant_tokens)
            except:
                pass
        
        choice_token_ids = list(choice_token_ids)
        
        if not choice_token_ids:
            if self.debug:
                print(f"      ❌ No token IDs found for choice '{choice}'")
            return None
        
        if self.debug:
            print(f"      🔍 Choice '{choice}' token candidates: {choice_token_ids}")
        
        # Filter to prioritize tokens that actually contain the choice letter
        filtered_choice_tokens = []
        for token_id in choice_token_ids:
            token_str = self.tokenizer.decode([token_id])
            if choice.lower() in token_str.lower():
                filtered_choice_tokens.append((token_id, token_str, 1))  # High priority
            else:
                filtered_choice_tokens.append((token_id, token_str, 0))  # Low priority
        
        # Get the maximum logit among high-priority variants first, then fall back to low-priority
        max_logit = float('-inf')
        best_token_id = None
        
        # First try high-priority tokens
        for token_id, token_str, priority in filtered_choice_tokens:
            if priority == 1 and token_id < logits.shape[0]:
                logit_val = logits[token_id].item()
                if self.debug:
                    print(f"         🔥 HIGH PRIORITY: Token {token_id} -> {repr(token_str)}: logit {logit_val:.4f}")
                if logit_val > max_logit:
                    max_logit = logit_val
                    best_token_id = token_id
        
        # If no high-priority tokens found, try low-priority
        if max_logit == float('-inf'):
            for token_id, token_str, priority in filtered_choice_tokens:
                if priority == 0 and token_id < logits.shape[0]:
                    logit_val = logits[token_id].item()
                    if self.debug:
                        print(f"         ⚡ LOW PRIORITY: Token {token_id} -> {repr(token_str)}: logit {logit_val:.4f}")
                    if logit_val > max_logit:
                        max_logit = logit_val
                        best_token_id = token_id
        
        if best_token_id is not None:
            token_str = self.tokenizer.decode([best_token_id])
            if self.debug:
                print(f"      ✅ Best '{choice}' choice token: {best_token_id} -> {repr(token_str)} (logit: {max_logit:.4f})")
        else:
            if self.debug:
                print(f"      ❌ No valid logit found for choice '{choice}'")
        
        return max_logit if max_logit != float('-inf') else None
    
    def extract_ptrue_numeric(self, llm_input: str, system_prompt: str, 
                              model_response: str, ptrue_response: str, 
                              record_index: int = None) -> Optional[Dict[str, Any]]:
        """Extract P(True) probability from responses using logits."""
        if self.debug:
            print(f"\n" + "🎯" + "="*90)
            print(f"🎯 PROCESSING RECORD {record_index}")
            print(f"🎯" + "="*90)
        
        if not ptrue_response.strip():
            logger.warning(f"No P(True) response found for record {record_index}")
            # Even if response is empty, try to extract logits from the prompt
            if self.use_local_model and self.model:
                logit_result = self.extract_ptrue_from_prompt_only(llm_input, system_prompt, model_response)
                if logit_result is not None:
                    ptrue, argmax_consistent = logit_result
                    return {
                        'confidence_score': ptrue,
                        'argmax_consistent': argmax_consistent,
                        'extraction_method': 'logits_prompt_only'
                    }
            # If all else fails, return uniform probability
            logger.warning(f"Using uniform probability (0.5) for record {record_index}")
            return {
                'confidence_score': 0.5,
                'argmax_consistent': False,
                'extraction_method': 'uniform_fallback'
            }
        
        result = {
            'confidence_score': None,
            'argmax_consistent': None,
            'extraction_method': None
        }
        
        # Method 1: Try to extract from logits (preferred method)
        if self.use_local_model and self.model:
            if self.debug:
                print(f"🔬 Attempting logit extraction...")
            logit_result = self.extract_ptrue_from_logits(llm_input, system_prompt, model_response, ptrue_response)
            if logit_result is not None:
                ptrue, argmax_consistent = logit_result
                result['confidence_score'] = ptrue
                result['argmax_consistent'] = argmax_consistent
                result['extraction_method'] = 'logits'
                if self.debug:
                    print(f"✅ Logit extraction successful: P(True)={ptrue:.4f}, consistent={argmax_consistent}")
                return result
            else:
                if self.debug:
                    print(f"⚠️ Logit extraction from response failed, trying prompt-only extraction...")
                # Fallback: Extract logits from just the prompt (ignore the response)
                logit_result = self.extract_ptrue_from_prompt_only(llm_input, system_prompt, model_response)
                if logit_result is not None:
                    ptrue, argmax_consistent = logit_result
                    result['confidence_score'] = ptrue
                    result['argmax_consistent'] = argmax_consistent
                    result['extraction_method'] = 'logits_prompt_only'
                    if self.debug:
                        print(f"✅ Prompt-only logit extraction successful: P(True)={ptrue:.4f}")
                    return result
        
        # Final fallback: Use uniform probability
        if self.debug:
            print(f"⚠️ All extraction methods failed, using uniform probability (0.5)")
        logger.warning(f"Using uniform probability (0.5) for record {record_index}")
        return {
            'confidence_score': 0.5,
            'argmax_consistent': False,
            'extraction_method': 'uniform_fallback'
        }
    
    def process_csv_file(self, csv_path: Path, model_response_dict: Dict[int, str], 
                        ptrue_response_dict: Dict[int, str], split_name: str) -> Tuple[List[Dict[str, Any]], int]:
        """Process a single CSV file and return structured records."""
        logger.info(f"Processing file: {csv_path}")
        
        if not csv_path.exists():
            logger.error(f"File does not exist: {csv_path}")
            return [], 0
        
        # Read CSV
        df = read_table(csv_path)
        
        # Validate required columns
        required_columns = ['llm_input', 'correctness']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return [], 0
        
        # Note: sidx column may exist but is not used as record_id (matching PIK/CCPS format)
        # We use row number (i) as record_id to ensure uniqueness and consistency with PIK/CCPS
        
        # Determine dataset name based on split_name
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        
        # Get system_prompt if available
        has_system_prompt = 'system_prompt' in df.columns
        
        total_records = len(df)
        processed_records = []
        
        # Statistics for logit extraction
        logit_success_count = 0
        argmax_consistent_count = 0

        for i, row in tqdm(df.iterrows(), total=total_records, desc="Processing records"):
            if self.debug:
                print(f"\n" + "🚀" + "="*90)
                print(f"🚀 PROCESSING RECORD {i+1}/{total_records}")
                print(f"🚀" + "="*90)
            
            # Get ground truth correctness
            ground_truth_label = int(row['correctness'])
            
            # Get llm_input and system_prompt
            llm_input = str(row['llm_input'])
            system_prompt = str(row['system_prompt']) if has_system_prompt else ""
            
            # Get model responses (should already be generated)
            model_response = model_response_dict.get(i, "")
            ptrue_response = ptrue_response_dict.get(i, "")
            
            if not model_response:
                logger.warning(f"Missing model response for record {i}, using empty string")
                model_response = ""
            
            if not ptrue_response:
                logger.warning(f"Missing P(True) response for record {i}, will use prompt-only extraction")
                ptrue_response = ""
            
            extraction_result = self.extract_ptrue_numeric(
                llm_input, system_prompt, model_response, ptrue_response, i
            )
            
            # Use row number as record_id to ensure uniqueness (matching PIK/CCPS format)
            # Note: sidx is not used as record_id to avoid duplicates in test set
            record_id = str(i)
            
            # Store record info (matching PIK/CCPS format - no sidx or original_index)
            processed_record = {
                'record_id': record_id,
                'dataset': dataset_name,
                'ground_truth_correctness': ground_truth_label,
                'confidence_score': None,
                'argmax_consistent': None,
                'extraction_method': None
            }
            
            if extraction_result:
                processed_record.update(extraction_result)
                
                # Update statistics
                if extraction_result['extraction_method'] == 'logits':
                    logit_success_count += 1
                    if extraction_result['argmax_consistent']:
                        argmax_consistent_count += 1
            
            processed_records.append(processed_record)

        # Count records with different extraction methods
        records_with_scores = [r for r in processed_records if r['confidence_score'] is not None]
        records_without_scores = [r for r in processed_records if r['confidence_score'] is None]
        
        if records_without_scores:
            logger.warning(f"  ⚠️ Found {len(records_without_scores)} records without confidence scores!")
            logger.warning(f"  ⚠️ This should not happen with the new robust extraction. Assigning uniform probability.")
            # Assign uniform probability to any remaining records without scores
            for record in records_without_scores:
                record['confidence_score'] = 0.5
                record['argmax_consistent'] = False
                record['extraction_method'] = 'emergency_fallback'
        
        # Now all records should have confidence scores
        final_records = processed_records
        
        logger.info(f"\n🎯 PROCESSING SUMMARY:")
        logger.info(f"  📊 Total input records: {total_records}")
        logger.info(f"  🎯 Records with P(True) scores: {len(final_records)}")
        logger.info(f"  🔬 Logit extractions successful: {logit_success_count}")
        logger.info(f"  ✅ Argmax consistent: {argmax_consistent_count}/{logit_success_count if logit_success_count > 0 else 1}")
        
        # Count extraction methods
        method_counts = {}
        for record in final_records:
            method = record.get('extraction_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logger.info(f"  📊 Extraction methods used:")
        for method, count in sorted(method_counts.items()):
            logger.info(f"    - {method}: {count} records ({count/len(final_records)*100:.1f}%)")

        return final_records, total_records
    
    def query_model_for_responses(self, csv_path: Path, llm, temperature: float = 0.0, 
                                  max_tokens_answer: int = 10, max_tokens_ptrue: int = 50,
                                  chat_template: str = "openai") -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Query the model twice for each row:
        1. First query: Get answer to MCQ (using llm_input)
        2. Second query: Get P(True) response (appending uncertainty query)
        
        Returns (model_response_dict, ptrue_response_dict)
        """
        logger.info(f"Querying model for responses from: {csv_path}")
        
        # Read CSV
        df = read_table(csv_path)
        
        # Validate required columns
        if 'llm_input' not in df.columns:
            logger.error(f"Missing 'llm_input' column")
            return {}, {}
        
        has_system_prompt = 'system_prompt' in df.columns
        uncertainty_query = get_uncertainty_query()
        
        # Prepare conversations for first query (MCQ answers)
        conversations_answer = []
        for idx, row in df.iterrows():
            llm_input = str(row['llm_input'])
            if has_system_prompt:
                system_prompt = str(row['system_prompt'])
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_input}
                ]
            else:
                conversation = [{"role": "user", "content": llm_input}]
            conversations_answer.append(conversation)
        
        # Query for answers
        logger.info("Querying model for MCQ answers...")
        model_responses = llm.batch_chat_query(
            conversations_answer,
            temperature=temperature,
            max_tokens=max_tokens_answer,
            use_tqdm=True,
            chat_template_content_format=chat_template
        )
        
        model_response_dict = {i: resp for i, resp in enumerate(model_responses)}
        
        # Prepare conversations for second query (P(True) responses)
        conversations_ptrue = []
        for idx, row in df.iterrows():
            llm_input = str(row['llm_input'])
            model_response = model_responses[idx]
            
            if has_system_prompt:
                system_prompt = str(row['system_prompt'])
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_input},
                    {"role": "assistant", "content": model_response},
                    {"role": "user", "content": uncertainty_query}
                ]
            else:
                conversation = [
                    {"role": "user", "content": llm_input},
                    {"role": "assistant", "content": model_response},
                    {"role": "user", "content": uncertainty_query}
                ]
            conversations_ptrue.append(conversation)
        
        # Query for P(True) responses
        logger.info("Querying model for P(True) responses...")
        ptrue_responses = llm.batch_chat_query(
            conversations_ptrue,
            temperature=temperature,
            max_tokens=max_tokens_ptrue,
            use_tqdm=True,
            chat_template_content_format=chat_template
        )
        
        ptrue_response_dict = {i: resp for i, resp in enumerate(ptrue_responses)}
        
        logger.info(f"Generated {len(model_responses)} model responses and {len(ptrue_responses)} P(True) responses")
        
        return model_response_dict, ptrue_response_dict
    
    def prepare_evaluation_records(self, processed_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare records for evaluation in the standard format (matching PIK format)."""
        evaluation_records = []
        
        for result in processed_records:
            # Extract ground truth
            ground_truth_label = result.get('ground_truth_correctness')
            if ground_truth_label is None:
                continue
            
            evaluation_record = {
                'record_id': result['record_id'],
                'dataset': result.get('dataset', 'unknown'),
                'category': 'all',  # Dummy category for MCQA (matching PIK/CCPS pattern)
                'ground_truth_correctness': int(ground_truth_label),
                'confidence_score': result.get('confidence_score'),
                'original_result': result  # Store full original result
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
                                   processed_records: List[Dict[str, Any]],
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
        categories = np.array([r['category'] for r in evaluation_records])
        
        # Calculate comprehensive evaluation results
        evaluation_results = evaluate_by_groups(ground_truth_labels, confidence_scores, datasets, categories)
        evaluation_results['overall']['n_total_samples'] = n_total_samples
        
        # Calculate method-specific statistics from processed_records
        method_stats = {}
        for method in ['logits', 'patterns', 'gpt']:
            method_records = [r for r in processed_records if r.get('extraction_method') == method]
            method_stats[method] = len(method_records)
        
        # Calculate argmax consistency for logit extractions
        logit_records = [r for r in processed_records if r.get('extraction_method') == 'logits']
        argmax_consistent_count = sum(1 for r in logit_records if r.get('argmax_consistent', False))
        
        # Determine dataset name based on split_name
        dataset_name = "EHRNoteQA" if split_name == "train" else "CORAL-MCQA"
        
        # Add metadata (matching PIK format)
        evaluation_results['metadata'] = {
            'model_id': model_id,
            'split_name': split_name,
            'dataset': dataset_name,
            'total_records': len(evaluation_records),
            'unique_datasets': list(set(r['dataset'] for r in evaluation_records)),
            'evaluation_timestamp': str(np.datetime64('now')),
            'ptrue_statistics': {
                'extraction_method_counts': method_stats,
                'logit_extractions_count': len(logit_records),
                'argmax_consistent_count': argmax_consistent_count,
                'argmax_consistency_rate': argmax_consistent_count / len(logit_records) if logit_records else 0,
                'local_model_used': self.use_local_model and self.model is not None
            }
        }
        
        # Save evaluation results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_results(evaluation_results, output_path)
        
        # Print summary
        self.print_results_summary(evaluation_results, split_name)
    
    def print_results_summary(self, results: Dict[str, Any], split_name: str) -> None:
        """Print a summary of evaluation results."""
        print(f"\n{split_name.upper()} EVALUATION SUMMARY:")
        print("=" * 50)
        
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
        
        # Extraction method statistics
        if 'metadata' in results:
            metadata = results['metadata']
            print(f"\nExtraction Statistics:")
            for method, count in metadata.get('extraction_method_counts', {}).items():
                print(f"  {method.capitalize()}: {count} records")
            
            if metadata.get('logit_extractions_count', 0) > 0:
                consistency_rate = metadata.get('argmax_consistency_rate', 0)
                print(f"  Argmax consistency: {metadata.get('argmax_consistent_count', 0)}/{metadata.get('logit_extractions_count', 0)} ({consistency_rate:.2%})")
        
        print("=" * 50)

    def process_split(self, csv_path: Path, output_dir: Path, split_name: str, 
                     model_id: str, llm, temperature: float = 0.0,
                     max_tokens_answer: int = 10, max_tokens_ptrue: int = 50,
                     chat_template: str = "openai") -> None:
        """Process a single split (train or test)."""
        if not csv_path.exists():
            logger.warning(f"{split_name} file not found: {csv_path}")
            return
        
        # Step 1: Query model for responses
        model_response_dict, ptrue_response_dict = self.query_model_for_responses(
            csv_path, llm, temperature, max_tokens_answer, max_tokens_ptrue, chat_template
        )
        
        if not model_response_dict or not ptrue_response_dict:
            logger.error(f"Failed to get model responses from {csv_path}")
            return
        
        # Step 2: Process the file and extract P(True) scores
        processed_records, n_total_samples = self.process_csv_file(
            csv_path, model_response_dict, ptrue_response_dict, split_name
        )
        
        if not processed_records:
            logger.error(f"No valid records processed from {csv_path}")
            return
        
        # Step 3: Prepare evaluation records (matching PIK format)
        evaluation_records = self.prepare_evaluation_records(processed_records)
        
        # Step 4: Save processed data
        labels_path = output_dir / f"{split_name}_labels.json"
        self.save_processed_data(evaluation_records, labels_path)

        # Step 5: Calculate and save evaluation results
        results_path = output_dir / f"{split_name}_results.json"
        self.calculate_and_save_results(evaluation_records, processed_records, results_path, split_name, model_id, n_total_samples)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process MCQA CSV files and calculate P(True) confidence scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input configuration
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing labeled CSV files (labeled_v2/{model_name}/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (directory name in labeled_v2/)")
    parser.add_argument("--model-id", type=str, required=True,
                       help="Model ID for loading local model (e.g., 'mistralai/Magistral-Small-2506')")
    
    # File configuration
    parser.add_argument("--train-file", type=str, default="ehrnoteqa_train_mcqa_lbl.csv",
                       help="Training CSV file name (default: ehrnoteqa_train_mcqa_lbl.csv)")
    parser.add_argument("--test-file", type=str, default="CORTEX_contextual_labeled.jsonl",
                       help="Test CSV file name (default: CORTEX_contextual_labeled.jsonl)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save processed data and results")
    
    # Processing options
    parser.add_argument("--train-only", action="store_true",
                       help="Process only training data")
    parser.add_argument("--test-only", action="store_true",
                       help="Process only test data")
    parser.add_argument("--disable-local-model", action="store_true",
                       help="Disable local model loading (will only work if responses are pre-generated)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode for verbose logging")
    
    # LLM query parameters
    parser.add_argument("--llm-id", type=str, required=True,
                       help="LLM ID for vLLM (e.g., 'meta-llama/Llama-3.1-8B-Instruct')")
    parser.add_argument("--llm-dir", type=str, default=None,
                       help="LLM directory path (optional, defaults to llm-id)")
    parser.add_argument("--visible-cudas", type=str, required=True,
                       help="Visible CUDA devices (e.g., '0,1')")
    parser.add_argument("--dtype", type=str, default=None,
                       help="Model precision (float16, bfloat16, float32)")
    parser.add_argument("--temp", type=float, default=0.0,
                       help="Temperature for LLM queries")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                       help="GPU memory utilization for vLLM")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    parser.add_argument("--max-tokens-answer", type=int, default=10,
                       help="Max tokens for MCQ answer generation")
    parser.add_argument("--max-tokens-ptrue", type=int, default=50,
                       help="Max tokens for P(True) response generation")
    parser.add_argument("--chat-template", type=str, default="openai",
                       help="Chat template format for vLLM")
    parser.add_argument("--enforce-eager", action="store_true",
                       help="Use eager mode instead of compilation (fixes torch.compile issues)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate arguments
    if args.train_only and args.test_only:
        raise ValueError("Cannot specify both --train-only and --test-only")
    
    # Set CUDA visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas
    
    # Import Talk2LLM
    sys.path.append(str(Path(__file__).parent.parent / "utils"))
    from talk2llm import Talk2LLM
    
    # Initialize LLM for querying
    logger.info("Initializing vLLM model for querying...")
    llm = Talk2LLM(
        model_id=args.llm_dir if args.llm_dir else args.llm_id,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
        enforce_eager=args.enforce_eager,
        tokenizer_mode=None,
        config_format=None,
        quantization=None,
        load_format=None,
    )
    logger.info("vLLM model initialized successfully")
    
    # Initialize processor
    use_local_model = not args.disable_local_model
    processor = PTrueProcessor(
        model_id=args.model_id if use_local_model else None,
        use_local_model=use_local_model,
        debug=args.debug
    )
    
    # Create input and output paths
    # Extract just the model name (last part after /) for directory paths
    model_name_for_path = args.model_name.split("/")[-1]
    input_dir = Path(args.input_dir) / model_name_for_path
    train_path = input_dir / args.train_file
    test_path = input_dir / args.test_file
    
    output_path = Path(args.output_dir) / model_name_for_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Model ID (for logits): {args.model_id}")
    logger.info(f"LLM ID (for querying): {args.llm_id}")
    logger.info(f"Train file: {train_path}")
    logger.info(f"Test file: {test_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Local model enabled: {use_local_model}")
    logger.info(f"Debug mode enabled: {args.debug}")
    
    # Process based on arguments
    if not args.test_only:
        processor.process_split(
            train_path, output_path, 'train', args.model_id, llm,
            temperature=args.temp,
            max_tokens_answer=args.max_tokens_answer,
            max_tokens_ptrue=args.max_tokens_ptrue,
            chat_template=args.chat_template
        )
    
    if not args.train_only:
        processor.process_split(
            test_path, output_path, 'test', args.model_id, llm,
            temperature=args.temp,
            max_tokens_answer=args.max_tokens_answer,
            max_tokens_ptrue=args.max_tokens_ptrue,
            chat_template=args.chat_template
        )
    
    print(f"\nProcessing complete! Results saved to: {output_path}")
    print("\nGenerated files:")
    print("- *_labels.json: Structured records with ground truth and P(True) scores")
    print("- *_results.json: Comprehensive evaluation metrics")


if __name__ == "__main__":
    main()

# Qwen/Qwen2.5-0.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# Qwen/Qwen2.5-3B-Instruct

# Example usage:
# python PTRUE.py \
#     --input-dir ../data/labeled_v2 \
#     --model-name meta-llama/Llama-3.2-3B-Instruct \
#     --model-id meta-llama/Llama-3.2-3B-Instruct \
#     --output-dir ../results/PTRUE \
#     --llm-id meta-llama/Llama-3.2-1B-Instruct \
#     --visible-cudas "2" \
#     --dtype bfloat16 \
#     --temp 0.0 \
#     --gpu-memory 0.9 \
#     --tensor-parallel 1 \
#     --max-tokens-answer 10 \
#     --max-tokens-ptrue 50 \
#     --chat-template llama \
#     --train-file ehrnoteqa_train_mcqa_lbl.csv \
#     --test-file CORTEX_contextual_labeled.jsonl \
#     --debug

# python PTRUE.py \
#     --input-dir ../data/labeled_l3 \
#     --model-name Qwen/Qwen2.5-3B-Instruct \
#     --model-id Qwen/Qwen2.5-3B-Instruct \
#     --output-dir ../results/PTRUE_l3 \
#     --llm-id Qwen/Qwen2.5-3B-Instruct \
#     --visible-cudas "4" \
#     --dtype bfloat16 \
#     --temp 0.0 \
#     --gpu-memory 0.9 \
#     --tensor-parallel 1 \
#     --max-tokens-answer 10 \
#     --max-tokens-ptrue 50 \
#     --chat-template qwen \
#     --test-file CORTEX_clinical_inference_labeled.jsonl \
#     --test-only