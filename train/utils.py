import pandas as pd
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import librosa
import yaml
import jiwer
import re

def clean_text(s):
    # Remove tags like <ST> or <anything>
    s = re.sub(r'<[^>]+>', '', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s)
    # Remove leading and trailing spaces
    s = s.strip()
    return s

def load_config(CONFIG_FILE):
    """Loads configuration parameters from the YAML file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        # Exit or raise error, depending on desired robustness
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

def get_training_components(model_name):

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, language="english", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
    
    return feature_extractor, tokenizer, processor

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
         # Extract the valid token IDs from the tokenizer's vocab
        valid_token_ids = set(self.processor.tokenizer.get_vocab().values())
        
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Filter out invalid token IDs in label sequences
        for label in label_features:
            label["input_ids"] = [token for token in label["input_ids"] if token in valid_token_ids]

        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def filter_labels(labels):
    return len(labels) < 512