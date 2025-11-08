import pandas as pd
import numpy as np
#import evaluate
import jiwer
import re
import os
import yaml
#import evaluate
from transformers import WhisperFeatureExtractor, WhisperTokenizer, pipeline, WhisperProcessor
import adapters

#metric = evaluate.load("wer")

'''adapter_dict = {
    "seqBN": adapters.SeqBnConfig,
    "parBN": adapters.ParBnConfig,
    "whisper_lora": adapters.LoRAConfig
}'''

def get_testing_configuration(config_path):
    # getting the training configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def clean_text(s):
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s)
    # Remove leading and trailing spaces
    s = s.strip()
    return s.lower()

def compute_wer(pred, tru):
    wer = jiwer.wer(hypothesis=[clean_text(pred)], truth=[clean_text(tru)])
    return wer

def get_testing_components(config_dict):

    feature_extractor = WhisperFeatureExtractor.from_pretrained(config_dict["name"], language=config_dict["language"], task=config_dict["task"], chunk_length=30)
    tokenizer = WhisperTokenizer.from_pretrained(config_dict["name"], language=config_dict["language"], task=config_dict["task"])
    processor = WhisperProcessor.from_pretrained(config_dict["name"], language=config_dict["language"], task=config_dict["task"], truncation=True)
    
    return feature_extractor, tokenizer, processor