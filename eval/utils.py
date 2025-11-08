import pandas as pd
import numpy as np
import os
import librosa
import jiwer
import re
import yaml
from transformers import WhisperFeatureExtractor, WhisperTokenizer, pipeline, WhisperProcessor

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

def get_testing_components(model_name):

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, language="english", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="english", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="english", task="transcribe")
    
    return feature_extractor, tokenizer, processor

def compute_wer(pred, tru):
    wer = jiwer.wer(hypothesis=pred, truth=tru)
    return wer

def add_audio_dir(s):
    return os.path.join('/ceph/dpandya/notsofar/nsfd_adap_segments/eval/', s)