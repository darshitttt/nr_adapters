import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import datasets
import re

def clean_text(s):
    # Convert to string to handle potential non-string inputs gracefully
    s = str(s)
    # 1. Remove text enclosed in angle brackets (e.g., <ST>, <UNKNOWN>)
    s = re.sub(r'<[^>]+>', '', s)
    # 2. Remove punctuation
    # This regex matches any character that is NOT a word character (alphanumeric + underscore) or whitespace
    s = re.sub(r'[^\w\s]', '', s)
    # 3. Remove extra spaces (replace multiple spaces with a single space)
    s = re.sub(r'\s+', ' ', s)
    # 4. Remove leading and trailing spaces
    s = s.strip()
    # 5. Convert to lowercase for case-insensitive comparison
    return s.lower()


segmented_df = pd.read_csv('/ceph/dpandya/notsofar/train_set/240825.1_train/new_segmented_audios/new_segmented_audios.csv')
segmented_df = segmented_df.dropna(subset=['text'])
segmented_df['text'] = segmented_df['text'].apply(clean_text)

aud_files = {'audio':list(segmented_df['segmented_audio_file'])}
train_ds = datasets.Dataset.from_dict(aud_files).cast_column('audio', datasets.Audio(sampling_rate=16000))
train_ds = train_ds.add_column("text", list(segmented_df['text']))

whisper_model_name = "openai/whisper-medium"
language = "english"
task = 'transcribe'

feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name, language=language, task=task)

# Prepare Dataset function that will help us in preparing our data for training and evaluation
def prepare_dataset(batch):
    
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"], truncation=True).input_ids
    return batch

train_ds = train_ds.map(prepare_dataset, remove_columns=["audio", "text"])
print('Done with mapping')

save_dir = "/ceph/dpandya/notsofar/train_set_features/large/nsfdTrain/"

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_ds.save_to_disk(save_dir)