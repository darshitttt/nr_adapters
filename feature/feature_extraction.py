import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import datasets
import utils

# The original train dir has audio segments with a max len of 25s
# The new_train dir has audio segments with a max len of 20s (to increase the number of segments)
audio_dir = '/ceph/dpandya/notsofar/nsfd_adap_segments/new_train/'

segmented_df = pd.read_csv('/ceph/dpandya/notsofar/nsfd_adap_segments/new_train_segments.csv')
segmented_df['segmented_audio_file'] = segmented_df['segmented_audio_file'].apply(utils.add_audio_dir)
segmented_df = segmented_df.dropna(subset=['segmented_text'])
segmented_df['text'] = segmented_df['segmented_text'].apply(utils.clean_text)

aud_files = {'audio':list(segmented_df['segmented_audio_file'])}
train_ds = datasets.Dataset.from_dict(aud_files).cast_column('audio', datasets.Audio(sampling_rate=16000))
train_ds = train_ds.add_column("text", list(segmented_df['segmented_text']))

whisper_model_name = "openai/whisper-large-v3"
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

save_dir = "/ceph/dpandya/notsofar/nsfd_adap_segments/train_set_features/new_large/"

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_ds.save_to_disk(save_dir)