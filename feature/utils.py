import pandas as pd
import numpy as np
import os
import librosa
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

def add_audio_dir(s):
    return os.path.join('/ceph/dpandya/notsofar/nsfd_adap_segments/new_train/', s)
