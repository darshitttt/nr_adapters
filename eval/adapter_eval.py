import utils as eu
import os
import pandas as pd
import numpy as np
import datasets
import adapters
from transformers import AutomaticSpeechRecognitionPipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import torch # Import torch to check for CUDA availability

# Setting config file
config_path = 'eval_config.yaml'
# Assuming this script is run with CUDA_VISIBLE_DEVICES="1",
# the GPU index visible to the script will be 0.
# If no specific GPU is needed, just remove the line, but it's good practice.
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(["0"]) 

def main():
    config = eu.load_config(config_path)

    # Setup from config
    model_name = config['model']['model_name']
    model_size = config['model']['model_size']
    
    # Setting up eval features
    eval_df = pd.read_csv(config['data']['eval_csv']).sample(540, random_state=42)
    eval_df['transcription'] = eval_df['segmented_text'].apply(eu.clean_text)
    eval_df['segmented_audio_file'] = eval_df['segmented_audio_file'].apply(eu.add_audio_dir)

    eval_files = {'audio':list(eval_df['segmented_audio_file'])}
    eval_ds = datasets.Dataset.from_dict(eval_files).cast_column('audio', datasets.Audio(sampling_rate=16000))
    # It's better to keep 'segmented_text' as a separate column for comparison
    eval_ds = eval_ds.add_column("text", list(eval_df['segmented_text']))

    # Load components
    feature_extractor, tokenizer, processor = eu.get_testing_components(model_name)
    
    # Add adapter
    adap_dir = config["adapter"]["dir"]
    # Get a list of full adapter paths
    adapter_paths = [os.path.join(adap_dir, x) for x in os.listdir(adap_dir) if "seq" in x if os.path.isdir(os.path.join(adap_dir, x))]
    print(f"Found adapters: {adapter_paths}")
    
    
    # Optimal batch size for GPU inference
    OPTIMAL_BATCH_SIZE = 1 

    for adap_path in adapter_paths:

        whisper_model = adapters.WhisperAdapterModel.from_pretrained(
        model_name,
        language="english",
        task="transcribe"
    )
        whisper_model.config.max_length=512
        whisper_model.config.use_cache=False
        
        # Load and activate the adapter
        adapter_name = whisper_model.load_adapter(adap_path, set_active=True)
        # The line below is redundant after set_active=True in load_adapter, but harmless
        whisper_model.set_active_adapters(adapter_name) 
        
        print(f"\n--- Starting Evaluation for Adapter: **{adapter_name}** ---")
        # Initialize the base pipeline outside the loop
        eval_pipeline = AutomaticSpeechRecognitionPipeline(
            model=whisper_model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device="cuda:0", # Use GPU 0, or -1 for CPU
            chunk_length_s=30
        )
        decoder_prompt_list = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        decoder_input_ids = torch.tensor(decoder_prompt_list, device="cuda:0")

        predictions = []
        # Use KeyDataset with tqdm for correct progress tracking
        #dataset_for_pipeline = KeyDataset(eval_ds, 'audio')
        
        for prediction in tqdm(
            eval_pipeline(
                KeyDataset(eval_ds, 'audio'),
                batch_size=OPTIMAL_BATCH_SIZE, # Increased batch size for speed
                generate_kwargs={
                     "decoder_input_ids": decoder_input_ids,
                     "forced_decoder_ids": None
                },
                max_new_tokens=256
            ), total=len(eval_ds), desc=f'Evaluating {adapter_name}'
        ):
            # The pipeline returns a list of predictions when given KeyDataset
            # and batch_size > 1. Extend the list with the batch results.
                predictions.append(prediction)

        # Check for matching lengths (optional safety check)
        if len(predictions) != len(eval_ds):
            print(f"Warning: Number of predictions ({len(predictions)}) does not match dataset size ({len(eval_ds)}).")
        
        col_name = 'preds_text_'+adapter_name
        
        # Extract the 'text' key from the prediction dictionaries
        predicted_texts = [p['text'] for p in predictions]
        
        eval_df[col_name] = predicted_texts
        eval_df[col_name] = eval_df[col_name].apply(eu.clean_text)

        # Calculating WER
        output_file = os.path.join(adap_dir, f'eval_{adapter_name}.csv')
        
        # Use eu.compute_wer to calculate Word Error Rate
        eval_df['wer'] = eval_df.apply(lambda x: eu.compute_wer(x[col_name], x['transcription']), axis=1)
        mean_wer = eval_df["wer"].mean()
        print(f'✅ WER for **{adapter_name}**: **{mean_wer:.4f}**')
        
        # Save results for this adapter
        eval_df_to_save = eval_df[['segmented_audio_file', 'transcription', col_name, 'wer']].copy()
        eval_df_to_save.to_csv(output_file, index=False)

        print(f"Evaluation complete for {adapter_name}. Results saved to {output_file}")
        
        # Important: Deactivate the adapter before the next loop iteration
        whisper_model.set_active_adapters(None)

if __name__ == "__main__":
    main()
        