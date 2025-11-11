import utils as eu
import os
import pandas as pd
import numpy as np
import datasets
import adapters
from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import torch # Import torch to check for CUDA availability

# Setting config file
config_path = 'eval_config.yaml'
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
    adapter_paths = [os.path.join(adap_dir, x) for x in os.listdir(adap_dir) if "seq" in x if "20E" in x if os.path.isdir(os.path.join(adap_dir, x))]
    print(f"Found adapters: {adapter_paths}")
    
    
    # Optimal batch size for GPU inference
    OPTIMAL_BATCH_SIZE = 1 

    for adap_path in adapter_paths:
        # We change the model init here based on the experiments we did in scrap.ipynb
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
        )

        whisper_model.config.max_length=512
        whisper_model.config.use_cache=False

        # Init whisper model for adapters
        adapters.init(whisper_model)        
        
        # Load and activate the adapter
        adapter_name = whisper_model.load_adapter(adap_path)
        whisper_model.set_active_adapters(adapter_name)
        
        print(f"\n--- Starting Evaluation for Adapter: **{adapter_name}** ---")
        # Initialize the base pipeline outside the loop
        eval_pipeline = AutomaticSpeechRecognitionPipeline(
            model=whisper_model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            device="cuda:0", # Use GPU 0, or -1 for CPU
            chunk_length_s=30
        )
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

        predictions = []
        # Use KeyDataset with tqdm for correct progress tracking
        
        for prediction in tqdm(
            eval_pipeline(
                KeyDataset(eval_ds, 'audio'),
                batch_size=OPTIMAL_BATCH_SIZE, # Increased batch size for speed
                generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
                max_new_tokens=256
            ), total=len(eval_ds), desc=f'Evaluating {adapter_name}'
        ):
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