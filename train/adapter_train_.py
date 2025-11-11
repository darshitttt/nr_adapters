import yaml
import os
import adapters
import datasets
from transformers import TrainingArguments
from adapters import AdapterTrainer, Seq2SeqAdapterTrainer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import utils as tu # Assuming this utility file contains necessary components
import json
from dataclasses import asdict

# --- Configuration Constants (Will be loaded from YAML) ---
CONFIG_FILE = 'train_config.yaml'


def main():
    config = tu.load_config(CONFIG_FILE)

    # --- Setup from Config ---
    model_name = config['model']['model_name']
    model_size = config['model']['model_size']
    train_ds_dir = config['data']['train_ds_dir']
    #rfs_to_train = config['experiment']['reduction_factors']
    rfs_to_train = [4,8,16,32]
    adapter_base_dir = config['data']['adapter_base_dir']
    
    # Environment Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(["0"])

    # --- Model and Processor Initialization (Done once) ---
    print("Initializing Whisper Components...")
    feature_extractor, tokenizer, processor = tu.get_training_components(model_name)
    data_collator = tu.DataCollatorSpeechSeq2SeqWithPadding(processor)
    
    # --- Dataset Loading and Splitting ---
    print(f'Loading dataset from: {train_ds_dir}/{model_size}')
    # Load the full dataset (which is intended to be split)
    full_ds = datasets.load_from_disk(os.path.join(train_ds_dir, model_size))
    
    # Apply filter function from utils
    full_ds = full_ds.filter(tu.filter_labels, input_columns=["labels"])
    
    # Split the dataset: 85% train, 15% validation
    split_datasets = full_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split_datasets['train']
    eval_ds = split_datasets['test']
    
    print(f"Dataset split completed. Train samples: {len(train_ds)}, Validation samples: {len(eval_ds)}")
    
    # Dictionary to store results (loss/WER) for analysis
    experiment_results = {}

    # --- Training Loop over Reduction Factors (RFs) ---
    for rf in rfs_to_train:
        print(f"\n--- Starting Training for reduction_factor (RF) = {rf} ---")
        adapter_name = f'seqBN_r{rf}_20e'
        
        # 1. Initialize a clean model for this RF run
        # Note: Must initialize a fresh model to avoid adapter contamination
        whisper_model = adapters.WhisperAdapterModel.from_pretrained(
            model_name, 
            language='english', 
            task='transcribe'
        )
        whisper_model.config.max_length = 512
        whisper_model.config.use_cache = False
        whisper_model.freeze_encoder()
        
        # 2. Define Adapter Configuration (Instantiate with current RF)
        adapter_config_instance = adapters.SeqBnConfig(mh_adapter=False, output_adapter=True, reduction_factor=rf)
        
        # 3. Add the Adapter (using the name and config instance)
        whisper_model.add_adapter(adapter_name=adapter_name, config=adapter_config_instance)
        
        # 4. Set Adapter as Active
        whisper_model.set_active_adapters(adapter_name)

        # 5. Define Training Arguments (Update output directory)
        current_save_dir = os.path.join(adapter_base_dir, model_size, adapter_name)
        if not os.path.exists(current_save_dir):
            os.makedirs(current_save_dir)
        print(f'Trained Adapter will be saved at {current_save_dir}')

        # Training arguments from config
        #training_args_config = config['training_args']
        whisper_model.train_adapter(adapter_name)
        
        training_args = TrainingArguments(
            output_dir=current_save_dir, 
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-4,
            warmup_steps=50,
            num_train_epochs=20,
            fp16=True,
            logging_steps=50,
            #save_steps=training_args_config['save_steps'], # Added save_steps for checkpoints
            #evaluation_strategy="epoch", # CRITICAL: Enables evaluation on eval_dataset
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # 6. Initialize Trainer
        trainer = AdapterTrainer(
            model=whisper_model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=train_ds,
            eval_dataset=eval_ds, # CRITICAL: Include validation set here
            args=training_args,
        )

        # 7. Start Training
        
        trainer.train()
        
        # 8. Store Validation Metrics
        # Extract the relevant validation metrics from the logs (usually the last evaluation step)
        eval_metrics = {}
        for log in trainer.state.log_history:
            if 'eval_loss' in log:
                eval_metrics.update({k: v for k, v in log.items() if k.startswith('eval_')})
        
        experiment_results[adapter_name] = {
            'reduction_factor': rf,
            #'final_train_loss': train_result.training_loss,
            'log_history': trainer.state.log_history,
            'final_eval_metrics': eval_metrics
        }
        
        print(f"Final Validation Metrics for RF={rf}: {eval_metrics}")

        # 9. Save the Final Adapter
        whisper_model.save_adapter(current_save_dir, adapter_name)
        print(f'Adapter "{adapter_name}" saved to {current_save_dir}')

    # --- Final Analysis (Saving All Results) ---
    results_path = os.path.join(adapter_base_dir, model_size, 'experiment_results.json')
    # Use JSON to store results for easy analysis later
    with open(results_path, 'w') as f:
        # Convert dataclasses/other complex objects to serializable dicts
        serializable_results = experiment_results
        json.dump(serializable_results, f, indent=4)
        
    print(f"\nAll experiment results saved to {results_path}")
    print("\nTraining run complete.")

if __name__ == '__main__':
    main()
