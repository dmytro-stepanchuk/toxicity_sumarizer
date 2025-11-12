"""
SFT (Supervised Fine-Tuning) with LoRA Module

This module handles:
1. Loading pretrained model and applying LoRA adapters
2. Preparing dataset with proper prompts for summarization
3. Training with SFTTrainer from TRL
4. Saving trained model
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SFTLoRATrainer:
    """
    Handles Supervised Fine-Tuning with LoRA for summarization task.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize SFT LoRA Trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.dataset_dir = Path(self.config['directories']['dataset_dir']) / "clean"
        self.models_dir = Path(self.config['directories']['models_dir'])
        self.output_dir = Path(self.config['sft']['training']['output_dir'])
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        logger.info("SFTLoRATrainer initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config['wandb']['enabled']:
            wandb_config = self.config['wandb']
            
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                name=wandb_config['run_names']['sft'],
                config={
                    'model': self.config['models']['base_model']['name'],
                    'task': 'summarization',
                    'language': 'ukrainian',
                    'method': 'SFT-LoRA'
                }
            )
            logger.info("Weights & Biases initialized")
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer, apply LoRA configuration."""
        logger.info("Loading model and tokenizer...")
        
        model_config = self.config['models']['base_model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            use_fast=self.config['tokenization']['use_fast']
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded: {model_config['name']}")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=getattr(torch, model_config['torch_dtype']),
            device_map=model_config['device_map'],
            attn_implementation=model_config.get('attn_implementation', 'eager')
        )
        
        logger.info(f"Base model loaded: {model_config['name']}")
        
        # Setup LoRA
        lora_config_dict = self.config['sft']['lora']
        lora_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            target_modules=lora_config_dict['target_modules'],
            lora_dropout=lora_config_dict['lora_dropout'],
            bias=lora_config_dict['bias'],
            task_type=lora_config_dict['task_type']
        )
        
        # Prepare model for training
        if self.config['sft']['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
        # Disable cache for training
        self.model.config.use_cache = self.config['hardware']['use_cache']
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA adapters applied successfully")
    
    def load_datasets(self):
        """Load and prepare train and validation datasets."""
        logger.info("Loading datasets...")
        
        # Load clean datasets
        train_path = self.dataset_dir / "train.jsonl"
        val_path = self.dataset_dir / "validation.jsonl"
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Clean datasets not found in {self.dataset_dir}. "
                "Please run data_preparation.py first."
            )
        
        train_df = pd.read_json(train_path, orient='records', lines=True)
        val_df = pd.read_json(val_path, orient='records', lines=True)
        
        logger.info(f"Loaded {len(train_df)} training samples")
        logger.info(f"Loaded {len(val_df)} validation samples")
        
        # Format datasets
        self.train_dataset = self._format_dataset(train_df)
        self.eval_dataset = self._format_dataset(val_df)
        
        logger.info("Datasets formatted and ready for training")
    
    def _create_prompt(self, text: str, summary: str) -> str:
        """
        Create training prompt in Gemma chat format.
        
        Args:
            text: Original article text
            summary: Target summary
            
        Returns:
            Formatted prompt string
        """
        instruction = self.config['sft']['prompt']['instruction'].strip()
        
        prompt = (
            "<start_of_turn>user\n"
            f"{instruction}\n\n"
            f"ТЕКСТ:\n{text.strip()}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{summary.strip()}<end_of_turn>"
        )
        
        return prompt
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified number of tokens.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _format_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Format DataFrame into Dataset with prompts.
        
        Args:
            df: DataFrame with columns: text, summary
            
        Returns:
            HuggingFace Dataset object
        """
        max_seq_len = self.config['tokenization']['max_seq_length']
        gen_max_tokens = self.config['tokenization']['gen_max_new_tokens']
        
        formatted_data = []
        
        for _, row in df.iterrows():
            # Truncate text and summary
            text = self._truncate_text(row['text'], max_seq_len - gen_max_tokens)
            summary = self._truncate_text(row['summary'], gen_max_tokens)
            
            # Create prompt
            prompt = self._create_prompt(text, summary)
            
            formatted_data.append({'text': prompt})
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)
        
        return dataset
    
    def train(self):
        """Execute SFT training with LoRA."""
        logger.info("Starting SFT training...")
        
        # Setup W&B
        self._setup_wandb()
        
        # Training configuration
        training_config = self.config['sft']['training']
        
        # Create SFTConfig
        sft_config = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_ratio=training_config['warmup_ratio'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_strategy=training_config['eval_strategy'],
            eval_steps=training_config['eval_steps'],
            bf16=training_config['bf16'],
            fp16=self.config['hardware']['fp16'],
           
            packing=training_config['packing'],
            dataset_text_field="text",
            gradient_checkpointing=training_config['gradient_checkpointing'],
            seed=training_config['seed'],
            report_to="wandb" if self.config['wandb']['enabled'] else "none",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        print(sft_config)
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
        
        logger.info("SFTTrainer initialized")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
        logger.info(f"Training epochs: {training_config['num_train_epochs']}")
        
        # Train
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80 + "\n")
        
        trainer.train()
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80 + "\n")
        
        # Save model
        self.save_model(trainer)
        
        # Finish W&B
        if self.config['wandb']['enabled']:
            wandb.finish()
    
    def save_model(self, trainer: SFTTrainer):
        """
        Save trained model and tokenizer.
        
        Args:
            trainer: SFTTrainer instance
        """
        logger.info("Saving model...")
        
        # Save to models directory
        final_model_path = self.models_dir / "sft_final"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        trainer.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Model saved to {final_model_path}")
        
        # Also save to output directory
        trainer.save_model()
        logger.info(f"Model also saved to {self.output_dir}")
    
    def run_full_pipeline(self):
        """Run complete SFT training pipeline."""
        logger.info("\n" + "="*80)
        logger.info("SFT LORA TRAINING PIPELINE")
        logger.info("="*80 + "\n")
        
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Load datasets
            self.load_datasets()
            
            # Train
            self.train()
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function for SFT training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SFT LoRA Training for Summarization')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = SFTLoRATrainer(config_path=args.config)
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()