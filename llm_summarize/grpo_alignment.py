"""
GRPO (Group Relative Policy Optimization) Alignment Module

This module handles:
1. Loading SFT-trained model
2. Preparing datasets with toxic examples for detoxification
3. Setting up toxicity reward function
4. Training with GRPOTrainer for alignment
5. Saving aligned model
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GRPOAlignmentTrainer:
    """
    Handles GRPO alignment training for detoxification.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize GRPO Alignment Trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.dataset_dir = Path(self.config['directories']['dataset_dir'])
        self.models_dir = Path(self.config['directories']['models_dir'])
        self.output_dir = Path(self.config['grpo']['training']['output_dir'])
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.reward_pipeline = None
        self.train_dataset = None
        self.eval_dataset = None
        
        logger.info("GRPOAlignmentTrainer initialized successfully")
    
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
                name=wandb_config['run_names']['grpo'],
                config={
                    'model': self.config['models']['base_model']['name'],
                    'task': 'detoxification',
                    'language': 'ukrainian',
                    'method': 'GRPO'
                }
            )
            logger.info("Weights & Biases initialized")
    
    def load_model_and_tokenizer(self, sft_model_path: Optional[str] = None):
        """
        Load SFT-trained model and tokenizer.
        
        Args:
            sft_model_path: Path to SFT model. If None, uses path from config.
        """
        logger.info("Loading model and tokenizer...")
        
        # Determine model path
        if sft_model_path is None:
            sft_model_path = self.models_dir / "sft_final"
        
        sft_model_path = Path(sft_model_path)
        
        if not sft_model_path.exists():
            raise FileNotFoundError(
                f"SFT model not found at {sft_model_path}. "
                "Please train SFT model first using sft_lora_train.py"
            )
        
        model_config = self.config['models']['base_model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(sft_model_path))
        
        # Set padding configuration for GRPO
        padding_side = self.config['tokenization'].get('padding_side', 'left')
        self.tokenizer.padding_side = padding_side
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded from {sft_model_path}")
        logger.info(f"Padding side: {self.tokenizer.padding_side}")
        
        # Load base model
        base_model_name = model_config['name']
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=getattr(torch, model_config['torch_dtype']),
            device_map=model_config['device_map']
        )
        
        logger.info(f"Base model loaded: {base_model_name}")
        
        # Load PEFT adapters from SFT
        self.model = PeftModel.from_pretrained(base_model, str(sft_model_path))
        logger.info(f"SFT LoRA adapters loaded from {sft_model_path}")
        
        # Setup for GRPO training
        self.model.config.use_cache = self.config['hardware']['use_cache']
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_reward_model(self):
        """Load toxicity classifier for reward function."""
        logger.info("Loading toxicity reward model...")
        
        toxicity_config = self.config['models']['toxicity_classifier']
        
        self.reward_pipeline = pipeline(
            task="text-classification",
            model=toxicity_config['name'],
            device_map=toxicity_config['device_map'],
            batch_size=toxicity_config['batch_size']
        )
        
        logger.info(f"Toxicity classifier loaded: {toxicity_config['name']}")
    
    def toxicity_reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function based on toxicity classification.
        Returns higher reward for less toxic completions.
        
        Args:
            completions: List of generated completions
            
        Returns:
            List of reward scores (0-1, higher is better)
        """
        # Handle different completion formats
        if completions and isinstance(completions[0], list):
            texts = []
            for comp in completions:
                if comp and isinstance(comp[0], dict) and "content" in comp[0]:
                    texts.append(comp[0]["content"])
                else:
                    texts.append(" ".join([str(m.get("content", "")) for m in comp]))
        else:
            texts = [str(c) for c in completions]
        
        # Get toxicity predictions
        predictions = self.reward_pipeline(
            texts,
            truncation=True,
            batch_size=self.config['models']['toxicity_classifier']['batch_size'],
            return_all_scores=True
        )
        
        # Extract non-toxic scores as rewards
        rewards: List[float] = []
        label_non_toxic = self.config['grpo']['reward']['label_non_toxic']
        
        for scores in predictions:
            score_non_toxic = next(
                (s["score"] for s in scores if s["label"] == label_non_toxic),
                0.0
            )
            rewards.append(float(score_non_toxic))
        
        return rewards
    
    def load_datasets(self):
        """Load and prepare train and validation datasets with toxic examples."""
        logger.info("Loading datasets...")
        
        # Paths to toxic datasets
        toxic_dir = self.dataset_dir / "toxic"
        train_path = toxic_dir / "train_final.jsonl"
        val_path = toxic_dir / "validation_final.jsonl"
        
        # Also check clean dataset for original texts
        clean_dir = self.dataset_dir / "clean"
        clean_train_path = clean_dir / "train.jsonl"
        clean_val_path = clean_dir / "validation.jsonl"
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Toxic datasets not found in {toxic_dir}. "
                "Please run data_preparation.py with toxic generation enabled."
            )
        
        # Load toxic datasets
        train_df = pd.read_json(train_path, orient='records', lines=True)
        val_df = pd.read_json(val_path, orient='records', lines=True)
        
        logger.info(f"Loaded {len(train_df)} training samples (toxic)")
        logger.info(f"Loaded {len(val_df)} validation samples (toxic)")
        
        # Load clean datasets for original texts
        clean_train_df = pd.read_json(clean_train_path, orient='records', lines=True)
        clean_val_df = pd.read_json(clean_val_path, orient='records', lines=True)
        
        logger.info(f"Loaded {len(clean_train_df)} clean training samples")
        logger.info(f"Loaded {len(clean_val_df)} clean validation samples")
        
        # Apply subset sizes if configured
        grpo_samples_train = self.config['dataset'].get('subset_sizes', {}).get('train')
        grpo_samples_val = self.config['dataset'].get('subset_sizes', {}).get('validation')
        
        # For GRPO, we might want a smaller subset due to computational cost
        # Use config or take subset from toxic data
        if grpo_samples_train and len(train_df) > grpo_samples_train:
            train_df = train_df.sample(n=grpo_samples_train, random_state=self.config['seeds']['global'])
            logger.info(f"Sampled {grpo_samples_train} training samples for GRPO")
        
        if grpo_samples_val and len(val_df) > grpo_samples_val:
            val_df = val_df.sample(n=grpo_samples_val, random_state=self.config['seeds']['global'])
            logger.info(f"Sampled {grpo_samples_val} validation samples for GRPO")
        
        # Format datasets
        self.train_dataset = self._format_dataset(train_df)
        self.eval_dataset = self._format_dataset(val_df)
        
        logger.info("Datasets formatted and ready for GRPO training")
    
    def _format_prompt(self, text: str) -> str:
        """
        Create prompt for GRPO using Gemma's chat template.
        
        Args:
            text: Original article text
            
        Returns:
            Formatted prompt string
        """
        instruction = self.config['grpo']['prompt']['instruction'].strip()
        
        user_msg = f"{instruction}\n\nТЕКСТ:\n{text.strip()}"
        
        messages = [
            {
                "role": "user",
                "content": user_msg
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _format_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Format DataFrame into Dataset with prompts for GRPO.
        
        Args:
            df: DataFrame with column 'original_text' (from toxic dataset)
            
        Returns:
            HuggingFace Dataset object
        """
        formatted_data = []
        
        for _, row in df.iterrows():
            # Use original text (before toxic generation)
            text = row['original_text']
            
            # Create prompt
            prompt = self._format_prompt(text)
            
            formatted_data.append({'prompt': prompt})
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)
        
        return dataset
    
    def train(self):
        """Execute GRPO alignment training."""
        logger.info("Starting GRPO alignment training...")
        
        # Setup W&B
        self._setup_wandb()
        
        # Training configuration
        training_config = self.config['grpo']['training']
        print(GRPOConfig())
        # Create GRPOConfig
        grpo_config = GRPOConfig(
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
        
            seed=training_config['seed'],
            report_to="wandb" if self.config['wandb']['enabled'] else "none",
            save_total_limit=3,
            # GRPO-specific parameters
            max_completion_length=self.config['grpo']['max_new_tokens'],
            temperature=self.config['grpo']['temperature'],
            num_generations=self.config['grpo']['num_generations'],
        )
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=self.toxicity_reward_function,
        )
        
        logger.info("GRPOTrainer initialized")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
        logger.info(f"Training epochs: {training_config['num_train_epochs']}")
        logger.info(f"Generations per prompt: {self.config['grpo']['num_generations']}")
        
        # Train
        logger.info("\n" + "="*80)
        logger.info("STARTING GRPO ALIGNMENT")
        logger.info("="*80 + "\n")
        
        trainer.train()
        
        logger.info("\n" + "="*80)
        logger.info("GRPO ALIGNMENT COMPLETED")
        logger.info("="*80 + "\n")
        
        # Save model
        self.save_model(trainer)
        
        # Finish W&B
        if self.config['wandb']['enabled']:
            wandb.finish()
    
    def save_model(self, trainer: GRPOTrainer):
        """
        Save aligned model and tokenizer.
        
        Args:
            trainer: GRPOTrainer instance
        """
        logger.info("Saving aligned model...")
        
        # Save to models directory
        final_model_path = self.models_dir / "grpo_final"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        trainer.model.save_pretrained(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Model saved to {final_model_path}")
        
        # Also save to output directory
        trainer.save_model()
        logger.info(f"Model also saved to {self.output_dir}")
    
    def run_full_pipeline(self, sft_model_path: Optional[str] = None):
        """
        Run complete GRPO alignment pipeline.
        
        Args:
            sft_model_path: Path to SFT model (optional)
        """
        logger.info("\n" + "="*80)
        logger.info("GRPO ALIGNMENT PIPELINE")
        logger.info("="*80 + "\n")
        
        try:
            # Load model and tokenizer
            self.load_model_and_tokenizer(sft_model_path=sft_model_path)
            
            # Load reward model
            self.load_reward_model()
            
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
    """Main function for GRPO alignment training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GRPO Alignment for Detoxification')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sft-model',
        type=str,
        default=None,
        help='Path to SFT model (optional, uses config path if not provided)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = GRPOAlignmentTrainer(config_path=args.config)
    trainer.run_full_pipeline(sft_model_path=args.sft_model)


if __name__ == "__main__":
    main()