"""
Data Preparation Module

This module handles:
1. Loading and preprocessing the XLSum Ukrainian dataset
2. Creating subsets based on configuration
3. Generating toxic versions of articles for detoxification training
4. Saving prepared datasets locally
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Handles data loading, preprocessing, and toxic text generation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataPreparation with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._setup_random_seeds()
        
        # Dataset paths
        self.dataset_dir = Path(self.config['directories']['dataset_dir'])
        self.clean_dir = self.dataset_dir / "clean"
        self.toxic_dir = self.dataset_dir / "toxic"
        
        logger.info("DataPreparation initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.config['directories']['dataset_dir'],
            self.config['directories']['models_dir'],
            self.config['directories']['outputs_dir'],
            self.config['directories']['checkpoints_dir']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {dir_path}")
    
    def _setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['seeds']['global']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seeds set to {seed}")
    
    def load_clean_dataset(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare clean dataset (without toxic versions).
        
        Args:
            force_reload: If True, reload from HuggingFace even if local copy exists
            
        Returns:
            Dictionary with train, validation, test DataFrames
        """
        logger.info("Loading clean dataset...")
        
        # Create clean data directory
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if clean dataset already exists locally
        splits = ['train', 'validation', 'test']
        local_files = {
            split: self.clean_dir / f"{split}.jsonl" for split in splits
        }
        
        # Load from local if exists and not forcing reload
        if not force_reload and all(f.exists() for f in local_files.values()):
            logger.info("Loading clean dataset from local files...")
            datasets = {}
            for split, file_path in local_files.items():
                datasets[split] = pd.read_json(file_path, orient='records', lines=True)
                logger.info(f"Loaded {len(datasets[split])} samples from {file_path}")
            return datasets
        
        # Load from HuggingFace
        logger.info(f"Loading dataset from HuggingFace: {self.config['dataset']['name']}")
        dataset = load_dataset(
            self.config['dataset']['name'],
            self.config['dataset']['language']
        )
        
        # Create subsets based on configuration
        datasets = {}
        subset_sizes = self.config['dataset']['subset_sizes']
        seed = self.config['seeds']['data_sampling']
        
        for split in splits:
            subset_size = subset_sizes.get(split)
            
            if subset_size is not None:
                # Shuffle and select subset
                dataset_split = dataset[split].shuffle(seed=seed).select(range(subset_size))
                logger.info(f"{split}: selected {subset_size} samples")
            else:
                # Use full dataset
                dataset_split = dataset[split]
                logger.info(f"{split}: using full dataset ({len(dataset_split)} samples)")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame({
                'id': dataset_split['id'],
                'title': dataset_split['title'],
                'text': dataset_split['text'],
                'summary': dataset_split['summary']
            })
            
            datasets[split] = df
            
            # Save to local file
            output_path = local_files[split]
            df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Saved {len(df)} samples to {output_path}")
        
        logger.info("Clean dataset loaded and saved successfully")
        return datasets
    
    def generate_toxic_dataset(
        self,
        clean_datasets: Optional[Dict[str, pd.DataFrame]] = None,
        force_regenerate: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate toxic versions of articles for detoxification training.
        
        Args:
            clean_datasets: Dictionary with clean datasets (if None, will load)
            force_regenerate: If True, regenerate even if toxic dataset exists
            
        Returns:
            Dictionary with train, validation, test DataFrames containing toxic versions
        """
        logger.info("Preparing toxic dataset generation...")
        
        # Create toxic data directory
        self.toxic_dir.mkdir(parents=True, exist_ok=True)
        
        # Check configuration
        toxic_config = self.config['dataset']['toxic_dataset']
        
        if toxic_config['use_existing'] and not force_regenerate:
            # Try to load existing toxic datasets
            splits = ['train', 'validation', 'test']
            local_files = {
                split: self.toxic_dir / f"{split}_final.jsonl" for split in splits
            }
            
            if all(f.exists() for f in local_files.values()):
                logger.info("Loading existing toxic datasets from local files...")
                datasets = {}
                for split, file_path in local_files.items():
                    datasets[split] = pd.read_json(file_path, orient='records', lines=True)
                    logger.info(f"Loaded {len(datasets[split])} toxic samples from {file_path}")
                return datasets
        
        # Load clean datasets if not provided
        if clean_datasets is None:
            clean_datasets = self.load_clean_dataset()
        
        # Initialize toxic text generator
        logger.info("Initializing toxic text generator...")
        generator = ToxicTextGenerator(self.config)
        
        # Generate toxic versions for each split
        toxic_datasets = {}
        samples_per_split = toxic_config['samples_per_split']
        
        for split in ['train', 'validation', 'test']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating toxic versions for {split} split")
            logger.info(f"{'='*60}")
            
            num_samples = samples_per_split[split]
            clean_df = clean_datasets[split]
            
            # Sample articles for toxic generation
            if len(clean_df) > num_samples:
                seed = self.config['seeds']['toxic_generation']
                sampled_articles = clean_df.sample(n=num_samples, random_state=seed)
            else:
                sampled_articles = clean_df
            
            # Generate toxic versions
            toxic_df = generator.generate_toxic_versions(
                articles=sampled_articles,
                split_name=split,
                output_dir=self.toxic_dir
            )
            
            toxic_datasets[split] = toxic_df
            
            # Save final version
            output_path = self.toxic_dir / f"{split}_final.jsonl"
            toxic_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            logger.info(f"Saved {len(toxic_df)} toxic samples to {output_path}")
        
        logger.info("\nToxic dataset generation completed!")
        return toxic_datasets
    
    def prepare_all_datasets(
        self,
        force_reload_clean: bool = False,
        force_regenerate_toxic: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Prepare all datasets (clean and toxic versions).
        
        Args:
            force_reload_clean: Force reload clean dataset from HuggingFace
            force_regenerate_toxic: Force regenerate toxic dataset
            
        Returns:
            Dictionary with 'clean' and 'toxic' keys, each containing split DataFrames
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("="*80 + "\n")
        
        # Load clean dataset
        clean_datasets = self.load_clean_dataset(force_reload=force_reload_clean)
        
        # Generate toxic dataset if configured
        toxic_datasets = None
        if self.config['dataset']['toxic_dataset']['regenerate'] or force_regenerate_toxic:
            toxic_datasets = self.generate_toxic_dataset(
                clean_datasets=clean_datasets,
                force_regenerate=force_regenerate_toxic
            )
        elif self.config['dataset']['toxic_dataset']['use_existing']:
            # Try to load existing toxic dataset
            try:
                toxic_datasets = self.generate_toxic_dataset(
                    clean_datasets=clean_datasets,
                    force_regenerate=False
                )
            except Exception as e:
                logger.warning(f"Could not load existing toxic dataset: {e}")
                logger.info("Toxic dataset will be skipped")
        
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION COMPLETED")
        logger.info("="*80 + "\n")
        
        return {
            'clean': clean_datasets,
            'toxic': toxic_datasets
        }


class ToxicTextGenerator:
    """
    Generates toxic versions of articles using LLM.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize toxic text generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        model_config = config['models']['toxic_generator']
        logger.info(f"Loading toxic generator model: {model_config['name']}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=getattr(torch, model_config['torch_dtype']),
            device_map=model_config['device_map']
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.use_cache = True
        
        # Generation parameters
        self.max_new_tokens = model_config['max_new_tokens']
        self.temperature = model_config['temperature']
        self.top_p = model_config['top_p']
        self.batch_size = config['dataset']['toxic_dataset']['batch_size']
        
        # Prompt template
        self.prompt_template = config['toxic_generation']['prompt_template']
        
        logger.info("Toxic text generator initialized")
    
    def _create_prompt(self, title: str, text: str) -> str:
        """Create prompt for toxic text generation."""
        return self.prompt_template.format(title=title, text=text)
    
    def _generate_batch(self, batch_articles: List[Dict]) -> List[str]:
        """
        Generate toxic versions for a batch of articles.
        
        Args:
            batch_articles: List of article dictionaries
            
        Returns:
            List of generated toxic texts
        """
        prompts = [
            self._create_prompt(article['title'], article['text'])
            for article in batch_articles
        ]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for output in outputs:
            generated = self.tokenizer.decode(output, skip_special_tokens=True)
            if "model" in generated:
                result_text = generated.split("model")[-1].strip()
            else:
                result_text = generated.strip()
            results.append(result_text)
        
        return results
    
    def _load_checkpoint(self, checkpoint_path: Path) -> List[Dict]:
        """Load existing checkpoint."""
        if checkpoint_path.exists():
            results = []
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
            logger.info(f"Loaded {len(results)} records from checkpoint")
            return results
        return []
    
    def _save_checkpoint(self, results: List[Dict], checkpoint_path: Path):
        """Save checkpoint."""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Saved checkpoint with {len(results)} records")
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text from unwanted patterns."""
        import re
        patterns = self.config['toxic_generation']['cleanup_patterns']
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def generate_toxic_versions(
        self,
        articles: pd.DataFrame,
        split_name: str,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Generate toxic versions for all articles in the split.
        
        Args:
            articles: DataFrame with articles
            split_name: Name of the split (train/validation/test)
            output_dir: Directory to save results
            
        Returns:
            DataFrame with toxic versions
        """
        checkpoint_freq = self.config['dataset']['toxic_dataset']['checkpoint_frequency']
        checkpoint_path = output_dir / f"{split_name}_checkpoint.jsonl"
        
        # Load existing checkpoint if available
        results = self._load_checkpoint(checkpoint_path)
        start_idx = len(results)
        
        if start_idx > 0:
            logger.info(f"Resuming from checkpoint: {start_idx}/{len(articles)}")
        
        # Convert to list of dicts
        articles_list = articles.to_dict('records')
        
        # Generate in batches
        for i in tqdm(
            range(start_idx, len(articles_list), self.batch_size),
            desc=f"Generating {split_name}",
            initial=start_idx,
            total=len(articles_list)
        ):
            batch = articles_list[i:i + self.batch_size]
            
            # Generate toxic versions
            toxic_texts = self._generate_batch(batch)
            
            # Store results
            for article, toxic_text in zip(batch, toxic_texts):
                cleaned_text = self._clean_generated_text(toxic_text)
                results.append({
                    'id': article['id'],
                    'original_title': article['title'],
                    'original_text': article['text'],
                    'original_summary': article['summary'],
                    'generated_text': cleaned_text
                })
            
            # Save checkpoint periodically
            if (i + self.batch_size) % (checkpoint_freq * self.batch_size) == 0:
                self._save_checkpoint(results, checkpoint_path)
        
        # Final save
        self._save_checkpoint(results, checkpoint_path)
        
        # Convert to DataFrame
        toxic_df = pd.DataFrame(results)
        
        logger.info(f"Generated {len(toxic_df)} toxic versions for {split_name}")
        return toxic_df


def main():
    """Main function for testing."""
    # Initialize data preparation
    data_prep = DataPreparation(config_path="config/config.yaml")
    
    # Prepare all datasets
    datasets = data_prep.prepare_all_datasets(
        force_reload_clean=False,
        force_regenerate_toxic=False
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    for dataset_type, splits in datasets.items():
        if splits is None:
            continue
        print(f"\n{dataset_type.upper()} Dataset:")
        for split_name, df in splits.items():
            print(f"  {split_name}: {len(df)} samples")


if __name__ == "__main__":
    main()