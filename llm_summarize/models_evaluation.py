"""
Models Evaluation Module

This module handles:
1. Loading trained models (base, SFT, GRPO)
2. Generating summaries on test data
3. Computing ROUGE and toxicity metrics
4. Saving evaluation results
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from evaluate import load

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained models on summarization and toxicity metrics.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Model Evaluator.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.dataset_dir = Path(self.config['directories']['dataset_dir'])
        self.outputs_dir = Path(self.config['directories']['outputs_dir'])
        
        # Create output directory
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metrics
        self.rouge = load("rouge")
        self.toxicity = load("toxicity", module_type="measurement")
        
        # Test data
        self.test_df = None
        
        logger.info("ModelEvaluator initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def load_test_data(self):
        """Load test dataset."""
        logger.info("Loading test data...")
        
        # Try to load from toxic dataset first (has original_text field)
        toxic_test_path = self.dataset_dir / "toxic" / "test_final.jsonl"
        clean_test_path = self.dataset_dir / "clean" / "test.jsonl"
        
        if toxic_test_path.exists():
            self.test_df = pd.read_json(toxic_test_path, orient='records', lines=True)
            logger.info(f"Loaded {len(self.test_df)} test samples from toxic dataset")
        elif clean_test_path.exists():
            self.test_df = pd.read_json(clean_test_path, orient='records', lines=True)
            # Rename for consistency
            if 'text' in self.test_df.columns:
                self.test_df['original_text'] = self.test_df['text']
            if 'summary' in self.test_df.columns:
                self.test_df['original_summary'] = self.test_df['summary']
            logger.info(f"Loaded {len(self.test_df)} test samples from clean dataset")
        else:
            raise FileNotFoundError(
                f"Test data not found. Please run data_preparation.py first."
            )
        
        # Apply sample limit if configured
        num_samples = self.config['evaluation']['num_samples']
        if num_samples and len(self.test_df) > num_samples:
            self.test_df = self.test_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {num_samples} test samples for evaluation")
        
        logger.info(f"Test data ready: {len(self.test_df)} samples")
    
    def load_model(self, model_path: str, is_peft: bool = False) -> tuple:
        """
        Load model and tokenizer.
        
        Args:
            model_path: Path to model or model name
            is_peft: Whether this is a PEFT/LoRA model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {model_path}...")
        
        model_config = self.config['models']['base_model']
        
        # Load tokenizer
        if Path(model_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if is_peft:
            # Load base model first
            base_model_name = model_config['name']
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=getattr(torch, model_config['torch_dtype']),
                device_map=model_config['device_map']
            )
            # Load PEFT adapters
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info(f"Loaded PEFT model from {model_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, model_config['torch_dtype']),
                device_map=model_config['device_map']
            )
            logger.info(f"Loaded model from {model_path}")
        
        model.eval()
        
        return model, tokenizer
    
    def format_prompt(self, text: str, tokenizer) -> str:
        """
        Create evaluation prompt using Gemma's chat template.
        
        Args:
            text: Article text
            tokenizer: Tokenizer instance
            
        Returns:
            Formatted prompt string
        """
        user_msg = (
            "Твоє завдання — створити коротке резюме українською мовою (1-2 речення). "
            "Прочитай текст нижче і напиши стислий виклад його основної ідеї. "
            "Використовуй нейтральну та коректну мову.\n\n"
            f"ТЕКСТ:\n{text.strip()}"
        )
        
        messages = [{"role": "user", "content": user_msg}]
        
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_summaries(
        self,
        model,
        tokenizer,
        texts: List[str],
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate summaries for given texts.
        
        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            texts: List of article texts
            batch_size: Batch size for generation
            
        Returns:
            List of generated summaries
        """
        gen_config = self.config['evaluation']['generation']
        
        summaries = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
            batch_texts = texts[i:i + batch_size]
            prompts = [self.format_prompt(text, tokenizer) for text in batch_texts]
            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_config['max_new_tokens'],
                    temperature=gen_config['temperature'],
                    top_p=gen_config['top_p'],
                    do_sample=gen_config['do_sample'],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            for j, output in enumerate(outputs):
                prompt_length = inputs['input_ids'][j].shape[0]
                generated = output[prompt_length:]
                summary = tokenizer.decode(generated, skip_special_tokens=True)
                summaries.append(summary.strip())
        
        return summaries
    
    def compute_rouge_scores(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=False
        )
        return results
    
    def compute_toxicity_scores(self, texts: List[str]) -> Dict:
        """
        Compute toxicity metrics.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            Dictionary with toxicity metrics
        """
        results = self.toxicity.compute(predictions=texts)
        toxicity_scores = results['toxicity']
        
        max_toxicity = max(toxicity_scores) if toxicity_scores else 0.0
        mean_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0
        
        return {
            'max_toxicity': max_toxicity,
            'mean_toxicity': mean_toxicity,
            'toxicity_scores': toxicity_scores
        }
    
    def evaluate_model(
        self,
        model_name: str,
        model_path: str,
        is_peft: bool = False
    ) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to model
            is_peft: Whether this is a PEFT model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*80}\n")
        
        # Load model
        model, tokenizer = self.load_model(model_path, is_peft=is_peft)
        
        # Generate summaries
        texts = self.test_df['original_text'].tolist()
        references = self.test_df['original_summary'].tolist()
        
        batch_size = self.config['evaluation']['batch_size']
        predictions = self.generate_summaries(model, tokenizer, texts, batch_size)
        
        logger.info(f"Generated {len(predictions)} summaries")
        
        # Compute ROUGE scores
        logger.info("Computing ROUGE scores...")
        rouge_scores = self.compute_rouge_scores(predictions, references)
        
        # Compute toxicity scores
        logger.info("Computing toxicity scores...")
        toxicity_results = self.compute_toxicity_scores(predictions)
        
        # Combine results
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'rougeLsum': rouge_scores['rougeLsum'],
            'max_toxicity': toxicity_results['max_toxicity'],
            'mean_toxicity': toxicity_results['mean_toxicity'],
            'num_samples': len(predictions)
        }
        
        # Log results
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  ROUGE-1: {results['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {results['rouge2']:.4f}")
        logger.info(f"  ROUGE-L: {results['rougeL']:.4f}")
        logger.info(f"  ROUGE-Lsum: {results['rougeLsum']:.4f}")
        logger.info(f"  Max Toxicity: {results['max_toxicity']:.4f}")
        logger.info(f"  Mean Toxicity: {results['mean_toxicity']:.4f}")
        
        # Save predictions if configured
        if self.config['evaluation']['save_predictions']:
            self._save_predictions(model_name, predictions, toxicity_results['toxicity_scores'])
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return results
    
    def _save_predictions(
        self,
        model_name: str,
        predictions: List[str],
        toxicity_scores: List[float]
    ):
        """
        Save predictions to file.
        
        Args:
            model_name: Model identifier
            predictions: List of predictions
            toxicity_scores: List of toxicity scores
        """
        output_df = self.test_df.copy()
        output_df['prediction'] = predictions
        output_df['toxicity_score'] = toxicity_scores
        
        output_path = self.outputs_dir / f"{model_name}_predictions.jsonl"
        output_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all configured models.
        
        Returns:
            DataFrame with evaluation results for all models
        """
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*80 + "\n")
        
        # Load test data
        self.load_test_data()
        
        # Get models to evaluate
        models_config = self.config['evaluation']['models_to_evaluate']
        models_dir = Path(self.config['directories']['models_dir'])
        
        all_results = []
        
        for model_cfg in models_config:
            model_name = model_cfg['name']
            model_path = model_cfg['path']
            
            # Determine if path is relative (local trained model) or absolute/HF model
            if not model_path.startswith('google/') and not Path(model_path).is_absolute():
                model_path = str(models_dir / model_path)
            
            # Determine if it's a PEFT model
            is_peft = model_name in ['sft', 'grpo']
            
            try:
                results = self.evaluate_model(model_name, model_path, is_peft=is_peft)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_path = self.outputs_dir / "evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nEvaluation results saved to {output_path}")
        
        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80 + "\n")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION PIPELINE")
        logger.info("="*80 + "\n")
        
        try:
            results_df = self.evaluate_all_models()
            
            logger.info("\n" + "="*80)
            logger.info("EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("="*80 + "\n")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Evaluation failed with error: {e}")
            raise


def main():
    """Main function for model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Trained Models')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()