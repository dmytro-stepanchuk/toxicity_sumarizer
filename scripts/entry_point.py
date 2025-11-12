"""
Entry Point for LLM Summarization and Detoxification Pipeline

This script orchestrates the complete training and evaluation pipeline:
1. Data Preparation (clean + toxic datasets)
2. SFT LoRA Training
3. GRPO Alignment Training
4. Model Evaluation
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_summarize.data_preparation_and_generation import DataPreparation
from llm_summarize.sft_lora_train import SFTLoRATrainer
from llm_summarize.grpo_alignment import GRPOAlignmentTrainer
from llm_summarize.models_evaluation import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline orchestrator for the complete training workflow.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Track pipeline state
        self.start_time = None
        self.stages_completed = []
        
        logger.info("Pipeline initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _log_stage_start(self, stage_name: str):
        """Log the start of a pipeline stage."""
        logger.info("\n" + "="*100)
        logger.info(f"STAGE: {stage_name}")
        logger.info("="*100 + "\n")
    
    def _log_stage_complete(self, stage_name: str):
        """Log the completion of a pipeline stage."""
        self.stages_completed.append(stage_name)
        logger.info("\n" + "="*100)
        logger.info(f"STAGE COMPLETED: {stage_name}")
        logger.info("="*100 + "\n")
    
    def run_data_preparation(
        self,
        force_reload_clean: bool = False,
        force_regenerate_toxic: bool = False
    ):
        """
        Run data preparation stage.
        
        Args:
            force_reload_clean: Force reload clean dataset from HuggingFace
            force_regenerate_toxic: Force regenerate toxic dataset
        """
        self._log_stage_start("DATA PREPARATION")
        
        try:
            data_prep = DataPreparation(config_path=self.config_path)
            datasets = data_prep.prepare_all_datasets(
                force_reload_clean=force_reload_clean,
                force_regenerate_toxic=force_regenerate_toxic
            )
            
            self._log_stage_complete("DATA PREPARATION")
            return datasets
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def run_sft_training(self):
        """Run SFT LoRA training stage."""
        self._log_stage_start("SFT LORA TRAINING")
        
        try:
            trainer = SFTLoRATrainer(config_path=self.config_path)
            trainer.run_full_pipeline()
            
            self._log_stage_complete("SFT LORA TRAINING")
            
        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            raise
    
    def run_grpo_alignment(self, sft_model_path: str = None):
        """
        Run GRPO alignment stage.
        
        Args:
            sft_model_path: Optional custom path to SFT model
        """
        self._log_stage_start("GRPO ALIGNMENT")
        
        try:
            trainer = GRPOAlignmentTrainer(config_path=self.config_path)
            trainer.run_full_pipeline(sft_model_path=sft_model_path)
            
            self._log_stage_complete("GRPO ALIGNMENT")
            
        except Exception as e:
            logger.error(f"GRPO alignment failed: {e}")
            raise
    
    def run_evaluation(self):
        """Run model evaluation stage."""
        self._log_stage_start("MODEL EVALUATION")
        
        try:
            evaluator = ModelEvaluator(config_path=self.config_path)
            results = evaluator.run_evaluation()
            
            self._log_stage_complete("MODEL EVALUATION")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def run_full_pipeline(
        self,
        skip_data_prep: bool = False,
        skip_sft: bool = False,
        skip_grpo: bool = False,
        skip_evaluation: bool = False,
        force_reload_clean: bool = False,
        force_regenerate_toxic: bool = False
    ):
        """
        Run the complete pipeline.
        
        Args:
            skip_data_prep: Skip data preparation stage
            skip_sft: Skip SFT training stage
            skip_grpo: Skip GRPO alignment stage
            skip_evaluation: Skip evaluation stage
            force_reload_clean: Force reload clean dataset
            force_regenerate_toxic: Force regenerate toxic dataset
        """
        self.start_time = datetime.now()
        
        logger.info("\n" + "="*100)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100 + "\n")
        
        try:
            # Stage 1: Data Preparation
            if not skip_data_prep:
                self.run_data_preparation(
                    force_reload_clean=force_reload_clean,
                    force_regenerate_toxic=force_regenerate_toxic
                )
            else:
                logger.info("Skipping data preparation stage")
            
            # Stage 2: SFT Training
            if not skip_sft:
                self.run_sft_training()
            else:
                logger.info("Skipping SFT training stage")
            
            # Stage 3: GRPO Alignment
            if not skip_grpo:
                self.run_grpo_alignment()
            else:
                logger.info("Skipping GRPO alignment stage")
            
            # Stage 4: Evaluation
            if not skip_evaluation:
                self.run_evaluation()
            else:
                logger.info("Skipping evaluation stage")
            
            # Pipeline completed
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            logger.info("\n" + "="*100)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Stages completed: {', '.join(self.stages_completed)}")
            logger.info("="*100 + "\n")
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            logger.error("\n" + "="*100)
            logger.error("PIPELINE FAILED")
            logger.error(f"Error: {e}")
            logger.error(f"Failed after: {duration}")
            logger.error(f"Stages completed before failure: {', '.join(self.stages_completed)}")
            logger.error("="*100 + "\n")
            raise
    
    def run_custom_stages(self, stages: list):
        """
        Run specific stages of the pipeline.
        
        Args:
            stages: List of stage names to run
                    Options: 'data', 'sft', 'grpo', 'eval'
        """
        self.start_time = datetime.now()
        
        logger.info("\n" + "="*100)
        logger.info("STARTING CUSTOM PIPELINE")
        logger.info(f"Stages to run: {', '.join(stages)}")
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100 + "\n")
        
        stage_map = {
            'data': self.run_data_preparation,
            'sft': self.run_sft_training,
            'grpo': self.run_grpo_alignment,
            'eval': self.run_evaluation
        }
        
        try:
            for stage in stages:
                if stage not in stage_map:
                    logger.warning(f"Unknown stage: {stage}. Skipping.")
                    continue
                
                stage_map[stage]()
            
            # Pipeline completed
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            logger.info("\n" + "="*100)
            logger.info("CUSTOM PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Stages completed: {', '.join(self.stages_completed)}")
            logger.info("="*100 + "\n")
            
        except Exception as e:
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            logger.error("\n" + "="*100)
            logger.error("CUSTOM PIPELINE FAILED")
            logger.error(f"Error: {e}")
            logger.error(f"Failed after: {duration}")
            logger.error(f"Stages completed before failure: {', '.join(self.stages_completed)}")
            logger.error("="*100 + "\n")
            raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='LLM Summarization and Detoxification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/entry_point.py --full
  
  # Run specific stages
  python scripts/entry_point.py --stages data sft eval
  
  # Skip certain stages
  python scripts/entry_point.py --full --skip-grpo
  
  # Force regenerate data
  python scripts/entry_point.py --stages data --force-regenerate-toxic
  
  # Use custom config
  python scripts/entry_point.py --config my_config.yaml --full
        """
    )
    
    # Config
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Pipeline modes
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline (data prep -> SFT -> GRPO -> evaluation)'
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['data', 'sft', 'grpo', 'eval'],
        help='Run specific stages only'
    )
    
    # Stage skipping (for --full mode)
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation stage'
    )
    
    parser.add_argument(
        '--skip-sft',
        action='store_true',
        help='Skip SFT training stage'
    )
    
    parser.add_argument(
        '--skip-grpo',
        action='store_true',
        help='Skip GRPO alignment stage'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation stage'
    )
    
    # Data preparation options
    parser.add_argument(
        '--force-reload-clean',
        action='store_true',
        help='Force reload clean dataset from HuggingFace'
    )
    
    parser.add_argument(
        '--force-regenerate-toxic',
        action='store_true',
        help='Force regenerate toxic dataset'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.full and not args.stages:
        parser.error("Must specify either --full or --stages")
    
    if args.full and args.stages:
        parser.error("Cannot use both --full and --stages")
    
    # Initialize pipeline
    try:
        pipeline = Pipeline(config_path=args.config)
        
        if args.full:
            # Run full pipeline with skip options
            pipeline.run_full_pipeline(
                skip_data_prep=args.skip_data_prep,
                skip_sft=args.skip_sft,
                skip_grpo=args.skip_grpo,
                skip_evaluation=args.skip_evaluation,
                force_reload_clean=args.force_reload_clean,
                force_regenerate_toxic=args.force_regenerate_toxic
            )
        else:
            # Run custom stages
            pipeline.run_custom_stages(args.stages)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()