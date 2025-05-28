"""
Model trainer module for the Gmail Unsubscriber AI suite.

This module is responsible for:
- Loading prepared email data
- Splitting data into training and evaluation sets
- Loading pre-trained transformer model
- Fine-tuning the model on the email dataset
- Evaluating model performance
- Saving the fine-tuned model for prediction

The trained model is optimized for classifying emails as "important" or "unsubscribable".
"""

import os
import time
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Hugging Face imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
    PreTrainedTokenizer,
    PreTrainedModel
)
from datasets import Dataset
import torch.nn as nn

# Local imports
from . import config
from . import utils
from .task_utils import AiTaskLogger


def compute_metrics_for_classification(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model.
    
    Args:
        eval_pred: Tuple of predictions and labels from the trainer
        
    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1, etc.
    """
    predictions, labels = eval_pred
    
    # For classification, take the argmax to get predicted classes
    preds = np.argmax(predictions, axis=1)
    
    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    # Calculate metrics specific to the "unsubscribable" class
    # Get label positions: 0 for important, 1 for unsubscribable
    unsub_class_idx = config.LABEL_UNSUBSCRIBABLE_ID
    
    # Calculate class-specific precision, recall, F1
    precision_unsub, recall_unsub, f1_unsub, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[unsub_class_idx]
    )
    
    # Compute ROC AUC if possible (requires probability scores)
    try:
        # Use the probability score for the unsubscribable class
        probs_unsub = predictions[:, unsub_class_idx]
        
        # Convert labels to binary (1 for unsubscribable, 0 for others)
        binary_labels = (labels == unsub_class_idx).astype(int)
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(binary_labels, probs_unsub)
    except (ValueError, IndexError):
        roc_auc = 0.0
    
    # Return all metrics in a dictionary
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_unsub': float(precision_unsub[0]) if len(precision_unsub) > 0 else 0.0,
        'recall_unsub': float(recall_unsub[0]) if len(recall_unsub) > 0 else 0.0,
        'f1_unsub': float(f1_unsub[0]) if len(f1_unsub) > 0 else 0.0,
        'roc_auc': roc_auc
    }
    
    return metrics


class WeightedLossTrainer(Trainer):
    """Custom trainer that supports class weights for imbalanced datasets."""
    
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with class weights and optional label smoothing."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Create loss function with class weights
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(
    examples: Dict[str, List], 
    tokenizer: PreTrainedTokenizer, 
    max_length: int
) -> Dict[str, List]:
    """
    Tokenize a batch of examples.
    
    Args:
        examples: Dictionary of example lists
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of tokenized examples
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def prepare_datasets_for_training(
    data_file: str, 
    test_size: float, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int,
    task_logger: AiTaskLogger,
    compute_weights: bool = True
) -> Tuple[Dataset, Dataset, Optional[np.ndarray]]:
    """
    Load and prepare datasets for training.
    
    Args:
        data_file: Path to the prepared data file (CSV)
        test_size: Proportion of data to use for evaluation
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        task_logger: Logger for tracking progress
        compute_weights: Whether to compute class weights for imbalanced data
        
    Returns:
        Tuple of (train_dataset, eval_dataset, class_weights)
    """
    task_logger.info(f"Loading data from {data_file}")
    
    try:
        # Load the dataset
        df = pd.read_csv(data_file)
        task_logger.info(f"Loaded {len(df)} examples from {data_file}")
        
        # Check for required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            task_logger.error("Data file is missing required columns 'text' and/or 'label'")
            raise ValueError("Data file has wrong format")
        
        # Split into training and evaluation sets
        train_df, eval_df = train_test_split(
            df, test_size=test_size, stratify=df['label'], random_state=42
        )
        
        task_logger.info(f"Split into {len(train_df)} training examples and {len(eval_df)} evaluation examples")
        
        # Compute class weights for imbalanced data
        class_weights = None
        if compute_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=train_df['label'].values
            )
            task_logger.info(f"Computed class weights: {class_weights}")
        
        # Create HF datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Create a tokenization function that uses our tokenizer
        def tokenize_function(examples):
            return tokenize_dataset(examples, tokenizer, max_length)
        
        # Apply tokenization
        task_logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True,
            desc="Tokenizing training dataset"
        )
        eval_dataset = eval_dataset.map(
            tokenize_function, 
            batched=True,
            desc="Tokenizing evaluation dataset"
        )
        
        # Set format for PyTorch
        train_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'label']
        )
        eval_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'label']
        )
        
        task_logger.info("Datasets prepared successfully")
        
        return train_dataset, eval_dataset, class_weights
    
    except Exception as e:
        task_logger.error(f"Error preparing datasets: {str(e)}", e)
        raise


def train_unsubscriber_model(task_logger: AiTaskLogger) -> Dict[str, Any]:
    """
    Train the unsubscriber model on the prepared dataset.
    
    This function:
    1. Loads prepared data
    2. Initializes a pre-trained transformer model
    3. Fine-tunes it on the email classification task
    4. Evaluates performance
    5. Saves the model for later use
    
    Args:
        task_logger: Logger for tracking task status
        
    Returns:
        Dictionary with training results and metrics
    """
    # Start timing
    start_time = time.time()
    
    # Start the task
    task_logger.start_task("Starting AI model training")
    
    try:
        # Check if prepared data exists
        if not os.path.exists(config.PREPARED_DATA_FILE):
            task_logger.error(f"Prepared data file not found at {config.PREPARED_DATA_FILE}.")
            task_logger.fail_task("Training failed: No prepared data available. Please run data preparation first.")
            return {"success": False, "error": "No prepared data available"}
        
        # 1. Load and initialize the tokenizer
        task_logger.info(f"Loading tokenizer for model: {config.PRE_TRAINED_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        
        # 2. Prepare datasets
        task_logger.update_progress(0.1, "Preparing datasets")
        train_dataset, eval_dataset, class_weights = prepare_datasets_for_training(
            config.PREPARED_DATA_FILE,
            config.TEST_SPLIT_SIZE,
            tokenizer,
            config.MAX_SEQ_LENGTH,
            task_logger,
            compute_weights=True
        )
        
        # 3. Initialize model
        task_logger.update_progress(0.2, f"Initializing model: {config.PRE_TRAINED_MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.PRE_TRAINED_MODEL_NAME,
            num_labels=config.NUM_LABELS
        )
        
        # Check device availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fp16_enabled = config.FP16_TRAINING and torch.cuda.is_available()
        
        # Force GPU check and provide detailed info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            task_logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
            task_logger.info(f"CUDA version: {torch.version.cuda}")
            
            # Set CUDA device explicitly
            torch.cuda.set_device(0)
            
            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            task_logger.warning("No GPU detected! Training will be slow on CPU.")
            task_logger.warning("Make sure you have the CUDA version of PyTorch installed.")
            task_logger.warning("To install PyTorch with CUDA support, run:")
            task_logger.warning("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        task_logger.info(f"Training on device: {device}")
        if fp16_enabled:
            task_logger.info("FP16 mixed precision training enabled")
        
        # Move model to device
        model.to(device)
        
        # 4. Set up training arguments
        task_logger.update_progress(0.3, "Setting up training configuration")
        
        # Calculate number of steps
        num_train_examples = len(train_dataset)
        num_train_steps = (num_train_examples // config.TRAIN_BATCH_SIZE) * config.NUM_TRAIN_EPOCHS
        warmup_steps = int(num_train_steps * config.WARMUP_STEPS_RATIO)
        
        # Convert evaluation and save strategies to enum values
        eval_strategy = (IntervalStrategy.EPOCH 
                         if config.EVALUATION_STRATEGY == "epoch" 
                         else IntervalStrategy.STEPS)
        save_strategy = (IntervalStrategy.EPOCH 
                         if config.SAVE_STRATEGY == "epoch" 
                         else IntervalStrategy.STEPS)
        
        # Create output directory if it doesn't exist
        os.makedirs(config.FINE_TUNED_MODEL_DIR, exist_ok=True)
        
        # Define training arguments (with compatibility fixes for older transformers versions)
        training_args_dict = {
            "output_dir": config.FINE_TUNED_MODEL_DIR,
            "per_device_train_batch_size": config.TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": config.EVAL_BATCH_SIZE,
            "num_train_epochs": config.NUM_TRAIN_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "warmup_steps": warmup_steps,
            "logging_steps": 50,
            "save_total_limit": 2,
            "load_best_model_at_end": config.LOAD_BEST_MODEL_AT_END,
            "metric_for_best_model": config.METRIC_FOR_BEST_MODEL,
            "greater_is_better": True,
            "report_to": "none",
            "disable_tqdm": False,
            "no_cuda": False,  # Explicitly enable CUDA
            "use_cpu": False,  # Explicitly disable CPU-only mode
        }
        
        # Add evaluation and save strategies
        try:
            training_args_dict["evaluation_strategy"] = eval_strategy
            training_args_dict["save_strategy"] = save_strategy
        except:
            # Fallback for older versions
            training_args_dict["evaluation_strategy"] = "epoch"
            training_args_dict["save_strategy"] = "epoch"
        
        # Add logging dir if supported
        try:
            training_args_dict["logging_dir"] = os.path.join(config.FINE_TUNED_MODEL_DIR, "logs")
        except:
            pass
        
        # Add FP16 if supported and available
        if fp16_enabled:
            try:
                training_args_dict["fp16"] = True
            except:
                task_logger.warning("FP16 training not supported on this setup")
        
        # Add dataloader workers if supported
        try:
            training_args_dict["dataloader_num_workers"] = 2
        except:
            pass
        
        # Add gradient accumulation if configured
        if hasattr(config, 'GRADIENT_ACCUMULATION_STEPS'):
            training_args_dict["gradient_accumulation_steps"] = config.GRADIENT_ACCUMULATION_STEPS
        
        # Add label smoothing if configured
        if hasattr(config, 'LABEL_SMOOTHING_FACTOR'):
            training_args_dict["label_smoothing_factor"] = config.LABEL_SMOOTHING_FACTOR
        
        training_args = TrainingArguments(**training_args_dict)
        
        # 5. Initialize the trainer with class weights
        task_logger.update_progress(0.4, "Initializing trainer with class balancing")
        
        # Use weighted loss trainer if we have class weights
        if class_weights is not None:
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_for_classification,
                class_weights=class_weights,
                label_smoothing=config.LABEL_SMOOTHING_FACTOR if hasattr(config, 'LABEL_SMOOTHING_FACTOR') else 0.0,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                        early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
                    )
                ]
            )
            task_logger.info("Using weighted loss function for class imbalance")
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_for_classification,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                        early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD
                    )
                ]
            )
        
        # 6. Train the model
        task_logger.update_progress(0.5, "Starting model training")
        trainer.train()
        
        # 7. Evaluate the model
        task_logger.update_progress(0.9, "Evaluating model")
        eval_results = trainer.evaluate()
        
        # Format evaluations nicely for logging
        metrics_str = "\n".join([f"  {k}: {v:.4f}" for k, v in eval_results.items()])
        task_logger.info(f"Evaluation results:\n{metrics_str}")
        
        # 8. Save the final model
        task_logger.update_progress(0.95, "Saving fine-tuned model")
        trainer.save_model(config.FINE_TUNED_MODEL_DIR)
        tokenizer.save_pretrained(config.FINE_TUNED_MODEL_DIR)
        
        # Save a human-readable summary of model info
        with open(os.path.join(config.FINE_TUNED_MODEL_DIR, "model_info.txt"), "w") as f:
            f.write(f"Base model: {config.PRE_TRAINED_MODEL_NAME}\n")
            f.write(f"Training completed: {utils.get_current_timestamp()}\n")
            f.write(f"Training examples: {len(train_dataset)}\n")
            f.write(f"Evaluation examples: {len(eval_dataset)}\n")
            f.write(f"Evaluation metrics:\n")
            for k, v in eval_results.items():
                f.write(f"  {k}: {v:.4f}\n")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Complete the task
        result = {
            "success": True,
            "metrics": eval_results,
            "model_dir": config.FINE_TUNED_MODEL_DIR,
            "training_time": time_str,
            "base_model": config.PRE_TRAINED_MODEL_NAME,
            "num_train_examples": len(train_dataset),
            "num_eval_examples": len(eval_dataset)
        }
        
        task_logger.complete_task(
            f"Model training completed successfully in {time_str}", 
            result
        )
        
        return result
    
    except Exception as e:
        task_logger.error("Error during model training", e)
        task_logger.fail_task(f"Model training failed: {str(e)}")
        
        return {
            "success": False,
            "error": str(e)
        }