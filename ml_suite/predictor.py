"""
Predictor module for the Gmail Unsubscriber AI suite.

This module is responsible for:
- Loading the fine-tuned transformer model (base or personalized)
- Creating an efficient inference pipeline
- Providing real-time predictions for email classification
- Handling model loading errors gracefully
- Supporting personalized model selection based on user ID

The predictor provides a simple interface for the main application to classify
emails as "important" or "unsubscribable" with confidence scores, with the
ability to use personalized models when available.
"""

import os
import time
import logging
import torch
from typing import Dict, List, Optional, Any, Union
import traceback

# Hugging Face imports
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig
)

# Local imports
from . import config
from . import utils


# Global variables to hold the loaded models and tokenizers
# Main base model pipeline
base_classification_pipeline = None
base_model_load_status = "Not Loaded"
base_model_load_error = None
base_model_last_load_attempt = 0

# Dictionary to store personalized model pipelines for different users
personalized_pipelines = {}
personalized_load_status = {}

# Configuration
load_cooldown_seconds = 60  # Wait at least this long between load attempts


def is_predictor_ready() -> bool:
    """
    Check if the base predictor is ready for use.
    
    Returns:
        True if the base predictor is ready, False otherwise
    """
    return base_model_load_status == "Ready" and base_classification_pipeline is not None


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information
    """
    # Try to load model info from the model directory
    model_info_path = os.path.join(config.FINE_TUNED_MODEL_DIR, "model_info.txt")
    model_info = {
        "model_type": "base",
        "model_path": config.FINE_TUNED_MODEL_DIR,
        "trained_date": "unknown"
    }
    
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        value = value.strip()
                        model_info[key] = value
        except:
            pass
    
    return model_info


def initialize_predictor(app_logger: logging.Logger) -> bool:
    """
    Initialize the base predictor by loading the fine-tuned model.
    
    This function:
    1. Loads the tokenizer and model from the fine-tuned directory
    2. Creates a TextClassificationPipeline for efficient inference
    3. Sets global variables for status tracking
    
    Args:
        app_logger: Application logger for status and error reporting
        
    Returns:
        True if initialization successful, False otherwise
    """
    global base_classification_pipeline, base_model_load_status, base_model_load_error, base_model_last_load_attempt
    
    # Reset error tracking
    base_model_load_error = None
    
    # Check if we attempted to load recently (to prevent repeated failures)
    current_time = time.time()
    if (current_time - base_model_last_load_attempt) < load_cooldown_seconds and base_model_load_status == "Failed":
        app_logger.warning(
            f"Not attempting to reload model - cooling down after recent failure. "
            f"Will retry after {load_cooldown_seconds - (current_time - base_model_last_load_attempt):.0f} seconds."
        )
        return False
    
    base_model_last_load_attempt = current_time
    
    try:
        # Update status
        base_model_load_status = "Loading"
        app_logger.info(f"Initializing base AI predictor from {config.FINE_TUNED_MODEL_DIR}")
        
        # Check if model directory exists and contains necessary files
        if not os.path.exists(config.FINE_TUNED_MODEL_DIR):
            raise FileNotFoundError(f"Model directory not found: {config.FINE_TUNED_MODEL_DIR}")
        
        # Load the tokenizer
        app_logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            config.FINE_TUNED_MODEL_DIR,
            local_files_only=True  # Ensure we load locally without trying to download
        )
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            config.FINE_TUNED_MODEL_DIR,
            local_files_only=True
        )
        
        # Check if the model has the expected number of labels
        if model_config.num_labels != config.NUM_LABELS:
            app_logger.warning(
                f"Model has {model_config.num_labels} labels, "
                f"but config specifies {config.NUM_LABELS} labels. "
                f"This may cause issues with predictions."
            )
        
        # Load the model
        app_logger.info("Loading model")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.FINE_TUNED_MODEL_DIR,
            local_files_only=True  # Ensure we load locally without trying to download
        )
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app_logger.info(f"Using device: {device}")
        
        # Move model to the device
        model.to(device)
        
        # Create classification pipeline
        app_logger.info("Creating inference pipeline")
        
        # Set device index properly
        if device.type == "cuda":
            device_index = 0  # Use first GPU
            app_logger.info(f"Pipeline will use GPU device index: {device_index}")
        else:
            device_index = -1  # CPU
            app_logger.info("Pipeline will use CPU")
            
        base_classification_pipeline = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device_index,
            top_k=None,  # Get probabilities for all classes (replaces deprecated return_all_scores)
            function_to_apply="sigmoid"  # Apply sigmoid to logits for probability interpretation
        )
        
        # Update status
        base_model_load_status = "Ready"
        app_logger.info("Base AI predictor initialized successfully")
        
        return True
    
    except Exception as e:
        # Handle initialization failure
        base_model_load_status = "Failed"
        base_model_load_error = str(e)
        error_traceback = traceback.format_exc()
        
        app_logger.error(f"Error initializing base AI predictor: {str(e)}")
        app_logger.debug(f"Traceback:\n{error_traceback}")
        
        # Cleanup any partial loading
        base_classification_pipeline = None
        
        return False


def initialize_personalized_predictor(user_id: str, app_logger: logging.Logger) -> bool:
    """
    Initialize a personalized predictor for a specific user.
    
    Args:
        user_id: The user ID for which to load the personalized model
        app_logger: Application logger for status and error reporting
        
    Returns:
        True if initialization successful, False otherwise
    """
    global personalized_pipelines, personalized_load_status
    
    try:
        # Check if the user has a personalized model
        model_dir = config.PERSONALIZED_MODEL_DIR_TEMPLATE.format(user_id=user_id)
        if not os.path.exists(model_dir) or not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            app_logger.info(f"No personalized model found for user {user_id}")
            personalized_load_status[user_id] = "Not Available"
            return False
        
        # Update status
        personalized_load_status[user_id] = "Loading"
        app_logger.info(f"Initializing personalized AI predictor for user {user_id} from {model_dir}")
        
        # Load the tokenizer
        app_logger.info(f"Loading personalized tokenizer for user {user_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        # Load the model
        app_logger.info(f"Loading personalized model for user {user_id}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to the device
        model.to(device)
        
        # Create classification pipeline
        app_logger.info(f"Creating personalized inference pipeline for user {user_id}")
        
        # Set device index properly
        if device.type == "cuda":
            device_index = 0  # Use first GPU
        else:
            device_index = -1  # CPU
            
        personalized_pipelines[user_id] = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device_index,
            top_k=None,  # Get probabilities for all classes (replaces deprecated return_all_scores)
            function_to_apply="sigmoid"
        )
        
        # Update status
        personalized_load_status[user_id] = "Ready"
        app_logger.info(f"Personalized AI predictor for user {user_id} initialized successfully")
        
        return True
    
    except Exception as e:
        # Handle initialization failure
        personalized_load_status[user_id] = "Failed"
        error_traceback = traceback.format_exc()
        
        app_logger.error(f"Error initializing personalized AI predictor for user {user_id}: {str(e)}")
        app_logger.debug(f"Traceback:\n{error_traceback}")
        
        # Cleanup any partial loading
        if user_id in personalized_pipelines:
            del personalized_pipelines[user_id]
        
        return False


def get_model_status(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the current status of the model.
    
    Args:
        user_id: Optional user ID to get status for personalized model
        
    Returns:
        Dictionary with status information
    """
    if user_id is None:
        # Return base model status
        return {
            "status": base_model_load_status,
            "error": base_model_load_error,
            "last_load_attempt": base_model_last_load_attempt,
            "model_dir": config.FINE_TUNED_MODEL_DIR,
            "is_ready": base_model_load_status == "Ready" and base_classification_pipeline is not None,
            "is_personalized": False
        }
    else:
        # Check if personalized model is available
        if user_id not in personalized_load_status:
            personalized_load_status[user_id] = "Not Loaded"
        
        model_dir = config.PERSONALIZED_MODEL_DIR_TEMPLATE.format(user_id=user_id)
        has_personalized = (
            os.path.exists(model_dir) and 
            os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
        )
        
        return {
            "status": personalized_load_status.get(user_id, "Not Loaded"),
            "model_dir": model_dir if has_personalized else None,
            "is_ready": personalized_load_status.get(user_id) == "Ready" and user_id in personalized_pipelines,
            "is_personalized": True,
            "has_personalized_model": has_personalized
        }


def get_ai_prediction_for_email(
    email_text_content: str, 
    user_id: Optional[str] = None,
    app_logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """
    Get AI prediction for an email, optionally using a personalized model.
    
    This function:
    1. Checks if the requested model is loaded and ready
    2. Passes the email text to the appropriate classification pipeline
    3. Processes and returns the prediction results
    
    Args:
        email_text_content: The combined email text (subject + body)
        user_id: Optional user ID to use personalized model if available
        app_logger: Optional logger for error reporting
        
    Returns:
        Dictionary with prediction results (label, confidence, etc.) or None if prediction fails
    """
    global base_classification_pipeline, personalized_pipelines
    
    # Determine which pipeline to use
    pipeline = None
    using_personalized = False
    
    if user_id is not None and user_id in personalized_pipelines:
        # Try to use personalized model first
        if personalized_load_status.get(user_id) == "Ready":
            pipeline = personalized_pipelines[user_id]
            using_personalized = True
    
    # Fall back to base model if personalized isn't available
    if pipeline is None:
        if base_model_load_status != "Ready" or base_classification_pipeline is None:
            return None
        pipeline = base_classification_pipeline
    
    try:
        # Clean and normalize the input text
        cleaned_text = utils.clean_text_for_model(
            email_text_content, 
            max_length=config.EMAIL_SNIPPET_LENGTH_FOR_MODEL
        )
        
        # Skip prediction for extremely short text
        if len(cleaned_text) < config.MIN_TEXT_LENGTH_FOR_TRAINING:
            return {
                "label": "INDETERMINATE",
                "confidence": 0.0,
                "predicted_id": None,
                "error": "Text too short for reliable prediction",
                "using_personalized_model": using_personalized
            }
        
        # Get prediction
        predictions = pipeline(cleaned_text)
        
        # Process prediction results
        # The pipeline returns a list with one dictionary per input text
        # Each dictionary has a 'label' and 'score' for each possible class
        prediction_scores = {}
        for pred in predictions[0]:  # Get the first (and only) prediction
            label_id = int(pred['label'].split('_')[-1])
            label_name = config.ID_TO_LABEL_MAP.get(label_id)
            prediction_scores[label_name] = pred['score']
        
        # Find the highest scoring label
        max_label = max(prediction_scores, key=prediction_scores.get)
        max_score = prediction_scores[max_label]
        predicted_id = config.LABEL_TO_ID_MAP.get(max_label)
        
        # Apply confidence calibration to prevent overconfidence
        # Temperature scaling to soften extreme predictions
        temperature = 1.5  # Higher = less confident
        calibrated_score = max_score ** (1 / temperature)
        
        # Log prediction details for debugging
        if app_logger:
            app_logger.debug(f"AI Prediction: {max_label} (raw: {max_score:.3f}, calibrated: {calibrated_score:.3f})")
            app_logger.debug(f"All scores: {prediction_scores}")
            app_logger.debug(f"Email snippet: {cleaned_text[:100]}...")
        
        # Return the prediction
        return {
            "label": max_label,
            "confidence": calibrated_score,
            "raw_confidence": max_score,
            "predicted_id": predicted_id,
            "all_scores": prediction_scores,
            "using_personalized_model": using_personalized
        }
    
    except Exception as e:
        # Log the error details but don't expose them in the response
        if app_logger:
            app_logger.error(f"Error during AI prediction: {str(e)}")
        else:
            print(f"Error during AI prediction: {str(e)}")
        
        return {
            "label": "ERROR",
            "confidence": 0.0,
            "predicted_id": None,
            "error": "Prediction error occurred",
            "using_personalized_model": using_personalized
        }