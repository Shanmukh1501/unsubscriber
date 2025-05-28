"""
Centralized configuration for Gmail Unsubscriber AI Suite.

This module defines all configuration parameters for the ML components, including:
- Directory paths for models, datasets, and task status
- Hugging Face cache configuration
- Model specifications
- Data preparation parameters
- Training hyperparameters
- User data collection and personalization parameters

All directories are automatically created when this module is imported.
"""

import os

# --- Base Path Configuration ---
ML_SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ML_SUITE_DIR)

# --- Cache and Model Storage ---
MODELS_DIR = os.path.join(ML_SUITE_DIR, "models")
BASE_TRANSFORMER_CACHE_DIR = os.path.join(MODELS_DIR, "base_transformer_cache")
# FINE_TUNED_MODEL_DIR = os.path.join(MODELS_DIR, "fine_tuned_unsubscriber")  # Old model
FINE_TUNED_MODEL_DIR = os.path.join(PROJECT_ROOT, "final_optimized_model")  # New trained model

# Set Hugging Face environment variables to use project-local cache
os.environ['HF_HOME'] = BASE_TRANSFORMER_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = BASE_TRANSFORMER_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(BASE_TRANSFORMER_CACHE_DIR, 'datasets')
os.environ['HF_METRICS_CACHE'] = os.path.join(BASE_TRANSFORMER_CACHE_DIR, 'metrics')

# --- Dataset Storage ---
DATASETS_DIR = os.path.join(ML_SUITE_DIR, "datasets")
RAW_DATASETS_DIR = os.path.join(DATASETS_DIR, "raw")
EXTRACTED_DATASETS_DIR = os.path.join(DATASETS_DIR, "extracted")
PROCESSED_DATASETS_DIR = os.path.join(DATASETS_DIR, "processed")
PREPARED_DATA_FILE = os.path.join(PROCESSED_DATASETS_DIR, "unsubscriber_training_data.csv")
DATA_COLUMNS_SCHEMA = ['text', 'label']  # Schema for the training CSV

# --- Task Status Storage ---
TASK_STATUS_DIR = os.path.join(ML_SUITE_DIR, "task_status")
DATA_PREP_STATUS_FILE = os.path.join(TASK_STATUS_DIR, "data_preparation_status.json")
MODEL_TRAIN_STATUS_FILE = os.path.join(TASK_STATUS_DIR, "model_training_status.json")
PERSONALIZED_TRAIN_STATUS_FILE = os.path.join(TASK_STATUS_DIR, "personalized_training_status.json")

# --- User Data Collection and Personalization ---
USER_DATA_DIR = os.path.join(ML_SUITE_DIR, "user_data")
USER_FEEDBACK_DIR = os.path.join(USER_DATA_DIR, "feedback")
USER_MODELS_DIR = os.path.join(USER_DATA_DIR, "models")
USER_DATASETS_DIR = os.path.join(USER_DATA_DIR, "datasets")

# User feedback collection configuration
USER_FEEDBACK_FILE = os.path.join(USER_FEEDBACK_DIR, "user_feedback.csv")
FEEDBACK_COLUMNS_SCHEMA = ['email_id', 'text', 'predicted_label', 'predicted_confidence', 'user_feedback', 'timestamp', 'session_id']

# Personalized model configuration
PERSONALIZED_MODEL_DIR_TEMPLATE = os.path.join(USER_MODELS_DIR, "{user_id}")
PERSONALIZED_MODEL_FILE_TEMPLATE = os.path.join(PERSONALIZED_MODEL_DIR_TEMPLATE, "model.pt")
PERSONALIZED_MODEL_INFO_TEMPLATE = os.path.join(PERSONALIZED_MODEL_DIR_TEMPLATE, "model_info.json")
PERSONALIZED_DATASET_FILE_TEMPLATE = os.path.join(USER_DATASETS_DIR, "{user_id}_training_data.csv")

# Personalization hyperparameters
MIN_FEEDBACK_ENTRIES_FOR_PERSONALIZATION = 10  # Minimum number of user feedback entries required for personalization
PERSONALIZATION_WEIGHT = 0.7  # Weight given to user feedback vs. base model (higher = more personalized)
PERSONALIZATION_EPOCHS = 2  # Number of epochs for fine-tuning a personalized model

# --- Directory Creation (Updated with User Data directories) ---
for dir_path in [MODELS_DIR, BASE_TRANSFORMER_CACHE_DIR, FINE_TUNED_MODEL_DIR,
                 RAW_DATASETS_DIR, EXTRACTED_DATASETS_DIR, PROCESSED_DATASETS_DIR, TASK_STATUS_DIR,
                 USER_DATA_DIR, USER_FEEDBACK_DIR, USER_MODELS_DIR, USER_DATASETS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Transformer Model Configuration ---
# Choice: DistilBERT offers a good balance of performance and resource efficiency.
# Other candidates: 'bert-base-uncased', 'roberta-base', 'google/electra-small-discriminator'.
# The choice impacts download size, training time, and inference speed.
PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"

# --- Data Preparation Parameters ---
# Define sources for public email data. URLs and types guide the preparator.
PUBLIC_DATASETS_INFO = {
    "spamassassin_easy_ham_2003": {
        "url": "https://spamassassin.apache.org/publiccorpus/20030228_easy_ham.tar.bz2",
        "type": "important_leaning",  # Expected dominant class after heuristic application
        "extract_folder_name": "spamassassin_easy_ham_2003"
    },
    "spamassassin_spam_2003": {
        "url": "https://spamassassin.apache.org/publiccorpus/20030228_spam.tar.bz2",
        "type": "unsubscribable_leaning",
        "extract_folder_name": "spamassassin_spam_2003"
    },
    # Consider adding more diverse datasets like:
    # - Enron (requires significant parsing and ethical review for a suitable subset)
    # - Public mailing list archives (e.g., from Apache Software Foundation, carefully selected for relevance)
}
MIN_TEXT_LENGTH_FOR_TRAINING = 60  # Emails shorter than this (after cleaning) are likely not useful.
MAX_SAMPLES_PER_RAW_DATASET = 7500  # Limits processing time for initial data prep. Can be increased.
EMAIL_SNIPPET_LENGTH_FOR_MODEL = 1024  # Max characters from email body to combine with subject for model input.

# --- Training Hyperparameters & Configuration ---
NUM_LABELS = 2  # Binary classification: Unsubscribable vs. Important
LABEL_IMPORTANT_ID = 0
LABEL_UNSUBSCRIBABLE_ID = 1
ID_TO_LABEL_MAP = {LABEL_IMPORTANT_ID: "IMPORTANT", LABEL_UNSUBSCRIBABLE_ID: "UNSUBSCRIBABLE"}
LABEL_TO_ID_MAP = {"IMPORTANT": LABEL_IMPORTANT_ID, "UNSUBSCRIBABLE": LABEL_UNSUBSCRIBABLE_ID}

MAX_SEQ_LENGTH = 512        # Max token sequence length for Transformer. Impacts memory and context window.
TRAIN_BATCH_SIZE = 16        # Batch size for training. Reduced for GTX 1650 (4GB VRAM)
EVAL_BATCH_SIZE = 32        # Batch size for evaluation. Reduced for GTX 1650
NUM_TRAIN_EPOCHS = 8        # Number of full passes through the training data (increased for better learning).
LEARNING_RATE = 1e-5        # AdamW optimizer learning rate, slightly reduced for more stable training.
WEIGHT_DECAY = 0.02         # Regularization parameter.
WARMUP_STEPS_RATIO = 0.15    # Ratio of total training steps for learning rate warmup.
TEST_SPLIT_SIZE = 0.2       # Proportion of data for the evaluation set (increased for better validation).

# Hugging Face Trainer Arguments
EVALUATION_STRATEGY = "epoch"  # Evaluate at the end of each epoch.
SAVE_STRATEGY = "epoch"       # Save model checkpoint at the end of each epoch.
LOAD_BEST_MODEL_AT_END = True  # Reload the best model (based on metric_for_best_model) at the end of training.
METRIC_FOR_BEST_MODEL = "f1_unsub"  # Focus on F1 for the "unsubscribable" class.
FP16_TRAINING = True          # Enable mixed-precision training if a CUDA GPU is available and supports it.
EARLY_STOPPING_PATIENCE = 3   # Stop training if metric_for_best_model doesn't improve for this many epochs.
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum change to be considered an improvement.

# --- AI User Preferences (Defaults stored in JS, but can be defined here for reference) ---
DEFAULT_AI_ENABLED_ON_SCAN = True
DEFAULT_AI_CONFIDENCE_THRESHOLD = 0.5  # (50%) - Balanced threshold for optimal precision/recall

# --- API Endpoint Configuration for Backend Integration ---
API_ENDPOINTS = {
    "submit_feedback": "/api/ai/feedback",
    "get_feedback_stats": "/api/ai/feedback/stats",
    "train_personalized": "/api/ai/train_personalized",
    "reset_user_data": "/api/ai/user_data/reset",
    "export_user_data": "/api/ai/user_data/export",
    "import_user_data": "/api/ai/user_data/import"
}
# --- Advanced Transformer Configuration (2024 Research) ---
# Based on 2024 research showing RoBERTa and DistilBERT achieve 99%+ accuracy
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"  # Optimal balance of speed and accuracy
USE_MIXED_PRECISION = True  # FP16 training for efficiency
GRADIENT_ACCUMULATION_STEPS = 4  # Increased for GTX 1650 to simulate larger batch size
MAX_GRAD_NORM = 1.0  # Gradient clipping for stability
LABEL_SMOOTHING_FACTOR = 0.1  # Reduce overconfidence
SAVE_TOTAL_LIMIT = 3  # Keep only best 3 checkpoints
LOGGING_STEPS = 50  # Frequent logging for monitoring
EVAL_STEPS = 100  # Regular evaluation during training
DATALOADER_NUM_WORKERS = 2  # Reduced for GTX 1650 to avoid memory issues
