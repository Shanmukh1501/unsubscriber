"""
Advanced predictor with support for multiple models and ensemble predictions
"""

import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPredictor:
    """Advanced predictor with ensemble support and confidence calibration"""
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize with multiple models for ensemble prediction
        
        Args:
            model_paths: List of paths to model directories
            weights: Optional weights for each model (must sum to 1.0)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.tokenizers = []
        self.pipelines = []
        
        # Load all models
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"Loading model from {path}")
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                model = model.to(self.device)
                model.eval()
                
                # Create pipeline
                pipeline = TextClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None,
                    function_to_apply="sigmoid"
                )
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.pipelines.append(pipeline)
            else:
                logger.warning(f"Model path not found: {path}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
        
        # Set weights
        if weights:
            assert len(weights) == len(self.models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights
        else:
            # Equal weights by default
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(f"Initialized with {len(self.models)} models")
    
    def predict(self, text: str, return_all_scores: bool = False) -> Dict:
        """
        Make ensemble prediction
        
        Args:
            text: Email text to classify
            return_all_scores: Whether to return individual model scores
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        text = self._preprocess_email(text)
        
        # Get predictions from all models
        all_predictions = []
        for pipeline in self.pipelines:
            try:
                result = pipeline(text)
                all_predictions.append(result)
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                continue
        
        if not all_predictions:
            return {
                "label": "IMPORTANT",
                "score": 0.5,
                "confidence": "low",
                "error": "Prediction failed"
            }
        
        # Aggregate predictions
        ensemble_scores = self._aggregate_predictions(all_predictions)
        
        # Determine final prediction
        unsub_score = ensemble_scores.get("UNSUBSCRIBABLE", 0.5)
        important_score = ensemble_scores.get("IMPORTANT", 0.5)
        
        # Apply confidence calibration
        calibrated_unsub = self._calibrate_confidence(unsub_score)
        
        # Determine label
        if calibrated_unsub > 0.75:  # High confidence threshold
            label = "UNSUBSCRIBABLE"
            score = calibrated_unsub
        else:
            label = "IMPORTANT"
            score = important_score
        
        # Confidence level
        if score > 0.9:
            confidence = "high"
        elif score > 0.7:
            confidence = "medium"
        else:
            confidence = "low"
        
        result = {
            "label": label,
            "score": float(score),
            "confidence": confidence,
            "raw_scores": {
                "UNSUBSCRIBABLE": float(unsub_score),
                "IMPORTANT": float(important_score)
            }
        }
        
        if return_all_scores:
            result["model_predictions"] = all_predictions
        
        return result
    
    def _preprocess_email(self, text: str) -> str:
        """Advanced email preprocessing"""
        # Handle subject extraction
        if "Subject:" in text:
            parts = text.split("Subject:", 1)
            if len(parts) > 1:
                subject = parts[1].split("\n")[0].strip()
                body = parts[1][len(subject):].strip()
                # Emphasize subject
                text = f"Email Subject: {subject}. Email Body: {body}"
        
        # Clean text
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = " ".join(text.split())
        
        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def _aggregate_predictions(self, predictions: List) -> Dict[str, float]:
        """Aggregate predictions from multiple models using weighted voting"""
        aggregated = {"UNSUBSCRIBABLE": 0.0, "IMPORTANT": 0.0}
        
        for i, pred_list in enumerate(predictions):
            weight = self.weights[i]
            
            # Handle different prediction formats
            if isinstance(pred_list, list) and pred_list:
                for pred in pred_list:
                    label = pred.get("label", "").upper()
                    score = pred.get("score", 0.5)
                    
                    if label in aggregated:
                        aggregated[label] += score * weight
        
        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            for key in aggregated:
                aggregated[key] /= total
        
        return aggregated
    
    def _calibrate_confidence(self, score: float, temperature: float = 1.2) -> float:
        """Apply temperature scaling for confidence calibration"""
        # Convert to logit
        epsilon = 1e-7
        score = np.clip(score, epsilon, 1 - epsilon)
        logit = np.log(score / (1 - score))
        
        # Apply temperature scaling
        calibrated_logit = logit / temperature
        
        # Convert back to probability
        calibrated_score = 1 / (1 + np.exp(-calibrated_logit))
        
        return float(calibrated_score)
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict multiple emails efficiently"""
        results = []
        
        # Process in batches for efficiency
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.predict(text) for text in batch]
            results.extend(batch_results)
        
        return results
    
    def get_feature_importance(self, text: str) -> Dict:
        """Get feature importance for explainability"""
        # This is a simplified version - in production, use SHAP or LIME
        important_keywords = [
            "unsubscribe", "opt out", "preferences", "newsletter",
            "promotional", "marketing", "deal", "offer", "sale"
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in important_keywords if kw in text_lower]
        
        return {
            "important_features": found_keywords,
            "feature_count": len(found_keywords)
        }


def create_advanced_predictor():
    """Factory function to create predictor with best available models"""
    model_paths = []
    
    # Check for advanced model first
    if os.path.exists("./advanced_unsubscriber_model"):
        model_paths.append("./advanced_unsubscriber_model")
    
    # Check for optimized model
    if os.path.exists("./optimized_model"):
        model_paths.append("./optimized_model")
    
    # Fallback to original model
    if os.path.exists("./ml_suite/models/fine_tuned_unsubscriber"):
        model_paths.append("./ml_suite/models/fine_tuned_unsubscriber")
    
    if not model_paths:
        raise ValueError("No trained models found")
    
    # Use ensemble if multiple models available
    if len(model_paths) > 1:
        logger.info(f"Creating ensemble predictor with {len(model_paths)} models")
        # Give higher weight to advanced model
        weights = [0.6, 0.4] if len(model_paths) == 2 else None
        return AdvancedPredictor(model_paths, weights)
    else:
        logger.info(f"Creating single model predictor")
        return AdvancedPredictor(model_paths)


# Example usage
if __name__ == "__main__":
    # Test the predictor
    predictor = create_advanced_predictor()
    
    test_emails = [
        "Subject: 50% OFF Everything! Limited time offer. Click here to shop now. Unsubscribe from promotional emails.",
        "Subject: Security Alert: New login detected. We noticed a login from a new device. If this wasn't you, secure your account.",
        "Subject: Your monthly newsletter is here! Check out our latest articles and tips. Manage your email preferences.",
    ]
    
    for email in test_emails:
        result = predictor.predict(email, return_all_scores=True)
        print(f"\nEmail: {email[:100]}...")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']} ({result['score']:.2%})")
        print(f"Raw scores: {result['raw_scores']}")