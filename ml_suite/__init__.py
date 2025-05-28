"""
Machine Learning module for Gmail Unsubscriber application.

This module provides AI-powered classification of emails as 'unsubscribable' or 'important'.
It includes components for data preparation, model training, and prediction, with user
control over the AI lifecycle through an in-app AI panel.

The design emphasizes:
- User control over AI data preparation and training
- Seamless integration with the existing application
- Transparency in AI operations and decisions
- Graceful degradation when AI components are unavailable
"""

# Import configuration to ensure directories are created
from . import config

# Version information
__version__ = "0.1.0"