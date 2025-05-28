"""
Shared utilities for the ML suite.

This module provides shared functions used across the ML suite components:
- Email text analysis using heuristics (adapted from app.py)
- Text cleaning and normalization
- Timestamp and logging utilities
- HTML processing for email content extraction

These utilities ensure consistent processing across different components of the ML suite.
"""

import re
import os
import urllib.parse
import datetime
import html
from typing import Dict, List, Tuple, Optional, Any, Union
from bs4 import BeautifulSoup


# --- Email Heuristic Analysis ---

# Keywords that suggest an email is marketing/promotional/unsubscribable
UNSUBSCRIBE_KEYWORDS_FOR_AI_HEURISTICS = [
    'unsubscribe', 'opt-out', 'opt out', 'stop receiving', 'manage preferences', 
    'email preferences', 'subscription', 'marketing', 'newsletter', 'promotional',
    'offer', 'sale', 'discount', 'deal', 'coupon', 'promo code', 'promotion',
    'limited time', 'subscribe', 'update preferences', 'mailing list',
    'no longer wish to receive', 'manage subscriptions', 'manage your subscriptions'
]

# Keywords that suggest promotional content
PROMO_KEYWORDS_FOR_AI_HEURISTICS = [
    'limited time', 'exclusive', 'offer', 'sale', 'discount', 'deal', 'coupon',
    'promo code', 'promotion', 'savings', 'special offer', 'limited offer',
    'buy now', 'shop now', 'order now', 'click here', 'purchase', 'buy',
    'free shipping', 'free trial', 'new arrival', 'new product', 'flash sale'
]

# Common formatting patterns in promotional emails
FORMATTING_PATTERNS_FOR_AI_HEURISTICS = [
    r'\*+\s*[A-Z]+\s*\*+',  # ***TEXT***
    r'\*\*[^*]+\*\*',       # **TEXT**
    r'!{2,}',               # Multiple exclamation marks
    r'\$\d+(\.\d{2})?(\s+off|\s+discount|%\s+off)',  # Price patterns
    r'\d+%\s+off',          # Percentage discounts
    r'SAVE\s+\d+%',         # SAVE XX%
    r'SAVE\s+\$\d+',        # SAVE $XX
    r'HURRY',               # Urgency words
    r'LIMITED TIME',
    r'LAST CHANCE',
    r'ENDING SOON'
]


def analyze_email_heuristics_for_ai(subject_text: str, snippet_text: str, list_unsubscribe_header: Optional[str] = None) -> Dict[str, bool]:
    """
    Analyze email subject and body (snippet) text to determine if it's likely promotional/unsubscribable.
    
    This function is adapted from the original heuristic analysis in app.py but modified
    to be self-contained and not rely on Flask's app context. It examines the subject
    and body for patterns common in promotional emails and subscription-based content.
    
    Args:
        subject_text: The subject line of the email
        snippet_text: A snippet of the email body text
        list_unsubscribe_header: Optional List-Unsubscribe header value
        
    Returns:
        Dict of boolean flags indicating different heuristic results:
        {
            'has_unsubscribe_text': bool,  # Contains unsubscribe keywords
            'has_promotional_keywords': bool,  # Contains promotional keywords
            'has_promotional_formatting': bool,  # Contains typical promotional formatting
            'has_list_unsubscribe_header': bool,  # Has List-Unsubscribe header
            'likely_unsubscribable': bool  # Overall assessment
        }
    """
    # Ensure inputs are strings
    subject_text = str(subject_text).lower() if subject_text else ""
    snippet_text = str(snippet_text).lower() if snippet_text else ""
    combined_text = f"{subject_text} {snippet_text}".lower()
    
    # Initialize result with default values
    result = {
        'has_unsubscribe_text': False,
        'has_promotional_keywords': False,
        'has_promotional_formatting': False,
        'has_list_unsubscribe_header': False,
        'likely_unsubscribable': False
    }
    
    # Check for unsubscribe keywords
    for keyword in UNSUBSCRIBE_KEYWORDS_FOR_AI_HEURISTICS:
        if keyword.lower() in combined_text:
            result['has_unsubscribe_text'] = True
            break
    
    # Check for promotional keywords
    for keyword in PROMO_KEYWORDS_FOR_AI_HEURISTICS:
        if keyword.lower() in combined_text:
            result['has_promotional_keywords'] = True
            break
    
    # Check for promotional formatting patterns
    combined_text_original_case = f"{subject_text} {snippet_text}" if subject_text and snippet_text else ""
    for pattern in FORMATTING_PATTERNS_FOR_AI_HEURISTICS:
        if re.search(pattern, combined_text_original_case, re.IGNORECASE):
            result['has_promotional_formatting'] = True
            break
    
    # Check for List-Unsubscribe header
    if list_unsubscribe_header:
        result['has_list_unsubscribe_header'] = True
    
    # Overall assessment: likely unsubscribable if any of the criteria are met
    # For training data preparation, we want to be somewhat inclusive in what we label as potentially unsubscribable
    result['likely_unsubscribable'] = any([
        result['has_unsubscribe_text'],
        (result['has_promotional_keywords'] and result['has_promotional_formatting']),
        result['has_list_unsubscribe_header']
    ])
    
    return result


# --- Text Cleaning Utilities ---

def clean_html_text(html_content: str) -> str:
    """
    Clean HTML content and extract readable text.
    
    Args:
        html_content: Raw HTML content string
        
    Returns:
        Cleaned plain text extracted from HTML
    """
    if not html_content:
        return ""
    
    try:
        # Create BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            script_or_style.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text: replace multiple newlines, spaces, etc.
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    except Exception:
        # If parsing fails, try to extract text with regex (fallback)
        text = re.sub(r'<[^>]*>', ' ', html_content)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def normalize_spaces(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace newlines, tabs with spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    return text.strip()


def normalize_urls(text: str) -> str:
    """
    Replace URLs with a placeholder to reduce noise in training data.
    
    Args:
        text: Input text
        
    Returns:
        Text with URLs replaced by a placeholder
    """
    if not text:
        return ""
    
    # URL regex pattern
    url_pattern = r'(https?://[^\s]+)|(www\.[^\s]+\.[^\s]+)'
    
    # Replace URLs with placeholder
    return re.sub(url_pattern, '[URL]', text)


def clean_text_for_model(text: str, max_length: Optional[int] = None) -> str:
    """
    Clean and normalize text for model input.
    
    Args:
        text: Input text (can be HTML or plain text)
        max_length: Optional maximum length to truncate to
        
    Returns:
        Cleaned text ready for model input
    """
    if not text:
        return ""
    
    # Check if input is likely HTML
    if re.search(r'<\w+[^>]*>.*?</\w+>', text, re.DOTALL):
        text = clean_html_text(text)
    
    # Normalize whitespace
    text = normalize_spaces(text)
    
    # Replace URLs with placeholder
    text = normalize_urls(text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


# --- Timestamp and Path Utilities ---

def get_current_timestamp() -> str:
    """Returns ISO format timestamp for current time."""
    return datetime.datetime.now().isoformat()


def get_current_timestamp_log_prefix() -> str:
    """Returns a formatted timestamp string for log entries."""
    return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception:
        return False


# --- Email Header Analysis ---

def extract_email_addresses(header_value: str) -> List[str]:
    """
    Extract email addresses from a header value.
    
    Args:
        header_value: Raw header value containing email addresses
        
    Returns:
        List of extracted email addresses
    """
    if not header_value:
        return []
    
    # Basic email regex pattern
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    return re.findall(email_pattern, header_value)


def parse_list_unsubscribe_header(header_value: str) -> Dict[str, Any]:
    """
    Parse the List-Unsubscribe header to extract URLs and email addresses.
    
    Args:
        header_value: Raw List-Unsubscribe header value
        
    Returns:
        Dict with extracted URLs and email addresses
    """
    if not header_value:
        return {"urls": [], "emails": []}
    
    result = {"urls": [], "emails": []}
    
    # Split by comma and process each value
    for item in header_value.split(','):
        item = item.strip()
        
        # Handle <mailto:...> format
        if item.startswith('<mailto:') and item.endswith('>'):
            email = item[8:-1]  # Remove <mailto: and >
            result["emails"].append(email)
        
        # Handle <http...> format
        elif item.startswith('<http') and item.endswith('>'):
            url = item[1:-1]  # Remove < and >
            result["urls"].append(url)
    
    return result