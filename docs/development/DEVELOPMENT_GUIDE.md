# Development Guide

## Table of Contents
1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Code Style and Standards](#code-style-and-standards)
4. [Development Workflow](#development-workflow)
5. [Testing Guidelines](#testing-guidelines)
6. [Adding New Features](#adding-new-features)
7. [ML Development](#ml-development)
8. [API Development](#api-development)
9. [Frontend Development](#frontend-development)
10. [Debugging Tips](#debugging-tips)
11. [Contributing Guidelines](#contributing-guidelines)

## Development Setup

### Prerequisites

- Python 3.8+ (3.10 recommended)
- Git
- Virtual environment tool (venv/conda)
- Code editor (VS Code recommended)
- NVIDIA GPU with CUDA (optional, for ML development)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Unsubscriber.git
cd Unsubscriber

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "120"],
    "python.linting.pylintArgs": ["--max-line-length", "120"],
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "venv": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to virtual environment
2. Enable PEP 8 coding style
3. Configure Black formatter
4. Set line length to 120 characters

## Project Structure

```
Unsubscriber/
├── app.py                      # Main Flask application
├── unsubscriber.html          # Frontend SPA
├── requirements.txt           # Python dependencies
├── client_secret.json.example # OAuth config template
│
├── ml_suite/                  # Machine Learning module
│   ├── __init__.py           # Package initialization
│   ├── config.py             # ML configuration
│   ├── predictor.py          # Inference engine
│   ├── data_preparator.py    # Data preparation
│   ├── model_trainer.py      # Model training
│   ├── utils.py              # Shared utilities
│   ├── task_utils.py         # Task management
│   └── advanced_predictor.py # Advanced features
│
├── final_optimized_model/     # Production model
│   ├── model.safetensors     # Model weights
│   ├── config.json           # Model configuration
│   └── tokenizer files       # Tokenizer assets
│
├── docs/                      # Documentation
│   ├── README.md             # Documentation index
│   ├── architecture/         # Architecture docs
│   ├── api/                  # API documentation
│   └── ...                   # Other docs
│
└── tests/                     # Test suite (future)
    ├── test_app.py           # Application tests
    ├── test_ml_suite.py      # ML tests
    └── test_api.py           # API tests
```

## Code Style and Standards

### Python Style Guide

Follow PEP 8 with these modifications:
- Line length: 120 characters
- Use Black for formatting
- Use type hints where appropriate

#### Example Code Style

```python
from typing import Dict, List, Optional, Tuple
import logging

from flask import Flask, jsonify, request
from ml_suite.predictor import get_ai_prediction_for_email


class EmailClassifier:
    """Handles email classification using AI models."""
    
    def __init__(self, model_path: str, logger: logging.Logger):
        """
        Initialize the email classifier.
        
        Args:
            model_path: Path to the trained model
            logger: Application logger instance
        """
        self.model_path = model_path
        self.logger = logger
        self._model = None
    
    def classify_email(
        self, 
        email_text: str, 
        confidence_threshold: float = 0.75
    ) -> Dict[str, any]:
        """
        Classify an email as important or unsubscribable.
        
        Args:
            email_text: The email content to classify
            confidence_threshold: Minimum confidence for classification
            
        Returns:
            Dictionary containing classification results
        """
        try:
            prediction = get_ai_prediction_for_email(email_text)
            
            return {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "is_promotional": prediction["label"] == "UNSUBSCRIBABLE",
                "meets_threshold": prediction["confidence"] >= confidence_threshold
            }
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            raise
```

### JavaScript Style Guide

- Use ES6+ features
- Prefer const over let
- Use async/await over promises
- Add JSDoc comments

```javascript
/**
 * Scan emails with specified parameters
 * @param {Object} options - Scan options
 * @param {number} options.limit - Maximum emails to scan
 * @param {boolean} options.aiEnabled - Enable AI classification
 * @returns {Promise<Array>} Array of classified emails
 */
async function scanEmails(options = {}) {
    const defaultOptions = {
        limit: 100,
        aiEnabled: true,
        scanPeriod: '180d'
    };
    
    const scanOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch('/api/scan_emails?' + new URLSearchParams(scanOptions));
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Scan failed:', error);
        throw error;
    }
}
```

### HTML/CSS Guidelines

- Use semantic HTML5 elements
- Follow BEM naming convention for CSS classes
- Use Tailwind utilities when possible
- Keep custom CSS minimal

```html
<!-- Good -->
<article class="email-card email-card--promotional">
    <header class="email-card__header">
        <h3 class="email-card__sender">Newsletter Company</h3>
        <span class="email-card__count">15 emails</span>
    </header>
    <div class="email-card__actions">
        <button class="btn btn--primary">Unsubscribe</button>
    </div>
</article>

<!-- Avoid -->
<div class="card">
    <div class="top">
        <div class="name">Newsletter Company</div>
    </div>
</div>
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run tests (when available)
pytest

# Commit changes
git add .
git commit -m "feat: add new feature description"

# Push to remote
git push origin feature/your-feature-name
```

### 2. Commit Message Convention

Follow Conventional Commits:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tool changes

Examples:
```bash
git commit -m "feat(ml): add confidence calibration to predictions"
git commit -m "fix(api): handle rate limit errors gracefully"
git commit -m "docs: update installation guide for Windows users"
```

### 3. Code Review Process

1. Create pull request with description
2. Ensure CI checks pass
3. Request review from maintainers
4. Address feedback
5. Merge after approval

## Testing Guidelines

### Unit Tests

Create tests for new functionality:

```python
# tests/test_ml_suite.py
import pytest
from ml_suite.utils import clean_text_for_model


class TestTextCleaning:
    def test_clean_html_text(self):
        html = "<p>Hello <b>World</b></p>"
        result = clean_text_for_model(html)
        assert result == "Hello World"
    
    def test_normalize_whitespace(self):
        text = "Hello    \n\n   World"
        result = clean_text_for_model(text)
        assert result == "Hello World"
    
    def test_url_normalization(self):
        text = "Visit https://example.com for more"
        result = clean_text_for_model(text)
        assert result == "Visit [URL] for more"
```

### Integration Tests

```python
# tests/test_api.py
import pytest
from app import app


class TestEmailAPI:
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_auth_status_unauthenticated(self, client):
        response = client.get('/api/auth_status')
        assert response.status_code == 200
        data = response.get_json()
        assert data['isAuthenticated'] is False
    
    def test_scan_requires_auth(self, client):
        response = client.get('/api/scan_emails')
        assert response.status_code == 401
```

### ML Model Tests

```python
# tests/test_model.py
import pytest
from ml_suite.predictor import initialize_predictor, get_ai_prediction_for_email


class TestModelPredictions:
    @pytest.fixture(scope="class")
    def initialized_predictor(self):
        initialize_predictor(logging.getLogger())
        yield
    
    def test_promotional_email_classification(self, initialized_predictor):
        email = "Subject: 50% OFF Sale!\n\nLimited time offer. Unsubscribe here."
        prediction = get_ai_prediction_for_email(email)
        assert prediction['label'] == 'UNSUBSCRIBABLE'
        assert prediction['confidence'] > 0.8
    
    def test_important_email_classification(self, initialized_predictor):
        email = "Subject: Password Reset\n\nClick here to reset your password."
        prediction = get_ai_prediction_for_email(email)
        assert prediction['label'] == 'IMPORTANT'
```

## Adding New Features

### 1. Backend Feature

Example: Adding email statistics endpoint

```python
# In app.py
@app.route('/api/email_stats')
def get_email_stats():
    """Get statistics about scanned emails."""
    if not get_user_credentials():
        return jsonify({"error": "Authentication required"}), 401
    
    # Implementation
    stats = calculate_email_statistics()
    return jsonify(stats), 200


def calculate_email_statistics():
    """Calculate statistics from recent scans."""
    # Implementation details
    return {
        "total_scanned": 1000,
        "promotional_found": 750,
        "unsubscribe_success": 650,
        "time_saved_hours": 12.5
    }
```

### 2. Frontend Feature

Example: Adding statistics display

```javascript
// In unsubscriber.html
async function displayEmailStats() {
    try {
        const stats = await fetch('/api/email_stats').then(r => r.json());
        
        const statsHtml = `
            <div class="stats-container">
                <div class="stat-card">
                    <h3>${stats.total_scanned}</h3>
                    <p>Emails Scanned</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.promotional_found}</h3>
                    <p>Promotional Found</p>
                </div>
                <div class="stat-card">
                    <h3>${stats.time_saved_hours}h</h3>
                    <p>Time Saved</p>
                </div>
            </div>
        `;
        
        document.getElementById('stats-area').innerHTML = statsHtml;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}
```

### 3. ML Feature

Example: Adding email category detection

```python
# In ml_suite/advanced_features.py
def detect_email_category(email_text: str) -> str:
    """
    Detect specific email category beyond binary classification.
    
    Returns:
        Category: 'newsletter', 'marketing', 'transactional', etc.
    """
    # Extract features
    features = extract_email_features(email_text)
    
    # Apply rules or secondary model
    if 'newsletter' in features['keywords']:
        return 'newsletter'
    elif 'order' in features['keywords'] and 'shipped' in features['keywords']:
        return 'transactional'
    # ... more logic
    
    return 'general'
```

## ML Development

### Training Custom Models

1. **Prepare Data**:
```python
# Create custom dataset
import pandas as pd

df = pd.DataFrame({
    'text': ['email content...'],
    'label': [0]  # 0: Important, 1: Unsubscribable
})
df.to_csv('custom_training_data.csv', index=False)
```

2. **Configure Training**:
```python
# Modify ml_suite/config.py
CUSTOM_TRAINING_DATA = 'custom_training_data.csv'
CUSTOM_MODEL_NAME = 'custom_unsubscriber_v1'
```

3. **Run Training**:
```bash
python -m ml_suite.model_trainer --custom
```

### Model Experimentation

```python
# experiment.py
from ml_suite.model_trainer import train_unsubscriber_model
from ml_suite.config import *

# Experiment with hyperparameters
experiments = [
    {'learning_rate': 1e-5, 'epochs': 3},
    {'learning_rate': 3e-5, 'epochs': 5},
    {'learning_rate': 5e-5, 'epochs': 4}
]

for exp in experiments:
    # Update config
    LEARNING_RATE = exp['learning_rate']
    NUM_TRAIN_EPOCHS = exp['epochs']
    
    # Train
    results = train_unsubscriber_model()
    print(f"Experiment {exp}: {results['metrics']}")
```

## API Development

### Adding New Endpoints

1. **Plan the endpoint**:
   - Method (GET, POST, etc.)
   - URL path
   - Parameters
   - Response format

2. **Implement with validation**:
```python
from flask import request, jsonify
from werkzeug.exceptions import BadRequest

@app.route('/api/advanced_scan', methods=['POST'])
def advanced_scan():
    """Advanced email scanning with custom filters."""
    # Validate auth
    if not get_user_credentials():
        return jsonify({"error": "Authentication required"}), 401
    
    # Validate input
    data = request.get_json()
    if not data:
        raise BadRequest("No data provided")
    
    required_fields = ['filters', 'options']
    for field in required_fields:
        if field not in data:
            raise BadRequest(f"Missing required field: {field}")
    
    # Process request
    results = perform_advanced_scan(data['filters'], data['options'])
    
    # Return response
    return jsonify({
        "success": True,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }), 200
```

### API Versioning

Consider API versioning for breaking changes:

```python
# API v1 (current)
@app.route('/api/scan_emails')
def scan_emails_v1():
    # Current implementation
    pass

# API v2 (new)
@app.route('/api/v2/scan_emails')
def scan_emails_v2():
    # New implementation with breaking changes
    pass
```

## Frontend Development

### Adding UI Components

1. **Create reusable components**:
```javascript
// Component: EmailCard
function createEmailCard(mailer) {
    const card = document.createElement('div');
    card.className = 'email-card';
    card.innerHTML = `
        <div class="email-card__header">
            <input type="checkbox" data-id="${mailer.id}">
            <h3>${mailer.senderName}</h3>
            <span class="badge">${mailer.count} emails</span>
        </div>
        <div class="email-card__body">
            <p>${mailer.exampleSubject}</p>
            ${mailer.ai_classification ? 
                `<div class="ai-badge">${mailer.ai_classification.group_label}</div>` : 
                ''
            }
        </div>
        <div class="email-card__actions">
            <button onclick="unsubscribe('${mailer.id}')">Unsubscribe</button>
            <button onclick="addToKeepList('${mailer.id}')">Keep</button>
        </div>
    `;
    return card;
}
```

2. **State Management**:
```javascript
// Simple state management
const AppState = {
    user: null,
    mailers: [],
    selectedMailers: new Set(),
    settings: {
        aiEnabled: true,
        confidenceThreshold: 0.75
    },
    
    updateUser(user) {
        this.user = user;
        this.render();
    },
    
    updateMailers(mailers) {
        this.mailers = mailers;
        this.render();
    },
    
    render() {
        // Update UI based on state
        updateUserDisplay(this.user);
        updateMailersList(this.mailers);
    }
};
```

### Responsive Design

Ensure mobile compatibility:

```css
/* Mobile-first approach */
.email-card {
    padding: 1rem;
    margin: 0.5rem;
}

/* Tablet and up */
@media (min-width: 768px) {
    .email-card {
        padding: 1.5rem;
        margin: 1rem;
    }
}

/* Desktop */
@media (min-width: 1024px) {
    .email-cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
    }
}
```

## Debugging Tips

### 1. Enable Debug Mode

```python
# Development only!
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
```

### 2. Logging

```python
# Add detailed logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing email: {email_id}")
logger.info(f"Classified as: {result['label']}")
logger.error(f"Classification failed: {error}")
```

### 3. Browser DevTools

```javascript
// Add debug logging
console.group('Email Scan');
console.log('Options:', scanOptions);
console.log('Response:', response);
console.groupEnd();

// Performance monitoring
console.time('scanEmails');
const results = await scanEmails();
console.timeEnd('scanEmails');
```

### 4. Flask Debug Toolbar

```bash
pip install flask-debugtoolbar
```

```python
from flask_debugtoolbar import DebugToolbarExtension

app.config['DEBUG_TB_ENABLED'] = True
toolbar = DebugToolbarExtension(app)
```

### 5. ML Model Debugging

```python
# Visualize predictions
def debug_prediction(email_text):
    from ml_suite.predictor import base_classification_pipeline
    
    # Get raw outputs
    outputs = base_classification_pipeline(email_text)
    
    print(f"Input: {email_text[:100]}...")
    print(f"Outputs: {outputs}")
    print(f"Tokens: {base_classification_pipeline.tokenizer.tokenize(email_text)[:20]}")
    
    return outputs
```

## Contributing Guidelines

### 1. Before Contributing

- Read existing documentation
- Check existing issues/PRs
- Discuss major changes first

### 2. Contribution Process

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit pull request

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] No regressions

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes are backward compatible
```

### 4. Code Review Checklist

- [ ] Code is readable and well-commented
- [ ] No hardcoded values
- [ ] Error handling is appropriate
- [ ] Security best practices followed
- [ ] Performance impact considered
- [ ] Tests cover new functionality

## Development Tools

### Recommended Extensions

**VS Code**:
- Python
- Pylance
- Black Formatter
- GitLens
- Thunder Client (API testing)

**Chrome DevTools**:
- React Developer Tools (if migrating to React)
- Redux DevTools (if adding state management)

### Useful Commands

```bash
# Format code
black app.py ml_suite/

# Lint code
pylint app.py ml_suite/

# Type checking
mypy app.py

# Security scan
bandit -r .

# Generate requirements
pip freeze > requirements.txt

# Profile application
python -m cProfile app.py
```

## Future Development Areas

### Planned Enhancements

1. **Test Suite**: Comprehensive unit and integration tests
2. **CI/CD Pipeline**: Automated testing and deployment
3. **Plugin System**: Extensible architecture
4. **Multi-language Support**: i18n implementation
5. **Real-time Updates**: WebSocket integration
6. **Mobile App**: React Native companion app

### Research Areas

1. **Advanced ML**: Multi-label classification
2. **NLP Features**: Sentiment analysis
3. **Computer Vision**: Screenshot analysis
4. **Federated Learning**: Privacy-preserving ML

## Resources

### Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Gmail API Reference](https://developers.google.com/gmail/api)

### Learning Resources
- [Real Python](https://realpython.com/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Hugging Face Course](https://huggingface.co/course)

### Community
- Project Issues/Discussions
- Stack Overflow
- Reddit r/flask, r/MachineLearning

---

Happy coding! 🚀 We welcome contributions that improve Gmail Unsubscriber for everyone.