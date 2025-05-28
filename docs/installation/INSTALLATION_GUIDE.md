# Installation Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Google OAuth Setup](#google-oauth-setup)
5. [Environment Configuration](#environment-configuration)
6. [Dependency Installation](#dependency-installation)
7. [Model Setup](#model-setup)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing Gmail Unsubscriber, ensure you have the following:

### Required Software
- **Python 3.8 or higher** (3.10+ recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Web browser** (Chrome, Firefox, Safari, or Edge)

### Required Accounts
- **Google Account** with Gmail access
- **Google Cloud Console** access (free tier is sufficient)

### Optional but Recommended
- **NVIDIA GPU** with CUDA support for faster AI processing
- **Virtual environment** tool (venv or conda)
- **Code editor** (VS Code, PyCharm, etc.)

## System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| OS | Windows 10, macOS 10.14, Ubuntu 18.04+ |
| CPU | Dual-core processor 2.0 GHz+ |
| RAM | 4 GB |
| Storage | 2 GB free space |
| Network | Broadband internet connection |
| Python | 3.8+ |

### Recommended Requirements
| Component | Requirement |
|-----------|-------------|
| OS | Latest Windows 11, macOS, Ubuntu 22.04+ |
| CPU | Quad-core processor 3.0 GHz+ |
| RAM | 8 GB or more |
| GPU | NVIDIA GPU with 4GB+ VRAM |
| Storage | 5 GB free space |
| Network | High-speed internet connection |
| Python | 3.10+ |

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/Unsubscriber.git

# Navigate to the project directory
cd Unsubscriber
```

### Step 2: Create Virtual Environment

It's strongly recommended to use a virtual environment:

**Using venv (Python built-in):**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Using conda:**
```bash
# Create conda environment
conda create -n unsubscriber python=3.10

# Activate environment
conda activate unsubscriber
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

For GPU support (optional):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Google OAuth Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Name your project (e.g., "Gmail Unsubscriber")
4. Note the Project ID for later use

### Step 2: Enable Gmail API

1. In the Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Gmail API"
3. Click on "Gmail API" and press "Enable"
4. Wait for the API to be enabled

### Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External (or Internal for G Suite)
   - App name: Gmail Unsubscriber
   - User support email: Your email
   - Developer contact: Your email
   - Scopes: Add the following scopes:
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/gmail.modify`
     - `https://www.googleapis.com/auth/gmail.send`
     - `https://www.googleapis.com/auth/userinfo.email`
     - `https://www.googleapis.com/auth/userinfo.profile`
4. Create OAuth client ID:
   - Application type: Web application
   - Name: Gmail Unsubscriber Web Client
   - Authorized JavaScript origins:
     - `http://localhost:5000`
   - Authorized redirect URIs:
     - `http://localhost:5000/oauth2callback`
5. Download the credentials JSON file

### Step 4: Configure Credentials

1. Rename the downloaded file to `client_secret.json`
2. Place it in the project root directory
3. Verify the structure matches:
```json
{
  "web": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost:5000/oauth2callback"],
    "javascript_origins": ["http://localhost:5000"]
  }
}
```

## Environment Configuration

### Option 1: Using client_secret.json (Default)
No additional configuration needed if `client_secret.json` is in the project root.

### Option 2: Using Environment Variables
Create a `.env` file in the project root:
```bash
# Google OAuth credentials
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret

# Flask configuration
FLASK_SECRET_KEY=your_random_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True

# Optional: Port configuration
PORT=5000
```

Generate a secure secret key:
```python
import secrets
print(secrets.token_hex(32))
```

### Option 3: System Environment Variables
```bash
# On Windows (Command Prompt)
set GOOGLE_CLIENT_ID=your_client_id
set GOOGLE_CLIENT_SECRET=your_client_secret

# On Windows (PowerShell)
$env:GOOGLE_CLIENT_ID="your_client_id"
$env:GOOGLE_CLIENT_SECRET="your_client_secret"

# On macOS/Linux
export GOOGLE_CLIENT_ID="your_client_id"
export GOOGLE_CLIENT_SECRET="your_client_secret"
```

## Dependency Installation

### Core Dependencies
The `requirements.txt` file includes all necessary packages:

```txt
# Core Web Framework
Flask>=2.3.0
flask-caching>=2.0.0

# Google API Client
google-api-python-client>=2.80.0
google-auth>=2.17.0
google-auth-httplib2>=0.1.0
google-auth-oauthlib>=1.0.0

# Machine Learning & NLP
transformers==4.36.2
torch==2.2.2
accelerate==0.25.0
datasets==2.18.0
scikit-learn==1.4.2
pandas==2.2.1
nltk==3.8.1

# HTML/Email Processing
beautifulsoup4>=4.12.0

# HTTP & Utilities
requests>=2.30.0
joblib>=1.3.0
```

### Verify Installation
```bash
# Check Python version
python --version

# Check pip packages
pip list

# Test imports
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

## Model Setup

The project includes a pre-trained model in the `final_optimized_model/` directory. No additional setup is required for the model.

### Model Files
Verify these files exist:
- `final_optimized_model/model.safetensors`
- `final_optimized_model/config.json`
- `final_optimized_model/tokenizer.json`
- `final_optimized_model/tokenizer_config.json`

### Optional: Train Your Own Model
If you want to train a custom model:
```bash
# Option 1: Fast training (2-3 hours)
python train_fast_roberta.py

# Option 2: Optimized training (3-4 hours)
python train_optimized_final.py
```

## Verification

### Step 1: Test the Installation
```bash
# Run the application
python app.py
```

Expected output:
```
Initializing AI components...
AI predictor initialized successfully!
 * Running on http://127.0.0.1:5000
```

### Step 2: Access the Web Interface
1. Open your browser
2. Navigate to `http://localhost:5000`
3. You should see the Gmail Unsubscriber login page

### Step 3: Test Authentication
1. Click "Sign in with Google"
2. Complete the OAuth flow
3. Grant the requested permissions
4. You should be redirected back to the application

### Step 4: Verify Functionality
1. Click "Scan Emails" to test email fetching
2. The AI model should load automatically
3. Check that emails are classified correctly

## Troubleshooting

### Common Issues and Solutions

#### Issue: ModuleNotFoundError
```
Solution: Ensure virtual environment is activated and run:
pip install -r requirements.txt
```

#### Issue: Google OAuth Error
```
Solution: 
1. Verify redirect URI matches exactly
2. Check that Gmail API is enabled
3. Ensure client_secret.json is valid
```

#### Issue: CUDA/GPU Not Detected
```
Solution:
1. Install CUDA-compatible PyTorch:
   pip install torch --index-url https://download.pytorch.org/whl/cu118
2. Verify CUDA installation:
   python -c "import torch; print(torch.cuda.is_available())"
```

#### Issue: Port 5000 Already in Use
```
Solution:
1. Change port in app.py or use environment variable:
   PORT=5001 python app.py
2. Or kill the process using port 5000
```

#### Issue: Model Loading Error
```
Solution:
1. Verify model files exist in final_optimized_model/
2. Check file permissions
3. Ensure sufficient disk space
```

### Getting Help

If you encounter issues not covered here:
1. Check the [FAQ documentation](../troubleshooting/FAQ.md)
2. Review application logs in the console
3. Search for similar issues in the project repository
4. Create a new issue with detailed error information

## Next Steps

After successful installation:
1. Read the [User Guide](../user-guide/USER_GUIDE.md) to learn how to use the application
2. Review the [API Documentation](../api/API_REFERENCE.md) for integration options
3. Check the [Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md) for production setup
4. Explore the [Development Guide](../development/DEVELOPMENT_GUIDE.md) to contribute

Congratulations! You've successfully installed Gmail Unsubscriber. 🎉