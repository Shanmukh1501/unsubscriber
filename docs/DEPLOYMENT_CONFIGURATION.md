# Gmail Unsubscriber - Deployment & Configuration Guide

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [Configuration Management](#configuration-management)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Logging](#monitoring--logging)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

## Overview

The Gmail Unsubscriber is designed for flexible deployment across various environments, from local development to enterprise production systems. This guide provides comprehensive instructions for setting up, configuring, and maintaining the application in different deployment scenarios.

### Deployment Architecture

```
Production Deployment Architecture
├── Load Balancer (nginx/Apache)
│   ├── SSL Termination
│   ├── Rate Limiting
│   └── Static File Serving
├── Application Servers (Multiple Instances)
│   ├── Flask Application (Gunicorn/uWSGI)
│   ├── ML Model Loading
│   └── Background Task Processing
├── Cache Layer
│   ├── Redis (Session Storage)
│   ├── FileSystem Cache (Message Cache)
│   └── Model Cache
├── Storage
│   ├── Application Files
│   ├── ML Models (567MB)
│   ├── Cache Data
│   └── Log Files
└── External Services
    ├── Google OAuth 2.0
    ├── Gmail API
    └── Monitoring (Optional)
```

## System Requirements

### Minimum Requirements

```yaml
Hardware:
  CPU: 2 cores, 2.0GHz
  RAM: 4GB (8GB recommended for AI features)
  Storage: 5GB free space
  Network: Stable internet connection

Software:
  OS: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
  Python: 3.8+ (3.9 recommended)
  pip: 21.0+
  Git: 2.20+

Optional (for GPU acceleration):
  CUDA: 11.0+
  GPU: 4GB VRAM minimum
```

### Recommended Production Requirements

```yaml
Hardware:
  CPU: 4+ cores, 3.0GHz
  RAM: 16GB (for multiple concurrent users)
  Storage: 50GB SSD (for logs, cache, models)
  Network: High-speed internet with low latency to Google APIs

Software:
  OS: Ubuntu 22.04 LTS or CentOS 8+
  Python: 3.9+
  Reverse Proxy: nginx 1.18+ or Apache 2.4+
  Process Manager: systemd
  Monitoring: Prometheus + Grafana (optional)

Security:
  SSL Certificate: Valid HTTPS certificate
  Firewall: Configured for required ports only
  User Access: Non-root application user
```

## Environment Setup

### 1. Python Environment Setup

```bash
# Create project directory
mkdir -p /opt/gmail-unsubscriber
cd /opt/gmail-unsubscriber

# Clone repository
git clone https://github.com/your-repo/gmail-unsubscriber.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    nginx \
    redis-server \
    supervisor \
    certbot

# CentOS/RHEL
sudo yum update
sudo yum install -y \
    python3-devel \
    python3-pip \
    gcc \
    gcc-c++ \
    curl \
    nginx \
    redis \
    supervisor

# macOS (using Homebrew)
brew install python@3.9 nginx redis
```

### 3. GPU Support (Optional)

```bash
# NVIDIA GPU setup for AI acceleration
# Check GPU compatibility
nvidia-smi

# Install CUDA toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration Management

### 1. Environment Variables

Create a comprehensive environment configuration file:

```bash
# /opt/gmail-unsubscriber/.env
# Production Environment Configuration

# Flask Configuration
FLASK_ENV=production
FLASK_SECRET_KEY=your-super-secret-key-here-change-this-in-production
FLASK_DEBUG=False

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Application Configuration
PORT=5000
WORKERS=4
TIMEOUT=120
KEEPALIVE=5

# Cache Configuration
CACHE_TYPE=FileSystemCache
CACHE_DIR=/var/cache/gmail-unsubscriber
CACHE_DEFAULT_TIMEOUT=21600

# AI Model Configuration
MODEL_DIR=/opt/gmail-unsubscriber/final_optimized_model
ENABLE_AI_FEATURES=true
DEFAULT_AI_THRESHOLD=0.75
MODEL_CACHE_SIZE=1024

# Security Configuration
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Lax
PERMANENT_SESSION_LIFETIME=86400

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/var/log/gmail-unsubscriber/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5

# Performance Configuration
MAX_CONTENT_LENGTH=16777216
SEND_FILE_MAX_AGE_DEFAULT=31536000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Monitoring (Optional)
SENTRY_DSN=your-sentry-dsn-here
PROMETHEUS_METRICS=true
HEALTH_CHECK_ENABLED=true
```

### 2. Application Configuration

```python
# config.py - Advanced application configuration
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-key-change-in-production'
    
    # Cache configuration
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'FileSystemCache')
    CACHE_DIR = os.environ.get('CACHE_DIR', './flask_cache')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 21600))
    
    # Session configuration
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(
        seconds=int(os.environ.get('PERMANENT_SESSION_LIFETIME', 86400))
    )
    
    # File upload limits
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # AI Configuration
    ENABLE_AI_FEATURES = os.environ.get('ENABLE_AI_FEATURES', 'true').lower() == 'true'
    DEFAULT_AI_THRESHOLD = float(os.environ.get('DEFAULT_AI_THRESHOLD', 0.75))
    
    # Security headers
    SEND_FILE_MAX_AGE_DEFAULT = int(os.environ.get('SEND_FILE_MAX_AGE_DEFAULT', 31536000))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Relaxed security for development
    SESSION_COOKIE_SECURE = False
    
    # Development-specific settings
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    
    # Production optimizations
    CACHE_DEFAULT_TIMEOUT = 21600  # 6 hours
    LOG_LEVEL = 'INFO'
    
    # Production-specific features
    RATE_LIMITING = True
    METRICS_ENABLED = True

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Test-specific settings
    CACHE_TYPE = 'simple'
    SESSION_COOKIE_SECURE = False
    ENABLE_AI_FEATURES = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

### 3. OAuth Configuration

```json
# client_secret.json (for local development only)
{
  "web": {
    "client_id": "your-client-id.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "your-client-secret",
    "redirect_uris": [
      "http://localhost:5000/oauth2callback",
      "https://yourdomain.com/oauth2callback"
    ]
  }
}
```

## Local Development

### 1. Quick Start

```bash
# Clone and setup
git clone https://github.com/your-repo/gmail-unsubscriber.git
cd gmail-unsubscriber

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Setup Google OAuth
# 1. Go to Google Cloud Console
# 2. Create new project or select existing
# 3. Enable Gmail API
# 4. Create OAuth 2.0 credentials
# 5. Download client_secret.json

# Initialize AI model (if available)
python -c "from ml_suite.predictor import initialize_predictor; import logging; initialize_predictor(logging.getLogger())"

# Run development server
python app.py
```

### 2. Development Environment Configuration

```bash
# Development-specific environment variables
cat > .env.development << EOF
FLASK_ENV=development
FLASK_DEBUG=True
GOOGLE_CLIENT_ID=your-dev-client-id
GOOGLE_CLIENT_SECRET=your-dev-client-secret
CACHE_DEFAULT_TIMEOUT=300
LOG_LEVEL=DEBUG
OAUTHLIB_INSECURE_TRANSPORT=1
EOF

# Load environment
export $(cat .env.development | xargs)

# Run with hot reload
python app.py
```

### 3. Development Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting
black . --line-length 100
isort . --profile black

# Linting
flake8 . --max-line-length 100 --ignore E203,W503
pylint app.py ml_suite/

# Type checking
mypy app.py --ignore-missing-imports

# Testing
python -m pytest tests/ -v --cov=app --cov=ml_suite

# Security scanning
bandit -r . -ll

# Dependency scanning
safety check
```

## Production Deployment

### 1. Systemd Service Configuration

```ini
# /etc/systemd/system/gmail-unsubscriber.service
[Unit]
Description=Gmail Unsubscriber Application
After=network.target
Wants=network.target

[Service]
Type=forking
User=gmail-unsubscriber
Group=gmail-unsubscriber
WorkingDirectory=/opt/gmail-unsubscriber
Environment=PATH=/opt/gmail-unsubscriber/venv/bin
ExecStart=/opt/gmail-unsubscriber/venv/bin/gunicorn \
    --daemon \
    --workers 4 \
    --worker-class sync \
    --worker-connections 1000 \
    --timeout 120 \
    --keepalive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --bind 127.0.0.1:5000 \
    --pid /var/run/gmail-unsubscriber/app.pid \
    --access-logfile /var/log/gmail-unsubscriber/access.log \
    --error-logfile /var/log/gmail-unsubscriber/error.log \
    --capture-output \
    --enable-stdio-inheritance \
    wsgi:app
    
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PIDFile=/var/run/gmail-unsubscriber/app.pid

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/gmail-unsubscriber /var/log/gmail-unsubscriber /var/cache/gmail-unsubscriber

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Restart policy
Restart=always
RestartSec=5
StartLimitBurst=3
StartLimitInterval=60

[Install]
WantedBy=multi-user.target
```

### 2. WSGI Configuration

```python
# wsgi.py - Production WSGI entry point
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Add project directory to Python path
project_dir = '/opt/gmail-unsubscriber'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import application
from app import app

# Configure production logging
if not app.debug:
    # File handler for application logs
    log_file = '/var/log/gmail-unsubscriber/app.log'
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    
    # System handler for systemd journal
    from systemd.journal import JournalHandler
    journal_handler = JournalHandler(SYSLOG_IDENTIFIER='gmail-unsubscriber')
    journal_handler.setLevel(logging.INFO)
    app.logger.addHandler(journal_handler)
    
    app.logger.info('Gmail Unsubscriber startup')

# Initialize AI components on startup
try:
    from app import initialize_ai_components_on_app_start
    initialize_ai_components_on_app_start(app)
except Exception as e:
    app.logger.error(f'Failed to initialize AI components: {e}')

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers."""
    try:
        # Check AI predictor status
        from ml_suite.predictor import is_predictor_ready
        ai_status = is_predictor_ready()
        
        return {
            'status': 'healthy',
            'ai_ready': ai_status,
            'timestamp': time.time()
        }, 200
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }, 503

if __name__ == "__main__":
    app.run()
```

### 3. Nginx Configuration

```nginx
# /etc/nginx/sites-available/gmail-unsubscriber
upstream app_server {
    server 127.0.0.1:5000;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=1r/s;

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.tailwindcss.com; style-src 'self' 'unsafe-inline' cdn.tailwindcss.com; font-src 'self' cdnjs.cloudflare.com; img-src 'self' data: https:; connect-src 'self' accounts.google.com;";
    
    # General settings
    client_max_body_size 16M;
    keepalive_timeout 65;
    
    # Logging
    access_log /var/log/nginx/gmail-unsubscriber-access.log;
    error_log /var/log/nginx/gmail-unsubscriber-error.log;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/javascript
        application/xml+rss
        application/json;
    
    # Main application
    location / {
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Proxy settings
        proxy_pass http://app_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        
        # Keep alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # Authentication endpoints with stricter rate limiting
    location ~ ^/(login|oauth2callback) {
        limit_req zone=auth burst=5 nodelay;
        
        proxy_pass http://app_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API endpoints
    location /api/ {
        limit_req zone=api burst=30 nodelay;
        
        proxy_pass http://app_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Extended timeout for AI operations
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://app_server;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/gmail-unsubscriber/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Favicon
    location = /favicon.ico {
        log_not_found off;
        access_log off;
        return 204;
    }
    
    # Robots.txt
    location = /robots.txt {
        log_not_found off;
        access_log off;
        return 200 "User-agent: *\nDisallow: /\n";
        add_header Content-Type text/plain;
    }
}
```

### 4. SSL Certificate Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal setup
sudo crontab -e
# Add line:
0 12 * * * /usr/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"

# Test auto-renewal
sudo certbot renew --dry-run
```

### 5. Production Deployment Script

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

# Configuration
APP_USER="gmail-unsubscriber"
APP_DIR="/opt/gmail-unsubscriber"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="gmail-unsubscriber"
BACKUP_DIR="/opt/backups/gmail-unsubscriber"

echo "=== Gmail Unsubscriber Production Deployment ==="

# Create backup
echo "Creating backup..."
sudo mkdir -p "$BACKUP_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
sudo tar -czf "$BACKUP_DIR/backup_$timestamp.tar.gz" -C "$APP_DIR" . || true

# Stop service
echo "Stopping service..."
sudo systemctl stop "$SERVICE_NAME" || true

# Update code
echo "Updating application code..."
cd "$APP_DIR"
sudo -u "$APP_USER" git pull origin main

# Update dependencies
echo "Updating dependencies..."
sudo -u "$APP_USER" $VENV_DIR/bin/pip install --upgrade pip
sudo -u "$APP_USER" $VENV_DIR/bin/pip install -r requirements.txt

# Run database migrations (if applicable)
echo "Running migrations..."
# sudo -u "$APP_USER" $VENV_DIR/bin/python migrate.py

# Update AI models (if needed)
echo "Checking AI model status..."
sudo -u "$APP_USER" $VENV_DIR/bin/python -c "
from ml_suite.predictor import get_model_status
status = get_model_status()
print(f'Model status: {status}')
"

# Set correct permissions
echo "Setting permissions..."
sudo chown -R "$APP_USER:$APP_USER" "$APP_DIR"
sudo chmod -R 755 "$APP_DIR"
sudo chmod 600 "$APP_DIR/.env"

# Restart service
echo "Starting service..."
sudo systemctl start "$SERVICE_NAME"
sudo systemctl enable "$SERVICE_NAME"

# Verify deployment
echo "Verifying deployment..."
sleep 5
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✓ Service is running"
else
    echo "✗ Service failed to start"
    sudo systemctl status "$SERVICE_NAME"
    exit 1
fi

# Health check
echo "Performing health check..."
health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health || echo "failed")
if [ "$health_response" = "200" ]; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed (HTTP $health_response)"
    exit 1
fi

echo "=== Deployment completed successfully ==="
echo "Application is running at: https://yourdomain.com"
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Multi-stage Dockerfile for optimized production builds
FROM python:3.9-slim as base

# Build arguments
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -g $APP_GID $APP_USER && \
    useradd -u $APP_UID -g $APP_GID -m -s /bin/bash $APP_USER

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY --chown=$APP_USER:$APP_USER . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/models && \
    chown -R $APP_USER:$APP_USER /app

# Switch to application user
USER $APP_USER

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "wsgi:app"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml - Complete containerized deployment
version: '3.8'

services:
  app:
    build:
      context: .
      target: production
    image: gmail-unsubscriber:latest
    container_name: gmail-unsubscriber-app
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - FLASK_SECRET_KEY=${FLASK_SECRET_KEY}
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
      - GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
      - CACHE_TYPE=FileSystemCache
      - CACHE_DIR=/app/cache
      - LOG_LEVEL=INFO
    volumes:
      - app_cache:/app/cache
      - app_logs:/app/logs
      - app_models:/app/models
      - ./final_optimized_model:/app/final_optimized_model:ro
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - app_network
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  redis:
    image: redis:7-alpine
    container_name: gmail-unsubscriber-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    networks:
      - app_network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  nginx:
    image: nginx:alpine
    container_name: gmail-unsubscriber-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./ssl:/etc/ssl/certs:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - app
    networks:
      - app_network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  app_cache:
    driver: local
  app_logs:
    driver: local
  app_models:
    driver: local
  redis_data:
    driver: local
  nginx_logs:
    driver: local

networks:
  app_network:
    driver: bridge
```

### 3. Docker Environment Configuration

```bash
# .env.docker - Docker-specific environment variables
FLASK_SECRET_KEY=your-super-secret-docker-key
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Docker Compose variables
COMPOSE_PROJECT_NAME=gmail-unsubscriber
COMPOSE_FILE=docker-compose.yml

# Resource limits
APP_MEMORY_LIMIT=2g
APP_CPU_LIMIT=1.5
REDIS_MEMORY_LIMIT=256m

# Scaling configuration
APP_REPLICAS=2
```

### 4. Docker Deployment Commands

```bash
# Build and deploy
docker-compose build
docker-compose up -d

# Scale application
docker-compose up -d --scale app=3

# View logs
docker-compose logs -f app

# Update deployment
docker-compose pull
docker-compose up -d --force-recreate

# Backup volumes
docker run --rm -v gmail-unsubscriber_app_cache:/data -v $(pwd):/backup alpine tar czf /backup/cache_backup.tar.gz /data

# Health check
curl -f http://localhost:5000/health

# Stop and cleanup
docker-compose down
docker-compose down -v  # Remove volumes
```

## Cloud Deployment

### 1. AWS Deployment

```yaml
# cloudformation-template.yml - AWS CloudFormation template
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Gmail Unsubscriber - Production Deployment'

Parameters:
  EnvironmentName:
    Type: String
    Default: production
  InstanceType:
    Type: String
    Default: t3.medium
    AllowedValues: [t3.small, t3.medium, t3.large, t3.xlarge]
  KeyPairName:
    Type: AWS::EC2::KeyPair::KeyName
    Description: EC2 Key Pair for SSH access

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-VPC

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-IGW

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Public-Subnet

  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-Public-Routes

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet

  # Security Groups
  ApplicationSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupName: !Sub ${EnvironmentName}-App-SG
      GroupDescription: Security group for Gmail Unsubscriber application
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
      SecurityGroupEgress:
        - IpProtocol: -1
          CidrIp: 0.0.0.0/0

  # IAM Role for EC2 instance
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Policies:
        - PolicyName: S3ModelAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                Resource: !Sub '${ModelBucket}/*'

  EC2InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref EC2Role

  # S3 Bucket for model storage
  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ${EnvironmentName}-gmail-unsubscriber-models
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # EC2 Instance
  ApplicationInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c02fb55956c7d316  # Amazon Linux 2
      InstanceType: !Ref InstanceType
      KeyName: !Ref KeyPairName
      SubnetId: !Ref PublicSubnet
      SecurityGroupIds:
        - !Ref ApplicationSecurityGroup
      IamInstanceProfile: !Ref EC2InstanceProfile
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          amazon-linux-extras install docker
          service docker start
          usermod -a -G docker ec2-user
          
          # Install docker-compose
          curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
          chmod +x /usr/local/bin/docker-compose
          
          # Clone application
          cd /opt
          git clone https://github.com/your-repo/gmail-unsubscriber.git
          cd gmail-unsubscriber
          
          # Setup environment
          cat > .env << EOF
          FLASK_SECRET_KEY=${AWS::StackId}
          GOOGLE_CLIENT_ID=${GoogleClientId}
          GOOGLE_CLIENT_SECRET=${GoogleClientSecret}
          AWS_DEFAULT_REGION=${AWS::Region}
          MODEL_BUCKET=${ModelBucket}
          EOF
          
          # Deploy with Docker Compose
          docker-compose up -d
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-App-Instance

Outputs:
  ApplicationURL:
    Description: Application URL
    Value: !Sub 'http://${ApplicationInstance.PublicDnsName}'
  
  SSHCommand:
    Description: SSH command to connect to instance
    Value: !Sub 'ssh -i ${KeyPairName}.pem ec2-user@${ApplicationInstance.PublicDnsName}'
```

### 2. Google Cloud Platform Deployment

```yaml
# gcp-deployment.yaml - Google Cloud Deployment Manager template
resources:
- name: gmail-unsubscriber-vm
  type: compute.v1.instance
  properties:
    zone: us-central1-a
    machineType: https://www.googleapis.com/compute/v1/projects/[PROJECT-ID]/zones/us-central1-a/machineTypes/n1-standard-2
    disks:
    - deviceName: boot
      type: PERSISTENT
      boot: true
      autoDelete: true
      initializeParams:
        sourceImage: https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts
        diskSizeGb: 50
    networkInterfaces:
    - network: https://www.googleapis.com/compute/v1/projects/[PROJECT-ID]/global/networks/default
      accessConfigs:
      - name: External NAT
        type: ONE_TO_ONE_NAT
    serviceAccounts:
    - email: default
      scopes:
      - https://www.googleapis.com/auth/cloud-platform
    metadata:
      items:
      - key: startup-script
        value: |
          #!/bin/bash
          apt-get update
          apt-get install -y docker.io docker-compose git python3-pip
          
          # Clone and deploy application
          cd /opt
          git clone https://github.com/your-repo/gmail-unsubscriber.git
          cd gmail-unsubscriber
          
          # Setup environment
          cat > .env << EOF
          FLASK_SECRET_KEY=gcp-production-key
          GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID
          GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET
          EOF
          
          # Deploy
          docker-compose up -d
    tags:
      items:
      - http-server
      - https-server

- name: gmail-unsubscriber-firewall
  type: compute.v1.firewall
  properties:
    allowed:
    - IPProtocol: TCP
      ports: [80, 443]
    sourceRanges: [0.0.0.0/0]
    targetTags: [http-server, https-server]
```

## Monitoring & Logging

### 1. Application Monitoring

```python
# monitoring.py - Application monitoring and metrics
import time
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics collection
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
AI_PREDICTIONS = Counter('ai_predictions_total', 'Total AI predictions', ['model_type', 'label'])
AI_PREDICTION_DURATION = Histogram('ai_prediction_duration_seconds', 'AI prediction duration')

# System metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_bytes', 'System disk usage')

class MetricsCollector:
    """Collect and expose application and system metrics."""
    
    def __init__(self, app):
        self.app = app
        self.start_time = time.time()
        self.active_sessions = set()
        
        # Start background metrics collection
        self.metrics_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.metrics_thread.start()
        
        # Setup Flask request middleware
        self.setup_request_monitoring()
    
    def setup_request_monitoring(self):
        """Setup request monitoring middleware."""
        
        @self.app.before_request
        def before_request():
            request.start_time = time.time()
        
        @self.app.after_request
        def after_request(response):
            # Record request metrics
            duration = time.time() - getattr(request, 'start_time', time.time())
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.endpoint or 'unknown',
                status=response.status_code
            ).inc()
            
            # Track active users (simplified)
            if 'user_info' in session:
                user_id = session['user_info'].get('email')
                if user_id:
                    self.active_sessions.add(user_id)
                    ACTIVE_USERS.set(len(self.active_sessions))
            
            return response
    
    def _collect_system_metrics(self):
        """Collect system metrics in background thread."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                SYSTEM_CPU_USAGE.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                SYSTEM_DISK_USAGE.set(disk.used)
                
                # Clean up old sessions (every 5 minutes)
                if len(self.active_sessions) > 1000:
                    self.active_sessions.clear()
                
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                self.app.logger.error(f'Metrics collection error: {e}')
                time.sleep(60)
    
    def record_ai_prediction(self, model_type, predicted_label, duration):
        """Record AI prediction metrics."""
        AI_PREDICTIONS.labels(model_type=model_type, label=predicted_label).inc()
        AI_PREDICTION_DURATION.observe(duration)

# Metrics endpoint
@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

# Initialize metrics collector
if os.environ.get('PROMETHEUS_METRICS', 'false').lower() == 'true':
    metrics_collector = MetricsCollector(app)
```

### 2. Structured Logging

```python
# logging_config.py - Structured logging configuration
import json
import logging
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add application info
        log_record['application'] = 'gmail-unsubscriber'
        log_record['version'] = '1.0.0'
        
        # Add request context if available
        try:
            from flask import request, g
            if request:
                log_record['request_id'] = getattr(g, 'request_id', None)
                log_record['user_agent'] = request.headers.get('User-Agent')
                log_record['remote_addr'] = request.remote_addr
                log_record['method'] = request.method
                log_record['path'] = request.path
        except RuntimeError:
            # Outside request context
            pass

def setup_logging(app):
    """Setup comprehensive logging for the application."""
    
    # Remove default Flask handlers
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)
    
    # Create formatters
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler (JSON format for log aggregation)
    if not app.debug:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            '/var/log/gmail-unsubscriber/app.json',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
    
    # Add console handler
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)
    
    # Setup other loggers
    for logger_name in ['werkzeug', 'gunicorn.error', 'gunicorn.access']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.addHandler(console_handler)
        if not app.debug:
            logger.addHandler(file_handler)
    
    # AI-specific logging
    ai_handler = RotatingFileHandler(
        '/var/log/gmail-unsubscriber/ai.json',
        maxBytes=50*1024*1024,
        backupCount=3
    )
    ai_handler.setFormatter(json_formatter)
    ai_logger = logging.getLogger('ml_suite')
    ai_logger.addHandler(ai_handler)
    ai_logger.setLevel(logging.INFO)
```

### 3. Health Checks

```python
# health_checks.py - Comprehensive health monitoring
import time
import threading
from datetime import datetime, timedelta

class HealthChecker:
    """Comprehensive application health monitoring."""
    
    def __init__(self, app):
        self.app = app
        self.checks = {}
        self.last_check_time = {}
        self.check_interval = 60  # seconds
        
        # Register health checks
        self.register_checks()
        
        # Start background health monitoring
        self.monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitor_thread.start()
    
    def register_checks(self):
        """Register all health check functions."""
        self.checks = {
            'database': self._check_database,
            'cache': self._check_cache,
            'ai_model': self._check_ai_model,
            'external_apis': self._check_external_apis,
            'disk_space': self._check_disk_space,
            'memory': self._check_memory
        }
    
    def _check_database(self):
        """Check database connectivity (if applicable)."""
        # Placeholder for database check
        return {'status': 'healthy', 'details': 'No database configured'}
    
    def _check_cache(self):
        """Check cache system health."""
        try:
            from app import cache
            # Test cache write/read
            test_key = 'health_check_test'
            test_value = str(time.time())
            cache.set(test_key, test_value, timeout=10)
            retrieved_value = cache.get(test_key)
            
            if retrieved_value == test_value:
                return {'status': 'healthy', 'details': 'Cache read/write successful'}
            else:
                return {'status': 'unhealthy', 'details': 'Cache read/write failed'}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Cache error: {str(e)}'}
    
    def _check_ai_model(self):
        """Check AI model availability and performance."""
        try:
            from ml_suite.predictor import is_predictor_ready, get_ai_prediction_for_email
            
            if not is_predictor_ready():
                return {'status': 'unhealthy', 'details': 'AI predictor not ready'}
            
            # Test prediction performance
            start_time = time.time()
            test_text = "Subject: Test Email\n\nThis is a test email for health check."
            prediction = get_ai_prediction_for_email(test_text)
            prediction_time = (time.time() - start_time) * 1000
            
            if prediction and prediction.get('label'):
                return {
                    'status': 'healthy',
                    'details': f'AI prediction successful in {prediction_time:.1f}ms',
                    'prediction_time_ms': prediction_time
                }
            else:
                return {'status': 'unhealthy', 'details': 'AI prediction failed'}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'AI model error: {str(e)}'}
    
    def _check_external_apis(self):
        """Check external API connectivity."""
        try:
            import requests
            
            # Check Google OAuth endpoint
            response = requests.get(
                'https://accounts.google.com/.well-known/openid_configuration',
                timeout=10
            )
            
            if response.status_code == 200:
                return {'status': 'healthy', 'details': 'Google APIs accessible'}
            else:
                return {'status': 'unhealthy', 'details': f'Google APIs returned {response.status_code}'}
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'External API error: {str(e)}'}
    
    def _check_disk_space(self):
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_percent = (free / total) * 100
            
            if free_percent > 10:  # More than 10% free
                return {
                    'status': 'healthy',
                    'details': f'{free_percent:.1f}% disk space available',
                    'free_gb': free // (1024**3)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'details': f'Low disk space: {free_percent:.1f}% available',
                    'free_gb': free // (1024**3)
                }
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Disk check error: {str(e)}'}
    
    def _check_memory(self):
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 90:  # Less than 90% used
                return {
                    'status': 'healthy',
                    'details': f'{memory_percent:.1f}% memory used',
                    'available_gb': memory.available // (1024**3)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'details': f'High memory usage: {memory_percent:.1f}%',
                    'available_gb': memory.available // (1024**3)
                }
        except Exception as e:
            return {'status': 'unhealthy', 'details': f'Memory check error: {str(e)}'}
    
    def _background_monitoring(self):
        """Background thread for continuous health monitoring."""
        while True:
            try:
                current_time = time.time()
                
                for check_name, check_func in self.checks.items():
                    last_check = self.last_check_time.get(check_name, 0)
                    
                    if current_time - last_check > self.check_interval:
                        result = check_func()
                        self.last_check_time[check_name] = current_time
                        
                        # Log unhealthy checks
                        if result['status'] == 'unhealthy':
                            self.app.logger.warning(f'Health check failed: {check_name} - {result["details"]}')
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.app.logger.error(f'Health monitoring error: {e}')
                time.sleep(60)
    
    def get_health_status(self):
        """Get current health status of all components."""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                health_status['checks'][check_name] = result
                
                if result['status'] == 'unhealthy':
                    health_status['overall_status'] = 'unhealthy'
            except Exception as e:
                health_status['checks'][check_name] = {
                    'status': 'unhealthy',
                    'details': f'Check failed: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
        
        return health_status

# Enhanced health endpoint
@app.route('/health')
def health_endpoint():
    """Enhanced health check endpoint."""
    health_checker = getattr(app, 'health_checker', None)
    
    if not health_checker:
        health_checker = HealthChecker(app)
        app.health_checker = health_checker
    
    health_status = health_checker.get_health_status()
    status_code = 200 if health_status['overall_status'] == 'healthy' else 503
    
    return health_status, status_code
```

This comprehensive deployment and configuration documentation provides everything needed to successfully deploy, configure, and maintain the Gmail Unsubscriber application across different environments and platforms.