# Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Local Development](#local-development)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment Options](#cloud-deployment-options)
6. [Docker Deployment](#docker-deployment)
7. [Environment Configuration](#environment-configuration)
8. [Security Considerations](#security-considerations)
9. [Performance Optimization](#performance-optimization)
10. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Overview

This guide covers deploying Gmail Unsubscriber in various environments, from local development to production cloud deployments. Choose the deployment method that best fits your needs and infrastructure.

## Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] Google OAuth credentials configured
- [ ] Python 3.8+ environment ready
- [ ] All dependencies installed
- [ ] ML model files present
- [ ] Secure secret key generated
- [ ] SSL certificate (for production)
- [ ] Domain name (for production)
- [ ] Backup strategy planned

## Local Development

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Unsubscriber.git
cd Unsubscriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Development Configuration

Create `.env` file:
```env
FLASK_ENV=development
FLASK_DEBUG=True
GOOGLE_CLIENT_ID=your_dev_client_id
GOOGLE_CLIENT_SECRET=your_dev_secret
FLASK_SECRET_KEY=dev_secret_key_change_in_production
OAUTHLIB_INSECURE_TRANSPORT=1  # Only for development!
```

## Production Deployment

### 1. Traditional Server Deployment

#### Using Gunicorn (Recommended)

Install Gunicorn:
```bash
pip install gunicorn
```

Create `gunicorn_config.py`:
```python
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 300
keepalive = 2

# Logging
accesslog = '/var/log/gunicorn/access.log'
errorlog = '/var/log/gunicorn/error.log'
loglevel = 'info'

# Process naming
proc_name = 'gmail_unsubscriber'

# Server mechanics
daemon = False
pidfile = '/var/run/gunicorn.pid'
user = 'www-data'
group = 'www-data'
tmp_upload_dir = None

# SSL (if not using reverse proxy)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'
```

Start Gunicorn:
```bash
gunicorn -c gunicorn_config.py app:app
```

#### Using uWSGI (Alternative)

Install uWSGI:
```bash
pip install uwsgi
```

Create `uwsgi.ini`:
```ini
[uwsgi]
module = app:app
master = true
processes = 4
socket = /tmp/uwsgi.sock
chmod-socket = 666
vacuum = true
die-on-term = true
logto = /var/log/uwsgi/app.log
```

Start uWSGI:
```bash
uwsgi --ini uwsgi.ini
```

### 2. Nginx Reverse Proxy

Install Nginx and create configuration:

`/etc/nginx/sites-available/gmail-unsubscriber`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Proxy settings
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running operations
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Static files (if separated)
    location /static {
        alias /var/www/gmail-unsubscriber/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Increase body size for model uploads
    client_max_body_size 1G;
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/gmail-unsubscriber /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 3. Systemd Service

Create `/etc/systemd/system/gmail-unsubscriber.service`:
```ini
[Unit]
Description=Gmail Unsubscriber Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/gmail-unsubscriber
Environment="PATH=/var/www/gmail-unsubscriber/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/var/www/gmail-unsubscriber/venv/bin/gunicorn -c gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gmail-unsubscriber
sudo systemctl start gmail-unsubscriber
sudo systemctl status gmail-unsubscriber
```

## Cloud Deployment Options

### AWS EC2 Deployment

1. **Launch EC2 Instance**:
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.medium (minimum)
   - Storage: 20GB SSD
   - Security Group: Allow HTTP, HTTPS, SSH

2. **Setup Instance**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv nginx supervisor -y

# Clone application
git clone https://github.com/yourusername/Unsubscriber.git
cd Unsubscriber

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# Configure and start services
# (Follow production deployment steps above)
```

3. **Configure Auto Scaling** (Optional):
```yaml
# autoscaling-config.yaml
AutoScalingGroup:
  MinSize: 1
  MaxSize: 5
  DesiredCapacity: 2
  TargetGroupARNs:
    - !Ref TargetGroup
  HealthCheckType: ELB
  HealthCheckGracePeriod: 300
```

### Google Cloud Platform (App Engine)

1. **Create `app.yaml`**:
```yaml
runtime: python39
instance_class: F2

env_variables:
  FLASK_ENV: "production"
  GOOGLE_CLIENT_ID: "your-client-id"
  GOOGLE_CLIENT_SECRET: "your-client-secret"

handlers:
- url: /static
  static_dir: static
  expiration: "30d"

- url: /.*
  script: auto
  secure: always

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6
```

2. **Deploy**:
```bash
gcloud app deploy
gcloud app browse
```

### Heroku Deployment

1. **Create `Procfile`**:
```
web: gunicorn app:app --timeout 300
```

2. **Create `runtime.txt`**:
```
python-3.10.11
```

3. **Deploy**:
```bash
heroku create your-app-name
heroku config:set FLASK_ENV=production
heroku config:set GOOGLE_CLIENT_ID=your-client-id
heroku config:set GOOGLE_CLIENT_SECRET=your-client-secret
git push heroku main
```

### Azure App Service

1. **Create deployment script**:
```bash
# deploy.sh
az webapp up --name gmail-unsubscriber --resource-group myRG --runtime "PYTHON:3.10"
az webapp config set --name gmail-unsubscriber --resource-group myRG --startup-file "gunicorn app:app"
```

2. **Configure environment**:
```bash
az webapp config appsettings set --name gmail-unsubscriber --resource-group myRG --settings \
  FLASK_ENV=production \
  GOOGLE_CLIENT_ID=your-client-id \
  GOOGLE_CLIENT_SECRET=your-client-secret
```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/auth_status')"

# Start command
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers", "4", "--timeout", "300", "app:app"]
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}
      - GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}
      - FLASK_SECRET_KEY=${FLASK_SECRET_KEY}
    volumes:
      - ./flask_cache:/app/flask_cache
      - ./ml_suite/models:/app/ml_suite/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/auth_status"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t gmail-unsubscriber .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale web=3
```

## Environment Configuration

### Production Environment Variables

Create `.env.production`:
```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_SECRET_KEY=your-very-secure-secret-key-here

# Google OAuth
GOOGLE_CLIENT_ID=your-production-client-id
GOOGLE_CLIENT_SECRET=your-production-client-secret

# Application Settings
PORT=8000
WORKERS=4
TIMEOUT=300
LOG_LEVEL=info

# Cache Configuration
CACHE_TYPE=RedisCache  # For production
CACHE_REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TIMEOUT=21600  # 6 hours

# Model Configuration
MODEL_PATH=/app/final_optimized_model
USE_GPU=True
MAX_BATCH_SIZE=32

# Security
SECURE_COOKIES=True
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
```

### Secrets Management

#### Using Environment Variables
```bash
# Export sensitive data
export GOOGLE_CLIENT_SECRET=$(cat /secure/path/client_secret.txt)
export FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
```

#### Using AWS Secrets Manager
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# In app.py
if os.environ.get('USE_AWS_SECRETS'):
    secrets = get_secret('gmail-unsubscriber-secrets')
    app.config.update(secrets)
```

## Security Considerations

### 1. SSL/TLS Configuration

Always use HTTPS in production:
```bash
# Generate SSL certificate with Let's Encrypt
sudo certbot certonly --nginx -d your-domain.com
```

### 2. Security Headers

Add security headers in Nginx or application:
```python
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app, force_https=True)
```

### 3. Rate Limiting

Implement rate limiting:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/scan_emails')
@limiter.limit("10 per minute")
def scan_emails():
    # ... route logic
```

### 4. Input Validation

Always validate user input:
```python
from flask import request
from werkzeug.exceptions import BadRequest

def validate_scan_params():
    limit = request.args.get('limit', type=int, default=100)
    if limit < 1 or limit > 1000:
        raise BadRequest('Invalid limit parameter')
    return limit
```

### 5. Session Security

Configure secure sessions:
```python
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)
```

## Performance Optimization

### 1. Caching Strategy

#### Redis Cache
```python
# Install: pip install redis flask-caching
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'
```

#### Memcached
```python
# Install: pip install pylibmc flask-caching
app.config['CACHE_TYPE'] = 'MemcachedCache'
app.config['CACHE_MEMCACHED_SERVERS'] = ['127.0.0.1:11211']
```

### 2. Database Optimization

For future database integration:
```python
# Connection pooling
from sqlalchemy import create_engine
engine = create_engine(
    'postgresql://user:pass@host/db',
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

### 3. Static Asset Optimization

#### CDN Integration
```nginx
location /static {
    proxy_pass https://cdn.your-domain.com;
    proxy_cache_valid 200 30d;
    add_header Cache-Control "public, max-age=2592000";
}
```

#### Asset Compression
```nginx
gzip on;
gzip_types text/plain text/css application/json application/javascript;
gzip_min_length 1000;
gzip_comp_level 6;
```

### 4. Model Optimization

#### Model Caching
```python
import functools

@functools.lru_cache(maxsize=1)
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
```

#### Batch Processing
```python
def process_emails_batch(emails, batch_size=32):
    results = []
    for i in range(0, len(emails), batch_size):
        batch = emails[i:i+batch_size]
        predictions = model.predict_batch(batch)
        results.extend(predictions)
    return results
```

## Monitoring and Maintenance

### 1. Application Monitoring

#### Prometheus + Grafana
```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')
```

#### Application Insights (Azure)
```python
from applicationinsights.flask.ext import AppInsights

app.config['APPINSIGHTS_INSTRUMENTATIONKEY'] = 'your-key'
appinsights = AppInsights(app)
```

### 2. Log Management

#### Centralized Logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/gmail-unsubscriber.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

#### ELK Stack Integration
```python
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
app.logger.addHandler(logHandler)
```

### 3. Health Checks

Create health check endpoint:
```python
@app.route('/health')
def health_check():
    checks = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': is_predictor_ready(),
        'cache_connected': cache.get('health_check') is not None
    }
    
    # Check Gmail API
    try:
        service = get_gmail_service()
        checks['gmail_api'] = service is not None
    except:
        checks['gmail_api'] = False
    
    status_code = 200 if all(checks.values()) else 503
    return jsonify(checks), status_code
```

### 4. Backup Strategy

#### Model Backup
```bash
#!/bin/bash
# backup-models.sh
BACKUP_DIR="/backups/models/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
cp -r /app/final_optimized_model $BACKUP_DIR/
cp -r /app/ml_suite/models $BACKUP_DIR/
```

#### Configuration Backup
```bash
# Backup configuration and credentials
tar -czf /backups/config-$(date +%Y%m%d).tar.gz \
  .env \
  client_secret.json \
  nginx.conf \
  gunicorn_config.py
```

### 5. Maintenance Tasks

#### Automated Cleanup
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('cron', day='*/7')
def cleanup_old_cache():
    """Clean cache files older than 7 days"""
    cache_dir = app.config['CACHE_DIR']
    cutoff = time.time() - (7 * 24 * 60 * 60)
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.getmtime(filepath) < cutoff:
            os.remove(filepath)

scheduler.start()
```

## Deployment Checklist

### Pre-Deployment
- [ ] Update Google OAuth redirect URIs
- [ ] Generate production secret key
- [ ] Configure SSL certificates
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Update environment variables
- [ ] Test in staging environment

### Post-Deployment
- [ ] Verify SSL is working
- [ ] Test OAuth flow
- [ ] Check model loading
- [ ] Monitor error logs
- [ ] Test email scanning
- [ ] Verify unsubscribe functions
- [ ] Load test if needed
- [ ] Document deployment details

## Troubleshooting Deployment Issues

### Common Issues

1. **502 Bad Gateway**
   - Check if application is running
   - Verify proxy_pass configuration
   - Check application logs

2. **OAuth Redirect Mismatch**
   - Update redirect URIs in Google Console
   - Ensure HTTPS in production
   - Check environment variables

3. **Model Loading Errors**
   - Verify model files are present
   - Check file permissions
   - Ensure sufficient memory

4. **Performance Issues**
   - Enable caching
   - Use production WSGI server
   - Consider horizontal scaling

## Conclusion

This deployment guide covers various deployment scenarios from simple local setups to complex cloud deployments. Choose the approach that best fits your needs, always prioritizing security and performance in production environments.