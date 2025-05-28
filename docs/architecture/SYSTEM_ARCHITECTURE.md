# System Architecture

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Security Architecture](#security-architecture)
6. [ML Architecture](#ml-architecture)
7. [API Architecture](#api-architecture)
8. [Database and Storage](#database-and-storage)
9. [Deployment Architecture](#deployment-architecture)

## Overview

The Gmail Unsubscriber system follows a modular, layered architecture that separates concerns between the web interface, business logic, machine learning components, and external integrations. The architecture prioritizes security, performance, and maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Browser                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Single Page Application                  │   │
│  │              (HTML + Tailwind CSS + Vanilla JS)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────┬───────────────────────────────────────────┘
                     │ HTTPS
┌────────────────────▼───────────────────────────────────────────┐
│                      Flask Web Server                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │   Routes    │  │   Session    │  │   Error Handler   │    │
│  │  Handler    │  │  Management  │  │   & Logging       │    │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────────┘    │
│         │                 │                   │                 │
│  ┌──────▼─────────────────▼──────────────────▼────────────┐   │
│  │              Business Logic Layer                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │   │
│  │  │   Email     │  │ Unsubscribe │  │    Keep      │  │   │
│  │  │  Scanner    │  │  Manager    │  │    List      │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬──────────────────┬──────────────────┬────────────────┘
         │                  │                  │
┌────────▼────────┐ ┌───────▼──────┐  ┌───────▼────────┐
│   ML Suite      │ │  Gmail API   │  │  Flask Cache   │
│  ┌──────────┐   │ │   Client     │  │                │
│  │Predictor │   │ └──────────────┘  └────────────────┘
│  │ Module   │   │
│  └──────────┘   │
│  ┌──────────┐   │
│  │  Model   │   │
│  │ Storage  │   │
│  └──────────┘   │
└─────────────────┘
```

## Component Architecture

### 1. Web Interface Layer

**Single Page Application (SPA)**
- **Technology**: HTML5, Tailwind CSS, Vanilla JavaScript
- **Responsibilities**:
  - User authentication flow
  - Email list display and management
  - Real-time status updates
  - Theme switching (dark/light mode)
  - Batch operation handling

**Key Files**:
- `unsubscriber.html`: Main application interface
- Embedded CSS: Tailwind utilities and custom styles
- Embedded JavaScript: Application logic and API communication

### 2. Application Server Layer

**Flask Application (`app.py`)**
- **Core Components**:
  - Route handlers for all endpoints
  - Session management with secure cookies
  - OAuth 2.0 flow implementation
  - Error handling and logging
  - Request validation and sanitization

**Key Features**:
- Thread-safe operation
- Graceful error handling
- Comprehensive logging
- CORS support for API access

### 3. Business Logic Layer

**Email Scanner**
- Fetches emails from Gmail API
- Applies heuristic filtering
- Integrates AI predictions
- Groups emails by sender
- Manages API rate limits

**Unsubscribe Manager**
- Implements multiple unsubscribe methods
- Handles List-Unsubscribe headers
- Processes mailto links
- Manages HTTP requests
- Tracks operation status

**Keep List Manager**
- Maintains protected sender list
- Session-based storage
- Real-time updates
- Import/export functionality

### 4. Machine Learning Suite

**Directory Structure**:
```
ml_suite/
├── __init__.py         # Package initialization
├── config.py           # ML configuration
├── predictor.py        # Inference engine
├── data_preparator.py  # Training data preparation
├── model_trainer.py    # Model training logic
├── utils.py           # Shared utilities
├── task_utils.py      # Async task management
└── advanced_predictor.py # Ensemble predictions
```

**Key Components**:

**Predictor Module (`predictor.py`)**
- Loads pre-trained transformer models
- Manages model lifecycle
- Provides inference API
- Handles confidence calibration
- Supports personalized models

**Data Preparation (`data_preparator.py`)**
- Downloads public datasets
- Creates synthetic training data
- Handles data cleaning and normalization
- Manages fallback strategies

**Model Trainer (`model_trainer.py`)**
- Fine-tunes transformer models
- Implements custom loss functions
- Manages training lifecycle
- Evaluates model performance
- Saves trained models

### 5. External Integrations

**Gmail API Integration**
- OAuth 2.0 authentication
- Message fetching with pagination
- Batch operations support
- Rate limit handling
- Error recovery mechanisms

**Cache Layer**
- FileSystem-based caching
- 6-hour default timeout
- Message detail caching
- API response caching
- Session data caching

## Data Flow

### 1. Authentication Flow
```
User → Login → Google OAuth → Callback → Session Creation → Redirect
```

### 2. Email Scanning Flow
```
User Request → API Call → Gmail API → Email Fetch → 
AI Classification → Grouping → Response
```

### 3. Unsubscribe Flow
```
User Selection → Method Detection → Action Execution →
Status Update → Result Display
```

### 4. ML Prediction Flow
```
Email Text → Preprocessing → Tokenization → Model Inference →
Confidence Calibration → Result
```

## Security Architecture

### Authentication & Authorization
- **OAuth 2.0**: Secure Google authentication
- **Session Management**: Server-side session storage
- **CSRF Protection**: State parameter validation
- **Token Refresh**: Automatic token renewal

### Data Protection
- **Local Processing**: No external data transmission
- **Secure Storage**: Encrypted session data
- **SSL/TLS**: HTTPS enforcement in production
- **Input Validation**: Comprehensive sanitization

### API Security
- **Rate Limiting**: Prevent abuse
- **Request Validation**: Parameter checking
- **Error Masking**: Hide sensitive information
- **Logging**: Security event tracking

## ML Architecture

### Model Architecture
- **Base Model**: Microsoft DeBERTa v3 Small
- **Architecture**: Transformer-based sequence classification
- **Parameters**: ~140M parameters
- **Input**: Tokenized email text (max 512 tokens)
- **Output**: Binary classification (Important/Unsubscribable)

### Training Pipeline
1. **Data Collection**: Public datasets + synthetic data
2. **Preprocessing**: Text cleaning, tokenization
3. **Training**: Fine-tuning with custom objectives
4. **Evaluation**: Multiple metrics tracking
5. **Deployment**: Model serialization and optimization

### Inference Pipeline
1. **Text Input**: Email subject and body
2. **Preprocessing**: Cleaning and truncation
3. **Tokenization**: Convert to model inputs
4. **Inference**: Forward pass through model
5. **Post-processing**: Confidence calibration
6. **Result**: Classification with confidence score

## API Architecture

### RESTful Endpoints

**Authentication APIs**
- `GET /`: Main application page
- `GET /login`: Initiate OAuth flow
- `GET /oauth2callback`: Handle OAuth callback
- `GET /logout`: Clear session and revoke tokens
- `GET /api/auth_status`: Check authentication status

**Email Management APIs**
- `GET /api/scan_emails`: Scan and classify emails
- `POST /api/unsubscribe_items`: Execute unsubscribe actions
- `POST /api/trash_items`: Delete emails by sender
- `GET /api/preview_sender_emails`: Preview sender's emails

**Keep List APIs**
- `POST /api/manage_keep_list`: Add/remove protected senders

**AI/ML APIs**
- `GET /api/ai/status`: Check AI model status
- `POST /api/ai/prepare_public_data`: Prepare training data
- `POST /api/ai/train_model`: Train AI model
- `POST /api/ai/feedback`: Submit prediction feedback

### Request/Response Format
- **Content-Type**: application/json
- **Authentication**: Session-based
- **Error Format**: Standardized error responses
- **Status Codes**: RESTful conventions

## Database and Storage

### Session Storage
- **Technology**: Server-side Flask sessions
- **Content**: User credentials, preferences, keep list
- **Lifetime**: Configurable timeout
- **Security**: Encrypted cookies

### Cache Storage
- **Technology**: FileSystem cache
- **Location**: `./flask_cache/`
- **Content**: API responses, message details
- **Eviction**: Time-based (6 hours default)

### Model Storage
- **Location**: `./final_optimized_model/`
- **Format**: Hugging Face model format
- **Components**: Model weights, tokenizer, config
- **Size**: ~567 MB

### Task Status Storage
- **Location**: `./ml_suite/task_status/`
- **Format**: JSON files
- **Content**: Training progress, results
- **Cleanup**: Manual or automated

## Deployment Architecture

### Development Environment
```
Local Machine
├── Python Virtual Environment
├── Flask Development Server
├── Local File Storage
└── Direct Gmail API Access
```

### Production Environment (Recommended)
```
Production Server
├── WSGI Server (Gunicorn/uWSGI)
├── Reverse Proxy (Nginx)
├── SSL Certificate
├── Process Manager (systemd/supervisor)
├── Monitoring (Prometheus/Grafana)
└── Log Aggregation
```

### Docker Deployment (Optional)
```
Docker Container
├── Python Runtime
├── Application Code
├── Model Files
├── Volume Mounts
│   ├── Cache Directory
│   ├── Model Directory
│   └── Log Directory
└── Environment Variables
```

### Cloud Deployment Options
- **AWS**: EC2 with Auto Scaling Groups
- **Google Cloud**: App Engine or Compute Engine
- **Azure**: App Service or Virtual Machines
- **Heroku**: With custom buildpacks

## Performance Considerations

### Optimization Strategies
1. **Caching**: Aggressive caching of API responses
2. **Batch Processing**: Group API calls
3. **Lazy Loading**: Load data on demand
4. **Connection Pooling**: Reuse HTTP connections
5. **GPU Acceleration**: Optional CUDA support

### Scalability Patterns
1. **Horizontal Scaling**: Multiple app instances
2. **Load Balancing**: Distribute requests
3. **Queue System**: Async task processing
4. **CDN**: Static asset delivery
5. **Database Sharding**: Future consideration

### Monitoring Points
1. **Application Metrics**: Response times, error rates
2. **System Metrics**: CPU, memory, disk usage
3. **ML Metrics**: Inference time, accuracy
4. **API Metrics**: Rate limits, quotas
5. **User Metrics**: Active sessions, operations

## Conclusion

The Gmail Unsubscriber architecture is designed for modularity, security, and performance. The separation of concerns allows for independent scaling and maintenance of components while maintaining a cohesive user experience. The architecture supports both simple deployments for personal use and complex deployments for organizational use.