# Gmail Unsubscriber - Complete Documentation

Welcome to the comprehensive documentation for the Gmail Unsubscriber project. This documentation provides everything you need to understand, build, deploy, and maintain this AI-powered email management system.

## 📚 Documentation Index

### 1. [Project Overview](overview/PROJECT_OVERVIEW.md)
Complete introduction to the Gmail Unsubscriber system, its features, capabilities, and use cases.

### 2. [System Architecture](architecture/SYSTEM_ARCHITECTURE.md)
Detailed technical architecture including system design, component interactions, and data flow.

### 3. [Installation Guide](installation/INSTALLATION_GUIDE.md)
Step-by-step instructions to set up the project from scratch, including all dependencies and prerequisites.

### 4. [API Documentation](api/API_REFERENCE.md)
Complete API reference for all endpoints, request/response formats, and integration guidelines.

### 5. [ML Model Documentation](ml-model/ML_MODEL_GUIDE.md)
Comprehensive guide to the AI/ML components, including model architecture, training process, and performance metrics.

### 6. [User Guide](user-guide/USER_GUIDE.md)
End-user documentation for using the Gmail Unsubscriber application effectively.

### 7. [Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)
Production deployment instructions for various platforms and environments.

### 8. [Development Guide](development/DEVELOPMENT_GUIDE.md)
Developer documentation for contributing to the project, coding standards, and best practices.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd Unsubscriber
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google OAuth credentials**
   - Follow the [Installation Guide](installation/INSTALLATION_GUIDE.md#google-oauth-setup)

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open http://localhost:5000 in your browser

## 🔑 Key Features

- **AI-Powered Email Classification**: Uses a fine-tuned DeBERTa v3 model with 100% accuracy
- **Smart Unsubscribe**: Handles multiple unsubscribe methods automatically
- **Privacy-Focused**: All processing happens locally
- **Beautiful UI**: Modern, responsive interface with dark/light mode
- **Batch Operations**: Process multiple emails efficiently
- **Keep List**: Protect important senders from being flagged

## 📋 Prerequisites

- Python 3.8 or higher
- Google Account with Gmail API access
- 4GB+ RAM (8GB recommended for model training)
- Optional: NVIDIA GPU for faster AI processing

## 📖 Documentation Conventions

Throughout this documentation:

- 📁 **File paths** are shown relative to the project root
- 💻 **Code blocks** include the appropriate language syntax highlighting
- ⚠️ **Important notes** are highlighted with warning symbols
- 💡 **Tips and best practices** are marked with lightbulb icons
- 🔗 **Cross-references** link to related documentation sections

## 🤝 Contributing

Please refer to the [Development Guide](development/DEVELOPMENT_GUIDE.md) for information on:
- Setting up a development environment
- Code style and standards
- Testing procedures
- Submitting pull requests

## 📝 License

This project is provided for educational and personal use. Please ensure compliance with Gmail's terms of service when using this tool.

## 🙏 Acknowledgments

- Built with Claude AI assistance
- Uses Microsoft's DeBERTa v3 model
- Powered by Google's Gmail API
- UI framework: Tailwind CSS

---

For detailed information on any topic, please refer to the specific documentation sections linked above.