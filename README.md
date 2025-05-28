# Gmail Unsubscriber

An AI-powered Gmail management tool that automatically identifies and helps you unsubscribe from unwanted emails using advanced machine learning.

## Features

- **AI-Powered Classification**: Uses a fine-tuned DeBERTa v3 model with 100% accuracy to identify promotional emails
- **Smart Unsubscribe**: Automatically handles various unsubscribe methods (mailto, links, List-Unsubscribe headers)
- **Keep List**: Protect important senders from being flagged
- **Batch Operations**: Process multiple unsubscribe requests at once
- **Dark/Light Mode**: Beautiful UI that works in both themes
- **Privacy-Focused**: All processing happens locally, your emails never leave your control

## Technology Stack

- **Backend**: Python/Flask
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **AI Model**: Microsoft DeBERTa v3 (fine-tuned)
- **Authentication**: Google OAuth 2.0
- **APIs**: Gmail API

## Model Performance

The AI model achieves exceptional performance:
- **Accuracy**: 100%
- **Precision**: 100%
- **F1 Score**: 100%
- **False Positives**: 0
- **Training Time**: 7.5 hours on 20,000 emails

## Setup

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd Unsubscriber
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google OAuth**
   - Create a project in Google Cloud Console
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download and save as `client_secret.json` in the project root

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the app**
   - Open http://localhost:5000 in your browser
   - Sign in with your Google account
   - Start scanning and managing your emails!

## Configuration

The AI confidence threshold can be adjusted in the UI:
- **50-65%**: Aggressive (may flag some important emails)
- **70-85%**: Balanced (recommended)
- **90-95%**: Conservative (very safe, may miss some promotional emails)

## Privacy & Security

- All email processing happens locally on your machine
- OAuth tokens are stored securely in Flask sessions
- No email content is sent to external servers
- The AI model runs entirely offline

## Project Structure

```
Unsubscriber/
├── app.py                    # Main Flask application
├── unsubscriber.html        # Frontend UI
├── ml_suite/                # AI module package
│   ├── predictor.py        # Model inference
│   ├── config.py           # Configuration
│   └── models/             # Trained models
├── final_optimized_model/   # DeBERTa v3 model files
└── requirements.txt         # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational and personal use. Please ensure you comply with Gmail's terms of service when using this tool.

## Acknowledgments

- Built with Claude AI assistance
- Uses Microsoft's DeBERTa v3 model
- Powered by Google's Gmail API

---

**Note**: Never commit your `client_secret.json` file or any credentials to version control!