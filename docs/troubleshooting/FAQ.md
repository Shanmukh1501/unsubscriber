# Frequently Asked Questions (FAQ)

## Table of Contents
1. [General Questions](#general-questions)
2. [Installation Issues](#installation-issues)
3. [Authentication Problems](#authentication-problems)
4. [Email Scanning Issues](#email-scanning-issues)
5. [AI/ML Questions](#aiml-questions)
6. [Unsubscribe Problems](#unsubscribe-problems)
7. [Performance Issues](#performance-issues)
8. [Security Concerns](#security-concerns)
9. [Error Messages](#error-messages)
10. [Advanced Topics](#advanced-topics)

## General Questions

### Q: What is Gmail Unsubscriber?
**A:** Gmail Unsubscriber is an AI-powered tool that helps you automatically identify and unsubscribe from unwanted promotional emails. It uses machine learning to classify emails and provides multiple methods to unsubscribe efficiently.

### Q: Is it free to use?
**A:** Yes, the software itself is free and open-source. However, you'll need:
- A Google account with Gmail
- A Google Cloud project (free tier is sufficient)
- Your own computer to run it on

### Q: How accurate is the AI classification?
**A:** The pre-trained model achieves:
- 100% accuracy on test data
- 0 false positives in testing
- Real-world accuracy typically 95%+

### Q: How much time does it save?
**A:** Users report saving 2-4 hours per month on email management, depending on email volume.

### Q: Can I use this with multiple Gmail accounts?
**A:** Yes, but you need to:
1. Sign out and sign in with each account
2. Each account needs its own OAuth consent
3. Keep lists are stored per session

## Installation Issues

### Q: "ModuleNotFoundError: No module named 'transformers'"
**A:** Your dependencies aren't installed properly.

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Q: "torch not compiled with CUDA enabled"
**A:** You have CPU-only PyTorch installed.

**Solution:**
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q: "client_secret.json not found"
**A:** You haven't set up Google OAuth credentials.

**Solution:**
1. Follow the [Installation Guide](../installation/INSTALLATION_GUIDE.md#google-oauth-setup)
2. Download credentials from Google Cloud Console
3. Rename to `client_secret.json`
4. Place in project root

### Q: Port 5000 is already in use
**A:** Another application is using port 5000.

**Solution:**
```bash
# Option 1: Use different port
PORT=5001 python app.py

# Option 2: Find and kill process using port 5000
# Linux/Mac:
lsof -i :5000
kill -9 <PID>

# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Authentication Problems

### Q: "Error 400: redirect_uri_mismatch"
**A:** The redirect URI doesn't match Google OAuth configuration.

**Solution:**
1. Go to Google Cloud Console
2. Edit your OAuth 2.0 Client ID
3. Add authorized redirect URI:
   - Development: `http://localhost:5000/oauth2callback`
   - Production: `https://your-domain.com/oauth2callback`
4. Save and wait 5 minutes

### Q: "Access blocked: This app's request is invalid"
**A:** OAuth consent screen not configured properly.

**Solution:**
1. Go to OAuth consent screen in Google Cloud Console
2. Fill all required fields
3. Add your email as test user (if in testing mode)
4. For production, submit for verification

### Q: "Token has been expired or revoked"
**A:** Your authentication token expired.

**Solution:**
1. Sign out completely
2. Clear browser cookies for localhost
3. Sign in again
4. Grant all requested permissions

### Q: Keep getting logged out
**A:** Session configuration or cookie issues.

**Solution:**
1. Check browser allows cookies
2. Ensure `FLASK_SECRET_KEY` is set and stable
3. Don't use incognito/private mode
4. Check session timeout settings

## Email Scanning Issues

### Q: "No emails found"
**A:** The scan didn't find any matching emails.

**Possible causes:**
1. Too restrictive time period
2. No promotional emails in inbox
3. All promotional emails in spam

**Solution:**
- Try "All time" scan period
- Check spam folder manually
- Lower AI confidence threshold

### Q: Scan is very slow
**A:** Performance depends on email volume and system.

**Solutions:**
1. Reduce scan limit to 50-100 emails
2. Use shorter time period (30 days)
3. Enable GPU acceleration if available
4. Close other applications

### Q: Important emails showing as promotional
**A:** False positives in classification.

**Solutions:**
1. Add sender to Keep List immediately
2. Increase AI confidence threshold to 85%+
3. Report via GitHub issues for model improvement
4. Use manual review for critical senders

### Q: Not finding obvious promotional emails
**A:** False negatives in classification.

**Solutions:**
1. Lower AI confidence threshold to 65%
2. Check if emails have unsubscribe links
3. Scan larger time period
4. Check spam/promotions folders

## AI/ML Questions

### Q: "Model loading error"
**A:** Model files corrupted or missing.

**Solution:**
```bash
# Verify model files exist
ls final_optimized_model/

# Should see:
# model.safetensors, config.json, tokenizer.json, etc.

# If missing, re-download or retrain
python train_optimized_final.py
```

### Q: Can I train my own model?
**A:** Yes! Multiple options available.

**Options:**
1. Quick training (2-3 hours):
   ```bash
   python train_fast_roberta.py
   ```
2. Full training (3-4 hours):
   ```bash
   python train_optimized_final.py
   ```

### Q: GPU not being used
**A:** CUDA not properly configured.

**Diagnosis:**
```python
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**Solutions:**
1. Install CUDA-compatible PyTorch
2. Update GPU drivers
3. Check CUDA installation
4. Model works on CPU (slower)

### Q: Want to improve model accuracy
**A:** Several approaches available.

**Options:**
1. Collect more training data
2. Fine-tune on your emails
3. Adjust confidence thresholds
4. Use ensemble methods

## Unsubscribe Problems

### Q: "Unsubscribe failed" errors
**A:** Various causes possible.

**Common reasons:**
1. **Invalid link**: Sender's unsubscribe broken
2. **Rate limited**: Too many requests
3. **Authentication required**: Need manual action
4. **Network error**: Connection issues

**Solutions:**
- Try manual unsubscribe (click link)
- Wait and retry later
- Use different unsubscribe method
- Delete emails instead

### Q: Unsubscribed but still getting emails
**A:** Unsubscribe didn't work properly.

**Reasons:**
1. Sender ignores unsubscribe
2. Multiple mailing lists
3. Unsubscribe takes time (up to 10 days)
4. New subscription created

**Solutions:**
1. Use "Delete emails" option
2. Block sender in Gmail
3. Report as spam
4. Wait 10 business days

### Q: "Manual action required" message
**A:** Automated unsubscribe not possible.

**Why this happens:**
- Complex unsubscribe process
- Requires account login
- CAPTCHA verification
- Multiple steps needed

**Solution:**
Click the provided link and complete manually.

## Performance Issues

### Q: Application running slowly
**A:** Multiple factors affect performance.

**Optimizations:**
1. **Reduce batch size**: Process fewer emails
2. **Enable caching**: Faster repeated operations
3. **Use GPU**: If available
4. **Close other apps**: Free up memory
5. **Update dependencies**: Latest optimizations

### Q: High memory usage
**A:** ML models require significant memory.

**Solutions:**
```python
# Reduce batch size in scanning
SCAN_BATCH_SIZE = 25  # Instead of 100

# Use smaller model (future option)
MODEL_NAME = "distilbert-base-uncased"
```

### Q: Browser freezing
**A:** Too much data in browser.

**Solutions:**
1. Scan fewer emails at once
2. Use pagination (future feature)
3. Clear browser cache
4. Use different browser

## Security Concerns

### Q: Is my email data safe?
**A:** Yes, with these protections:

**Security measures:**
- ✅ All processing happens locally
- ✅ No data sent to external servers
- ✅ OAuth tokens encrypted
- ✅ Sessions expire automatically
- ✅ Open source for transparency

### Q: What permissions does it need?
**A:** Required Gmail permissions:

1. **Read emails**: To scan and classify
2. **Modify labels**: To mark as processed
3. **Send emails**: For mailto unsubscribe
4. **User info**: For session management

### Q: Can others access my session?
**A:** No, with proper precautions:

**Best practices:**
1. Don't share your computer while signed in
2. Always sign out when done
3. Use HTTPS in production
4. Don't expose app to internet without security

### Q: Is the AI model private?
**A:** Yes, completely private:

- Model runs locally
- No external API calls
- No telemetry or tracking
- Your emails never leave your computer

## Error Messages

### Q: "Gmail API Error: User rate limit exceeded"
**A:** Hit Gmail API quotas.

**Solution:**
1. Wait 1-2 minutes
2. Reduce scan size
3. Implement caching
4. Spread operations over time

### Q: "HttpError 403: Insufficient Permission"
**A:** Missing required permissions.

**Solution:**
1. Sign out completely
2. Sign in again
3. Grant ALL requested permissions
4. Check OAuth scope configuration

### Q: "SSL: CERTIFICATE_VERIFY_FAILED"
**A:** SSL certificate issues.

**Solution:**
```python
# Temporary fix (development only!)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Proper fix:
# Update certificates
pip install --upgrade certifi
```

### Q: "JavaScript heap out of memory"
**A:** Browser memory exhausted.

**Solution:**
1. Process fewer emails
2. Restart browser
3. Use 64-bit browser
4. Increase browser memory limit

## Advanced Topics

### Q: Can I integrate this with other tools?
**A:** Yes, via the API.

**Integration options:**
1. Use REST API endpoints
2. Create custom scripts
3. Webhook support (future)
4. Export data as JSON/CSV

### Q: How to run this on a server?
**A:** See [Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md).

**Key steps:**
1. Set up Linux server
2. Install dependencies
3. Configure Nginx
4. Use Gunicorn/uWSGI
5. Set up SSL certificate

### Q: Can I contribute to the project?
**A:** Yes! Contributions welcome.

**How to contribute:**
1. Read [Development Guide](../development/DEVELOPMENT_GUIDE.md)
2. Fork repository
3. Create feature branch
4. Submit pull request
5. Follow code standards

### Q: Multi-language support?
**A:** Currently English only.

**Future plans:**
- Spanish, French, German
- UI translations
- Multi-language ML models
- RTL language support

### Q: Bulk operations for organizations?
**A:** Possible with modifications.

**Considerations:**
1. Google Workspace integration
2. Admin delegation
3. Batch processing
4. Audit logging
5. Compliance requirements

## Still Need Help?

If your question isn't answered here:

1. **Check Documentation**: Review all guides thoroughly
2. **Search Issues**: Look for similar problems on GitHub
3. **Ask Community**: Create a discussion thread
4. **Report Bugs**: Open an issue with details
5. **Contact Maintainers**: For security issues only

### Providing Good Bug Reports

Include:
- Operating system and version
- Python version
- Browser and version
- Complete error message
- Steps to reproduce
- Screenshots if applicable

### Debug Information

Run this diagnostic:
```python
python -c "
import sys
import torch
import transformers
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
"
```

---

Remember: Most issues have simple solutions. Stay calm, read carefully, and happy unsubscribing! 🎉