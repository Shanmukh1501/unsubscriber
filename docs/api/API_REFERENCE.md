# API Reference

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Email Management APIs](#email-management-apis)
5. [AI/ML APIs](#aiml-apis)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

## Overview

The Gmail Unsubscriber API provides RESTful endpoints for email management, AI operations, and user preferences. All API responses are in JSON format.

### Base URL
```
Development: http://localhost:5000
Production: https://your-domain.com
```

### Common Headers
```http
Content-Type: application/json
Accept: application/json
Cookie: session=<session_cookie>
```

### Response Format
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully"
}
```

### Error Response Format
```json
{
  "error": "Error message",
  "details": "Additional error information",
  "code": "ERROR_CODE"
}
```

## Authentication

### OAuth 2.0 Flow

#### 1. Initiate Login
```http
GET /login
```

**Description**: Initiates Google OAuth 2.0 authentication flow

**Response**: Redirect to Google OAuth consent page

**Example**:
```javascript
window.location.href = '/login';
```

#### 2. OAuth Callback
```http
GET /oauth2callback?code={auth_code}&state={state}
```

**Description**: Handles OAuth callback from Google

**Parameters**:
- `code` (string): Authorization code from Google
- `state` (string): CSRF protection state

**Response**: Redirect to main application

#### 3. Check Authentication Status
```http
GET /api/auth_status
```

**Description**: Returns current authentication status and user info

**Response**:
```json
{
  "isAuthenticated": true,
  "user": {
    "email": "user@example.com",
    "name": "John Doe",
    "picture": "https://..."
  },
  "keep_list": ["sender1@example.com", "sender2@example.com"]
}
```

#### 4. Logout
```http
GET /logout
```

**Description**: Logs out user and revokes Google tokens

**Response**: Redirect to home page

## Core APIs

### Get Application Page
```http
GET /
```

**Description**: Returns the main application HTML page

**Response**: HTML content

## Email Management APIs

### 1. Scan Emails
```http
GET /api/scan_emails
```

**Description**: Scans Gmail inbox for promotional emails using AI classification

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 100 | Maximum emails to process |
| scan_period | string | "180d" | Time period ("30d", "90d", "180d", "all_time") |
| ai_enabled | boolean | false | Enable AI classification |
| ai_threshold | float | 0.75 | AI confidence threshold (0.5-0.95) |
| use_personalized | boolean | true | Use personalized model if available |

**Response**:
```json
[
  {
    "id": "email_sender@example.com",
    "senderName": "Example Company",
    "senderEmail": "sender@example.com",
    "senderDomain": "example.com",
    "exampleSubject": "50% OFF Everything!",
    "count": 15,
    "unsubscribeType": "List-Header (POST)",
    "unsubscribeLink": "https://example.com/unsubscribe",
    "messageIds": ["msg1", "msg2", ...],
    "listIdHeader": "<list.example.com>",
    "ai_classification": {
      "group_label": "UNSUBSCRIBABLE",
      "unsubscribable_percent": 95.5,
      "average_unsub_confidence": 0.92,
      "total_emails_with_ai_prediction": 15
    },
    "heuristic_flags_summary": {
      "has_promotional_keywords": 12,
      "has_all_caps": 8
    }
  }
]
```

### 2. Unsubscribe from Emails
```http
POST /api/unsubscribe_items
```

**Description**: Executes unsubscribe actions for selected senders

**Request Body**:
```json
{
  "items": [
    {
      "id": "sender@example.com",
      "senderEmail": "sender@example.com",
      "unsubscribeType": "List-Header (POST)",
      "unsubscribeLink": "https://example.com/unsubscribe",
      "attempt_auto_get": false
    }
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "id": "sender@example.com",
      "senderEmail": "sender@example.com",
      "status": "Success - POST Request",
      "message": "POST request to https://example.com/unsubscribe successful (Status: 200).",
      "final_unsubscribe_url": "https://example.com/unsubscribe/confirm"
    }
  ]
}
```

**Unsubscribe Types**:
- `List-Header (POST)`: RFC 8058 one-click unsubscribe
- `Mailto`: Email-based unsubscribe
- `List-Header (Link)`: HTTP GET unsubscribe
- `Link in Body`: Unsubscribe link found in email body

### 3. Trash Emails by Sender
```http
POST /api/trash_items
```

**Description**: Moves all emails from specified senders to trash

**Request Body**:
```json
{
  "senderEmails": [
    "sender1@example.com",
    "sender2@example.com"
  ]
}
```

**Response**:
```json
{
  "message": "Smart deletion process finished. Total emails successfully trashed: 45. Total sender operations with failures: 0.",
  "details": [
    {
      "senderEmail": "sender1@example.com",
      "message": "Attempted trashing for 23 emails. Success: 23, Failed: 0.",
      "success_count": 23,
      "fail_count": 0
    }
  ]
}
```

### 4. Preview Sender Emails
```http
GET /api/preview_sender_emails?senderEmail={email}
```

**Description**: Preview recent emails from a specific sender

**Query Parameters**:
- `senderEmail` (string, required): Sender's email address

**Response**:
```json
[
  {
    "id": "message_id_123",
    "subject": "Your weekly newsletter",
    "snippet": "This week's top stories include...",
    "date": "Mon, 22 May 2025 10:30:00 GMT"
  }
]
```

### 5. Manage Keep List
```http
POST /api/manage_keep_list
```

**Description**: Add or remove senders from the protected keep list

**Request Body**:
```json
{
  "senderEmail": "important@example.com",
  "action": "add"
}
```

**Actions**:
- `add`: Add sender to keep list
- `remove`: Remove sender from keep list

**Response**:
```json
{
  "message": "Sender 'important@example.com' added to keep list.",
  "keep_list": ["important@example.com", "other@example.com"]
}
```

## AI/ML APIs

### 1. Get AI Status
```http
GET /api/ai/status
```

**Description**: Returns current AI model status and capabilities

**Response**:
```json
{
  "model_status": "ready",
  "base_model_name": "microsoft/deberta-v3-small",
  "last_trained_date": "2025-05-26",
  "data_prepared": true,
  "model_info": {
    "accuracy": 1.0,
    "precision": 1.0,
    "f1_score": 1.0,
    "false_positives": 0,
    "training_duration": "7.5 hours",
    "training_samples": 20000,
    "model_size": "567 MB"
  },
  "personalization": {
    "has_personalized_model": false,
    "feedback_stats": {
      "total_feedback": 0,
      "unsubscribable_feedback": 0,
      "important_feedback": 0
    }
  }
}
```

### 2. Prepare Training Data
```http
POST /api/ai/prepare_public_data
```

**Description**: Initiates data preparation for AI model training

**Response**:
```json
{
  "message": "Data preparation process initiated. Check status for progress.",
  "status_endpoint": "/api/ai/status"
}
```

### 3. Train AI Model
```http
POST /api/ai/train_model
```

**Description**: Starts AI model training process

**Response**:
```json
{
  "message": "Model training process initiated. Check status for progress.",
  "status_endpoint": "/api/ai/status"
}
```

### 4. Submit AI Feedback
```http
POST /api/ai/feedback
```

**Description**: Submit feedback on AI predictions for model improvement

**Request Body**:
```json
{
  "email_id": "msg_123",
  "email_text": "Subject: Sale! 50% off...",
  "predicted_label": "UNSUBSCRIBABLE",
  "predicted_confidence": 0.92,
  "user_feedback": "correct",
  "session_id": "session_123"
}
```

**User Feedback Values**:
- `correct`: Prediction was accurate
- `incorrect`: Prediction was wrong
- `unsure`: User is uncertain

**Response**:
```json
{
  "success": true,
  "message": "Feedback recorded successfully",
  "statistics": {
    "total_entries": 25,
    "correct_predictions": 23,
    "incorrect_predictions": 2
  }
}
```

## Error Handling

### HTTP Status Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (async operation started) |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 429 | Rate Limited |
| 500 | Internal Server Error |

### Common Error Responses

#### Authentication Required
```json
{
  "error": "Authentication required or failed. Please sign in again.",
  "code": "AUTH_REQUIRED"
}
```

#### Invalid Request
```json
{
  "error": "Missing required parameter: senderEmail",
  "code": "INVALID_REQUEST"
}
```

#### Gmail API Error
```json
{
  "error": "Gmail API Error: User rate limit exceeded",
  "code": "GMAIL_API_ERROR",
  "details": "Try again in 60 seconds"
}
```

## Rate Limiting

### Gmail API Limits
- **Per-user limit**: 250 quota units per user per second
- **Daily limit**: 1,000,000,000 quota units per day

### Application Rate Limits
| Endpoint | Limit | Window |
|----------|-------|--------|
| /api/scan_emails | 10 requests | 1 minute |
| /api/unsubscribe_items | 100 requests | 1 minute |
| /api/trash_items | 50 requests | 1 minute |

### Rate Limit Response
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "code": "RATE_LIMITED"
}
```

## Examples

### JavaScript/Fetch Examples

#### Scan Emails with AI
```javascript
async function scanEmails() {
  const response = await fetch('/api/scan_emails?' + new URLSearchParams({
    limit: 100,
    scan_period: '90d',
    ai_enabled: true,
    ai_threshold: 0.75
  }));
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const mailers = await response.json();
  console.log(`Found ${mailers.length} potential unsubscribe candidates`);
  return mailers;
}
```

#### Unsubscribe from Multiple Senders
```javascript
async function unsubscribeMultiple(items) {
  const response = await fetch('/api/unsubscribe_items', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ items })
  });
  
  const result = await response.json();
  
  result.results.forEach(item => {
    if (item.status.includes('Success')) {
      console.log(`✓ Unsubscribed from ${item.senderEmail}`);
    } else {
      console.log(`✗ Failed to unsubscribe from ${item.senderEmail}: ${item.message}`);
    }
  });
  
  return result;
}
```

#### Add to Keep List
```javascript
async function addToKeepList(senderEmail) {
  const response = await fetch('/api/manage_keep_list', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      senderEmail: senderEmail,
      action: 'add'
    })
  });
  
  const result = await response.json();
  console.log(result.message);
  return result.keep_list;
}
```

### Python Examples

#### Using requests library
```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# Scan emails
def scan_emails(session_cookie):
    response = requests.get(
        f"{BASE_URL}/api/scan_emails",
        params={
            "limit": 100,
            "ai_enabled": "true",
            "ai_threshold": 0.75
        },
        cookies={"session": session_cookie}
    )
    return response.json()

# Unsubscribe
def unsubscribe(items, session_cookie):
    response = requests.post(
        f"{BASE_URL}/api/unsubscribe_items",
        json={"items": items},
        cookies={"session": session_cookie}
    )
    return response.json()
```

### cURL Examples

#### Check Authentication Status
```bash
curl -X GET http://localhost:5000/api/auth_status \
  -H "Cookie: session=your_session_cookie"
```

#### Scan Emails
```bash
curl -X GET "http://localhost:5000/api/scan_emails?limit=50&ai_enabled=true" \
  -H "Cookie: session=your_session_cookie"
```

#### Unsubscribe
```bash
curl -X POST http://localhost:5000/api/unsubscribe_items \
  -H "Content-Type: application/json" \
  -H "Cookie: session=your_session_cookie" \
  -d '{
    "items": [{
      "id": "sender@example.com",
      "senderEmail": "sender@example.com",
      "unsubscribeType": "List-Header (POST)",
      "unsubscribeLink": "https://example.com/unsubscribe"
    }]
  }'
```

## Best Practices

1. **Always check authentication** before making API calls
2. **Handle rate limits** with exponential backoff
3. **Validate responses** before processing
4. **Use appropriate timeouts** for long-running operations
5. **Log errors** for debugging
6. **Cache responses** when appropriate
7. **Batch operations** to reduce API calls

## WebSocket Support (Future)

Future versions may include WebSocket support for real-time updates:
```javascript
// Future WebSocket implementation
const ws = new WebSocket('ws://localhost:5000/ws');
ws.on('scan_progress', (data) => {
  console.log(`Scan progress: ${data.progress}%`);
});
```

---

For more information on specific use cases, see the [User Guide](../user-guide/USER_GUIDE.md) or [Development Guide](../development/DEVELOPMENT_GUIDE.md).