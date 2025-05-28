# User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [User Interface Overview](#user-interface-overview)
3. [Scanning Emails](#scanning-emails)
4. [Managing Unsubscriptions](#managing-unsubscriptions)
5. [Keep List Management](#keep-list-management)
6. [AI Features](#ai-features)
7. [Batch Operations](#batch-operations)
8. [Settings and Preferences](#settings-and-preferences)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Common Use Cases](#common-use-cases)

## Getting Started

### First-Time Setup

1. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000` (or your deployment URL)
   - You'll see the Gmail Unsubscriber welcome page

2. **Sign In with Google**
   - Click the "Sign in with Google" button
   - Select your Google account
   - Review and accept the permissions:
     - Read your emails
     - Modify email labels
     - Send emails on your behalf
   - You'll be redirected back to the application

3. **Initial Configuration**
   - The app will load with default settings
   - AI features are enabled by default
   - Confidence threshold is set to 75%

## User Interface Overview

### Main Dashboard

The interface consists of several key sections:

```
┌─────────────────────────────────────────────────┐
│  Gmail Unsubscriber  [User Avatar] Sign Out    │
├─────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌────────────────────┐   │
│  │  Scan Controls  │  │   AI Settings      │   │
│  └─────────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────┤
│  Search: [_________]  Filter: [▼]  Sort: [▼]   │
├─────────────────────────────────────────────────┤
│  Select All □  Unsubscribe Selected  Delete    │
├─────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────┐   │
│  │         Email List Results              │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Key UI Elements

1. **Navigation Bar**
   - Application title
   - User profile picture and email
   - Sign out button
   - Theme toggle (auto-detects system preference)

2. **Control Panel**
   - Scan button with customization options
   - AI toggle and settings
   - Progress indicators

3. **Results Area**
   - Email sender cards
   - Unsubscribe options
   - Keep list controls

## Scanning Emails

### Basic Scan

1. Click the **"Scan Emails"** button
2. The system will:
   - Connect to your Gmail account
   - Fetch recent emails (last 6 months by default)
   - Analyze them using AI
   - Group by sender
   - Display results

### Advanced Scan Options

Click the settings icon next to "Scan Emails" to access:

- **Scan Period**:
  - Last 30 days
  - Last 90 days
  - Last 180 days (default)
  - All time

- **Email Limit**:
  - 50 emails
  - 100 emails (default)
  - 200 emails
  - 500 emails

- **AI Options**:
  - Enable/disable AI classification
  - Adjust confidence threshold (50-95%)
  - Use personalized model (if available)

### Understanding Scan Results

Each result card shows:

```
┌─────────────────────────────────────────┐
│ □ Company Name                          │
│   sender@example.com                    │
│   📧 15 emails | 🔗 List-Header (POST) │
│   Subject: "Latest deals and offers..." │
│   AI: 92% Promotional                   │
│   [Keep] [Preview] [Unsubscribe]       │
└─────────────────────────────────────────┘
```

**Elements Explained**:
- **Checkbox**: Select for batch operations
- **Company Name**: Sender's display name
- **Email Address**: Actual sender email
- **Email Count**: Number of emails from this sender
- **Unsubscribe Type**: Method available for unsubscribing
- **Subject Example**: Recent email subject
- **AI Confidence**: How certain the AI is about classification
- **Action Buttons**: Available actions for this sender

## Managing Unsubscriptions

### Unsubscribe Methods

The app supports multiple unsubscribe methods:

1. **List-Header (POST)** ✨ Best
   - One-click unsubscribe
   - RFC 8058 compliant
   - Automatic, no user action needed

2. **Mailto** 📧 Good
   - Sends unsubscribe email automatically
   - Uses your Gmail account
   - Includes proper subject line

3. **List-Header (Link)** 🔗 Manual
   - Provides unsubscribe link
   - Option to auto-visit (experimental)
   - May require manual confirmation

4. **Link in Body** 🔍 Manual
   - Found by scanning email content
   - Requires manual action
   - Click to open in new tab

### Single Unsubscribe

1. Find the sender you want to unsubscribe from
2. Click the **"Unsubscribe"** button
3. Depending on the method:
   - **Automatic**: Wait for confirmation
   - **Link**: New tab opens, follow instructions
   - **Email**: Automatically sent, check status

### Batch Unsubscribe

1. Select multiple senders using checkboxes
2. Click **"Unsubscribe Selected"** at the top
3. Confirm the action in the dialog
4. Monitor progress in the status area

### Status Indicators

- ✅ **Success**: Unsubscribe completed
- ⏳ **Pending**: Action in progress
- ⚠️ **Manual Required**: Need user action
- ❌ **Failed**: Action failed (see details)

## Keep List Management

### What is the Keep List?

The Keep List protects important senders from being flagged as promotional, even if they send marketing-style emails.

### Adding to Keep List

1. **From Scan Results**:
   - Click the **"Keep"** button on any sender card
   - Sender is immediately protected

2. **During Scanning**:
   - Kept senders are automatically skipped
   - Indicated with a 🛡️ shield icon

### Managing Keep List

1. View current keep list in scan results
2. Remove senders:
   - Find the kept sender (marked with shield)
   - Click **"Remove from Keep List"**
   - Confirm the action

### Keep List Best Practices

- Add important business contacts
- Include work-related newsletters
- Protect financial institutions
- Keep government communications

## AI Features

### AI Classification

The AI system uses advanced machine learning to classify emails:

- **UNSUBSCRIBABLE**: Promotional, marketing, newsletters
- **IMPORTANT**: Personal, business, transactional

### Confidence Levels

- **High (90%+)**: Very certain classification
- **Medium (70-89%)**: Likely correct
- **Low (<70%)**: Less certain, review manually

### Adjusting AI Settings

1. **Enable/Disable AI**:
   ```
   □ Use AI Classification
   ```

2. **Confidence Threshold**:
   ```
   AI Confidence: [====|==] 75%
   ```
   - Lower: More aggressive (finds more promotional)
   - Higher: More conservative (fewer false positives)

### AI Indicators

Look for these AI indicators:
- 🤖 **AI Active**: Shows when AI is analyzing
- 📊 **Confidence Score**: Percentage certainty
- 🎯 **Classification**: Promotional or Important

## Batch Operations

### Select Multiple Items

1. **Select All**:
   - Click checkbox in header
   - Selects all visible items

2. **Individual Selection**:
   - Click checkboxes on sender cards
   - Counter shows selected count

### Batch Actions

1. **Unsubscribe Multiple**:
   ```
   [Unsubscribe Selected (5)]
   ```
   - Processes all selected senders
   - Shows progress for each

2. **Delete Emails**:
   ```
   [Delete Selected Emails]
   ```
   - Moves all emails to trash
   - Confirms before action

### Batch Operation Tips

- Review selections before confirming
- Start with small batches
- Monitor progress indicators
- Check results after completion

## Settings and Preferences

### Theme Settings

The application automatically detects your system theme:
- 🌞 **Light Mode**: Clean, bright interface
- 🌙 **Dark Mode**: Easy on the eyes

### Scan Preferences

Customize default scan behavior:
- Default time period
- Maximum emails to scan
- AI enabled by default
- Default confidence threshold

### Performance Settings

- **Cache Duration**: How long to store results
- **API Rate**: Adjust for slower connections
- **Batch Size**: Number of items per operation

## Tips and Best Practices

### Optimal Scanning Strategy

1. **Start Conservative**:
   - Use 85% AI confidence initially
   - Review results carefully
   - Adjust threshold based on accuracy

2. **Regular Maintenance**:
   - Scan monthly for best results
   - Keep your Keep List updated
   - Review AI classifications

3. **Batch Wisely**:
   - Group similar senders
   - Unsubscribe in batches of 10-20
   - Verify important senders first

### Safety Checks

Before unsubscribing:
- ✓ Review sender name and email
- ✓ Check email count
- ✓ Verify it's truly promotional
- ✓ Consider adding to Keep List instead

### Performance Optimization

- **Limit Scan Size**: Start with 100 emails
- **Use Time Filters**: Scan recent emails first
- **Enable Caching**: Faster repeated scans
- **Close Other Tabs**: Reduce browser memory

## Common Use Cases

### 1. Initial Inbox Cleanup

**Scenario**: First-time user with cluttered inbox

**Steps**:
1. Set AI confidence to 80%
2. Scan last 180 days
3. Review all results
4. Add important senders to Keep List
5. Batch unsubscribe from obvious promotional
6. Handle edge cases manually

### 2. Monthly Maintenance

**Scenario**: Regular inbox hygiene

**Steps**:
1. Scan last 30 days
2. Use previous AI confidence
3. Quick review of new senders
4. Unsubscribe from unwanted
5. Update Keep List as needed

### 3. Aggressive Cleanup

**Scenario**: Maximum unsubscribe rate

**Steps**:
1. Set AI confidence to 65%
2. Scan all time
3. Sort by email count
4. Protect critical senders first
5. Batch unsubscribe liberally
6. Monitor for false positives

### 4. Conservative Approach

**Scenario**: Minimize false positives

**Steps**:
1. Set AI confidence to 90%
2. Scan last 90 days
3. Only unsubscribe from 95%+ confidence
4. Manually review 80-94% range
5. Extensive Keep List usage

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus search |
| `Space` | Select/deselect item |
| `Ctrl+A` | Select all |
| `Enter` | Perform default action |
| `Esc` | Close dialogs |

## Troubleshooting Common Issues

### "No emails found"
- Check your internet connection
- Verify you're signed in
- Try a shorter time period
- Check Gmail API access

### "AI not available"
- Refresh the page
- Check browser console
- Verify model files exist
- Contact support

### "Unsubscribe failed"
- Check the error message
- Try manual method
- Verify email is recent
- Some senders block automated unsubscribe

## Privacy and Security

### Your Data is Safe

- ✅ All processing happens locally
- ✅ No emails sent to external servers
- ✅ AI model runs on your machine
- ✅ Secure OAuth authentication
- ✅ Session data encrypted

### Best Practices

1. **Sign out** when done
2. **Don't share** your session
3. **Review permissions** periodically
4. **Use HTTPS** in production
5. **Keep software** updated

## Getting Help

If you need assistance:
1. Check the [FAQ](../troubleshooting/FAQ.md)
2. Review error messages
3. Check browser console (F12)
4. Report issues on GitHub

---

Happy unsubscribing! 🎉 Take control of your inbox with Gmail Unsubscriber.