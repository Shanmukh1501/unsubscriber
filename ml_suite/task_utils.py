"""
Utilities for task status management and logging in the ML suite.

This module provides functions and classes for:
- Managing task status files for long-running operations
- Logging both to Flask app logs and status files
- Standardized error handling for AI operations
- Progress reporting for data preparation and model training

These utilities ensure that AI operations have proper status tracking,
enabling the frontend to display progress and results to users.
"""

import json
import time
import uuid
import datetime
import traceback
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple


def get_current_timestamp() -> str:
    """Returns ISO format timestamp for current time."""
    return datetime.datetime.now().isoformat()


def get_current_timestamp_log_prefix() -> str:
    """Returns a formatted timestamp string for log entries."""
    return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"


def initialize_task_status_file(status_file_path: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    """Initialize a new task status file with default values.
    
    Args:
        status_file_path: Path to the status file
        task_id: Optional unique ID for the task (UUID generated if None)
        
    Returns:
        Dict containing the initialized status data
    """
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    status_data = {
        "status": "pending",
        "message": "Task initialized and pending execution",
        "progress": 0.0,
        "log": [],
        "task_id": task_id,
        "start_time": get_current_timestamp()
    }
    
    # Ensure the directory exists
    dir_path = os.path.dirname(status_file_path)
    if dir_path:  # Only create directory if dirname returns a non-empty string
        os.makedirs(dir_path, exist_ok=True)
    
    with open(status_file_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    return status_data


def update_task_status(
    status_file_path: str,
    status: Optional[str] = None,
    message: Optional[str] = None,
    log_entry: Optional[str] = None,
    log_level: str = "info",
    progress: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Update a task status file with new information.
    
    Args:
        status_file_path: Path to the status file
        status: Optional new status ('pending', 'in_progress', 'completed', 'failed')
        message: Optional status message
        log_entry: Optional log message to add to the log list
        log_level: Log level ('info', 'warning', 'error', 'debug')
        progress: Optional progress value (0.0 to 1.0)
        result: Optional result data (for completed tasks)
        error: Optional error data (for failed tasks)
        
    Returns:
        Dict containing the updated status data
    """
    try:
        with open(status_file_path, 'r') as f:
            status_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, initialize it
        status_data = initialize_task_status_file(status_file_path)
    
    # Update fields if provided
    if status is not None:
        status_data["status"] = status
        # If status is completed or failed, set end_time
        if status in ("completed", "failed"):
            status_data["end_time"] = get_current_timestamp()
    
    if message is not None:
        status_data["message"] = message
    
    if log_entry is not None:
        if "log" not in status_data:
            status_data["log"] = []
        status_data["log"].append({
            "timestamp": get_current_timestamp(),
            "level": log_level,
            "message": log_entry
        })
    
    if progress is not None:
        status_data["progress"] = max(0.0, min(1.0, float(progress)))  # Ensure between 0-1
    
    if result is not None and status_data.get("status") == "completed":
        status_data["result"] = result
    
    if error is not None and status_data.get("status") == "failed":
        status_data["error"] = error
    
    # Write updated status back to file
    with open(status_file_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    return status_data


def get_task_status(status_file_path: str) -> Dict[str, Any]:
    """Read and return the current task status.
    
    Args:
        status_file_path: Path to the status file
        
    Returns:
        Dict containing the current status data
    """
    try:
        with open(status_file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If status file doesn't exist or is invalid, return a default status
        return {
            "status": "unknown",
            "message": "Task status unknown or not initialized",
            "progress": 0.0,
            "log": []
        }


def log_task_error(status_file_path: str, error: Exception, message: str = "Task failed due to an error") -> Dict[str, Any]:
    """Log an error to the task status file.
    
    Args:
        status_file_path: Path to the status file
        error: The exception that occurred
        message: Human-readable error message
        
    Returns:
        Dict containing the updated status data
    """
    error_info = {
        "type": error.__class__.__name__,
        "message": str(error),
        "traceback": traceback.format_exc()
    }
    
    return update_task_status(
        status_file_path=status_file_path,
        status="failed",
        message=message,
        log_entry=f"ERROR: {message} - {error}",
        log_level="error",
        error=error_info
    )


class AiTaskLogger:
    """Logger for AI tasks that updates both the application logger and task status file.
    
    This logger provides methods for:
    - Logging info, warning, and error messages
    - Updating task progress
    - Marking tasks as started, completed, or failed
    - Ensuring consistent logging across both Flask app logs and task status files
    """
    
    def __init__(self, 
                 app_logger: logging.Logger, 
                 status_file_path: str, 
                 task_id: Optional[str] = None):
        """Initialize the task logger.
        
        Args:
            app_logger: Flask application logger
            status_file_path: Path to the task status file
            task_id: Optional unique ID for the task (UUID generated if None)
        """
        self.app_logger = app_logger
        self.status_file_path = status_file_path
        self.task_id = task_id or str(uuid.uuid4())
        self.short_task_id = self.task_id[:8]  # First 8 chars for log readability
        
        # Initialize the status file
        initialize_task_status_file(status_file_path, self.task_id)
    
    def info(self, message: str, update_progress: Optional[float] = None) -> None:
        """Log an info message and optionally update progress.
        
        Args:
            message: The message to log
            update_progress: Optional progress value (0.0 to 1.0)
        """
        self.app_logger.info(f"[AI Task {self.short_task_id}] {message}")
        update_task_status(
            self.status_file_path,
            log_entry=message,
            log_level="info",
            progress=update_progress
        )
    
    def warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: The warning message to log
        """
        self.app_logger.warning(f"[AI Task {self.short_task_id}] {message}")
        update_task_status(
            self.status_file_path,
            log_entry=message,
            log_level="warning"
        )
    
    def error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log an error message and optionally the exception details.
        
        Args:
            message: The error message to log
            error: Optional exception that caused the error
        """
        if error:
            self.app_logger.error(f"[AI Task {self.short_task_id}] {message}: {error}", exc_info=True)
            log_task_error(self.status_file_path, error, message)
        else:
            self.app_logger.error(f"[AI Task {self.short_task_id}] {message}")
            update_task_status(
                self.status_file_path,
                log_entry=message,
                log_level="error"
            )
    
    def start_task(self, message: str = "Task started") -> None:
        """Mark the task as started.
        
        Args:
            message: Optional message describing the task start
        """
        self.app_logger.info(f"[AI Task {self.short_task_id}] {message}")
        update_task_status(
            self.status_file_path,
            status="in_progress",
            message=message,
            log_entry=message,
            log_level="info",
            progress=0.0
        )
    
    def complete_task(self, message: str = "Task completed successfully", result: Optional[Dict[str, Any]] = None) -> None:
        """Mark the task as completed.
        
        Args:
            message: Optional completion message
            result: Optional result data to store
        """
        self.app_logger.info(f"[AI Task {self.short_task_id}] {message}")
        update_task_status(
            self.status_file_path,
            status="completed",
            message=message,
            log_entry=message,
            log_level="info",
            progress=1.0,
            result=result
        )
    
    def fail_task(self, message: str, error: Optional[Exception] = None) -> None:
        """Mark the task as failed.
        
        Args:
            message: Failure message
            error: Optional exception that caused the failure
        """
        if error:
            self.app_logger.error(f"[AI Task {self.short_task_id}] {message}: {error}", exc_info=True)
            log_task_error(self.status_file_path, error, message)
        else:
            self.app_logger.error(f"[AI Task {self.short_task_id}] {message}")
            update_task_status(
                self.status_file_path,
                status="failed",
                message=message,
                log_entry=message,
                log_level="error"
            )
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update the task progress.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
        """
        if message:
            self.app_logger.info(f"[AI Task {self.short_task_id}] {message} (Progress: {progress:.1%})")
            update_task_status(
                self.status_file_path,
                message=message,
                log_entry=message,
                log_level="info",
                progress=progress
            )
        else:
            update_task_status(
                self.status_file_path,
                progress=progress
            )