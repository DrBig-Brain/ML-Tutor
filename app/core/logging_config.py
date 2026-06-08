import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import traceback


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    Useful for production environments and log aggregation tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the record
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        
        # Add request ID if present (for tracking requests across services)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        
        # Add user context if present
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Custom formatter with colors for development console output.
    Makes logs easier to read during development.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Build log message
        log_message = (
            f"{color}{timestamp}{reset} | "
            f"{color}{record.levelname:<8}{reset} | "
            f"{color}{record.name}{reset} | "
            f"{record.getMessage()}"
        )
        
        # Add extra data if present
        if hasattr(record, "extra_data"):
            log_message += f"\n  Extra: {record.extra_data}"
        
        # Add exception info if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message


class ContextFilter(logging.Filter):
    """
    Filter that adds contextual information to log records.
    Can be used to add request IDs, user IDs, etc.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record"""
        for key, value in self._context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    debug: bool = False,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        debug: Enable debug logging if True
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        json_format: Use JSON formatter for production
        max_file_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of rotated log files to keep
    """
    
    # Determine log level
    if log_level:
        level = getattr(logging, log_level.upper())
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = ColoredConsoleFormatter() if not log_file else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent unlimited growth
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers to be less verbose
    _configure_third_party_loggers(debug)
    
    # Log startup message
    logger = get_logger(__name__)
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def _configure_third_party_loggers(debug: bool = False) -> None:
    """
    Configure log levels for third-party libraries to reduce noise.
    
    Args:
        debug: If True, keep third-party debug logs; otherwise suppress them
    """
    
    # Libraries that are too verbose at DEBUG level
    verbose_loggers = [
        "urllib3",
        "requests",
        "chromadb",
        "httpx",
        "httpcore",
        "langchain",
        "openai",
        "asyncio"
    ]
    
    for logger_name in verbose_loggers:
        logger = logging.getLogger(logger_name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
    
    # Keep FastAPI/uvicorn logs at INFO level for development
    if not debug:
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.INFO)


class LoggerContext:
    """
    Context manager for adding temporary context to logs.
    
    Usage:
        with LoggerContext(request_id="123", user_id="456"):
            logger.info("Processing request")  # Will include context
    
    Or as a decorator:
        @LoggerContext(request_id="123")
        def my_function():
            logger.info("Inside function")
    """
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.original_filters = []
    
    def __enter__(self):
        """Add context filter to root logger"""
        context_filter = ContextFilter(self.context)
        root_logger = logging.getLogger()
        self.original_filters = root_logger.filters[:]
        root_logger.addFilter(context_filter)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove context filter"""
        root_logger = logging.getLogger()
        root_logger.filters = self.original_filters
    
    def __call__(self, func):
        """Decorator usage"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name (typically __name__)
        **context: Additional context to add to all log messages
    
    Returns:
        Configured logger instance
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Hello world")
        
        # With context
        logger = get_logger(__name__, request_id="123")
        logger.info("Request processing")  # Will include request_id
    """
    logger = logging.getLogger(name)
    
    if context:
        # Add context filter to this specific logger
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


class PerformanceLogger:
    """
    Helper class for logging performance metrics.
    
    Usage:
        perf = PerformanceLogger("database_query")
        result = query_database()
        perf.log()  # Logs execution time
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing"""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log execution time on exit"""
        import time
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.info(
                f"Performance: {self.operation_name} completed in {duration_ms:.2f}ms",
                extra={"extra_data": {
                    "operation": self.operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "success": exc_type is None
                }}
            )
    
    def log(self, duration_ms: Optional[float] = None):
        """Log execution time manually"""
        if duration_ms is None and self.start_time:
            import time
            duration_ms = (time.time() - self.start_time) * 1000
        
        self.logger.info(
            f"Performance: {self.operation_name} took {duration_ms:.2f}ms",
            extra={"extra_data": {
                "operation": self.operation_name,
                "duration_ms": round(duration_ms, 2) if duration_ms else None
            }}
        )


# Convenience function for debugging
def debug_logger(logger: logging.Logger, data: Dict[str, Any], message: str = "Debug info") -> None:
    """
    Helper function to log debug data in a structured way.
    
    Args:
        logger: Logger instance
        data: Dictionary of data to log
        message: Log message
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            message,
            extra={"extra_data": data}
        )


# Example of how to use the logging configuration
if __name__ == "__main__":
    # Test the logging setup
    setup_logging(debug=True)
    
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("An error occurred", exc_info=True)
    
    # Test context manager
    with LoggerContext(request_id="test-123", user_id="user-456"):
        logger.info("This log has context")
    
    # Test performance logger
    import time
    with PerformanceLogger("test_operation", logger):
        time.sleep(0.1)