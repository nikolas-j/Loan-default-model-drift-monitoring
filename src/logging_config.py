import logging
import sys
from pathlib import Path
from src.config import config


def setup_logging(name: str = __name__, log_to_file: bool = True) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_to_file: Whether to also log to a file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger
    
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console gets INFO and above
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{name.replace('.', '_')}.log",
            mode='a'  # Append mode
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exception: The exception to log
        context: Additional context about where the exception occurred
    """
    if context:
        logger.error(f"{context}: {str(exception)}", exc_info=True)
    else:
        logger.error(str(exception), exc_info=True)
