import logging
import inspect

def get_log_level(level_str: str) -> int:
    """Convert a string to a logging level. Defaults to INFO if invalid."""
    level_str = level_str.upper()
    return getattr(logging, level_str, logging.INFO)


def setup_logger(logging_level_str: str = "info",
                 logger: logging.Logger = None,
                 external_level: int = logging.WARNING) -> logging.Logger:
    """
    Set up a logger for the calling module.

    Args:
        logger (logging.Logger, optional): Logger instance to configure.
        logging_level_str (str, optional): Logging level string (e.g., "debug").
        external_level (int, optional): Level for external modules like SchNetPack.

    Returns:
        logging.Logger: Configured logger.
    """
    if logger is None:
        # Get the name of the calling module
        caller_name = inspect.stack()[1].frame.f_globals["__name__"]
        logger = logging.getLogger(caller_name)

    logger.propagate = False  # Avoid double logging from root logger

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set level
    logging_level = get_log_level(logging_level_str)
    logger.setLevel(logging_level)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Silence external libraries
    for external_module in ["schnetpack"]:
        logging.getLogger(external_module).setLevel(external_level)

    return logger
