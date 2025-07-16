import logging
import pytest

@pytest.fixture(autouse=True, scope='session')
def disable_file_logging():
    """Disable file logging during testing session."""
    
    # Get all existing loggers
    loggers = [logging.getLogger()] # Root logger
    loggers.extend([logging.getLogger(name) for name in logging.root.manager.loggerDict]) # Named loggers
    
    # Store original loggers for teardown after test
    original_handlers = {logger: list(logger.handlers) for logger in loggers}
    
    # Remove FileHandler to disable file logging
    for logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

    yield
    
    # Restore original handlers
    for logger, handlers in original_handlers.items():
        logger.handlers = handlers
    