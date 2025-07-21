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
        
@pytest.fixture(autouse=True, scope='function')
def patch_setup_logging(monkeypatch):
    """Monkeypatch setup_logging to never output to file during pytest runs."""
    
    from chatwhatcartobuy.config import logging_config
    
    # Always override output_to_file and output_to_console to False in tests
    def dummy_setup_logging():
        return logging_config.setup_logging()
    
    monkeypatch.setattr(logging_config, "setup_logging", dummy_setup_logging)