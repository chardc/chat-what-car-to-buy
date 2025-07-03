import logging
import datetime as dt
from usedcaranalytics.utils.getpath import get_repo_root

def setup_logging(
    level=logging.DEBUG, output_to_file: bool=True, output_to_console: bool=False
    ):
    """
    Setup the logger globally throughout the project from entry point.
    Args:
        - level: Logging levels (logging.INFO, logging.DEBUG, etc.).
        - output_to_file: If true, saves log to a *.log file in root/data/logs.
        - output_to_console: If true, prints log to system console.
    Returns:
        - Configured global logger.
        - *.log file and log stream, if set to True.
    """
    # Make log dir if nonexistent
    logs_dir = get_repo_root() / 'data/logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    # File format: /project_root/data/logs/etl-20250630-103000.log
    logging_fp = logs_dir / f'usedcaranalytics-'f'{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.log'
    
    # Setup handlers for logger
    handlers = []
    if output_to_console:
        handlers.append(logging.StreamHandler())
    if output_to_file:
        handlers.append(logging.FileHandler(filename=logging_fp, encoding='utf-8'))
    
    # Configure logging
    logging.basicConfig(
        encoding='utf-8', level=level, handlers=handlers,
        format='%(levelname)s | %(asctime)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ' #ISO 8601 for UTC datetime
        )
    
if __name__ == '__main__':
    setup_logging(output_to_file=False, output_to_console=True)
    logger = logging.getLogger(__name__)
    logger.info(f'This is a test log from %s', __name__)