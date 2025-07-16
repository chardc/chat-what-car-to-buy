import logging
import datetime as dt
from chatwhatcartobuy.utils.getpath import get_repo_root

def setup_logging(
    level=logging.DEBUG, file_prefix: str=None, target_dir: str='data/logs',
    output_to_file: bool=False, output_to_console: bool=False
    ):
    """
    Setup the logger globally throughout the project from entry point.
    
    Args:
        level (Default=logging.DEBUG): Logging level.
        file_prefix: Prefix of log file name. Should be descriptive, preferably pertains
        to the main script which initialized logging.
        target_dir (Default='data/logs'): Directory within project root to store logs.
        output_to_file (Default=False): If true, saves log to a *.log file in 
        project_root/data/logs.
        output_to_console (Default=False): If true, prints log to system console.
    
    Notes:
        Configures the global logger. Outputs *.log file to project_root/data/logs
        and streams the logs to console when respective args are enabled.
    """
    # Input validation
    if output_to_file:
        if file_prefix is None:
            raise ValueError('File prefix must be provided when output_to_file is True. Ex. prefix: "etl-pipeline"; file_out: "etl-pipeline-Y-m-d.log"')
        if target_dir is None:
            raise ValueError('Target directory must be provided when output_to_file is True.')
    
    # Make log dir if nonexistent, and group all logs in the same date dir
    logs_dir = get_repo_root() / target_dir / f'{dt.datetime.now():%Y-%m-%d}'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # File format: /project_root/data/logs/etl-20250630-103000.log
    logging_fp = logs_dir / f'{file_prefix}-{dt.datetime.now():%Y%m%d-%H%M%S}.log'
    
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
        datefmt='%Y-%m-%d %H:%M:%S' # System time
        )
    
if __name__ == '__main__':
    setup_logging(output_to_file=False, output_to_console=True)
    logger = logging.getLogger(__name__)
    logger.info(f'This is a test log from %s', __name__)