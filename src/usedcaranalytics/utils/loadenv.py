from usedcaranalytics.utils.getpath import get_path
from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_env():
    """
    Abstraction of loading environment variables in main ETL script.
    
    Returns:
        tuple: Contains (client id, client secret, reddit password, user agent,
        reddit username).
    
    Notes:
        Loads environment variables from .env file.
    """
    # Load environment variables using absolute .env path
    env_path = get_path(Path(__file__), subdir='config', target_file='.env')
    
    logger.info('Loading environment variables containing Reddit API keys and OAuth login credentials.')
    logger.debug('.env loaded from %s', str(env_path))
    
    load_dotenv(env_path)
    
    # Import credentials initialized by config.api module
    from usedcaranalytics.config.api_config import PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME 
    
    # Return tuple in same arrangement as praw.Reddit constructor kwargs
    return PRAW_ID, PRAW_SECRET, PRAW_PASSWORD, PRAW_USER_AGENT, PRAW_USERNAME 